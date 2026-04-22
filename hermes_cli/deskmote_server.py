#!/usr/bin/env python3
"""
Deskmote HTTP Server — thin API layer wrapping AIAgent for Deskmote native apps.

Runs on the host machine (same as tmux sessions) and provides:
- POST /api/v1/chat          Send a message, get AI response
- GET  /api/v1/chat/{id}/history  Get conversation history
- GET  /health               Health check + version info

Auth via Bearer token stored at ~/.hermes/deskmote_token.

Usage:
    hermes deskmote-server                  # Run on default port 7420
    hermes deskmote-server --port 8080      # Custom port
    hermes deskmote-server --host 0.0.0.0   # Bind to all interfaces
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Agent cache config
_AGENT_CACHE_MAX_SIZE = 32
_AGENT_CACHE_IDLE_TTL_SECS = 1800.0  # 30 min idle eviction

# Token file
_TOKEN_FILENAME = "deskmote_token"


def get_token_path() -> Path:
    """Return path to the deskmote auth token file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _TOKEN_FILENAME


def ensure_token() -> str:
    """Read or generate the deskmote auth token."""
    token_path = get_token_path()
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    # Generate new token
    token = secrets.token_hex(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n")
    # Restrict permissions
    token_path.chmod(0o600)
    logger.info("Generated new deskmote auth token at %s", token_path)
    return token


def _resolve_runtime_kwargs() -> tuple[str, dict]:
    """Resolve LLM provider credentials from Hermes config."""
    from hermes_cli.config import load_config
    from hermes_cli.auth import resolve_runtime_provider

    config = load_config()
    model_config = config.get("model", {})
    if isinstance(model_config, str):
        model_config = {"default": model_config}

    model = model_config.get("default", "anthropic/claude-opus-4-20250514")
    provider = model_config.get("provider", "auto")

    runtime = resolve_runtime_provider(
        provider_hint=provider,
        model=model,
        config=config,
    )

    return model, runtime


def create_app(auth_token: str):
    """Create the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="Hermes Deskmote Server",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )

    # ------------------------------------------------------------------
    # Agent cache (LRU, per session_id)
    # ------------------------------------------------------------------
    _agent_cache: OrderedDict[str, tuple] = OrderedDict()
    _agent_locks: dict[str, asyncio.Lock] = {}
    _cache_lock = threading.Lock()

    def _get_or_create_lock(session_id: str) -> asyncio.Lock:
        with _cache_lock:
            if session_id not in _agent_locks:
                _agent_locks[session_id] = asyncio.Lock()
            return _agent_locks[session_id]

    def _get_agent(session_id: str):
        """Get or create an AIAgent for the given session."""
        from run_agent import AIAgent
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home

        with _cache_lock:
            if session_id in _agent_cache:
                agent, created_at = _agent_cache.pop(session_id)
                _agent_cache[session_id] = (agent, time.monotonic())
                return agent

        # Create new agent
        model, runtime_kwargs = _resolve_runtime_kwargs()

        db_path = get_hermes_home() / "state.db"
        session_db = SessionDB(db_path=db_path)

        agent = AIAgent(
            **runtime_kwargs,
            model=model,
            max_iterations=30,
            quiet_mode=True,
            enabled_toolsets=["deskmote", "terminal", "file", "web", "skills", "memory"],
            session_id=session_id,
            platform="deskmote",
            session_db=session_db,
        )

        with _cache_lock:
            _agent_cache[session_id] = (agent, time.monotonic())
            # Evict oldest if over capacity
            while len(_agent_cache) > _AGENT_CACHE_MAX_SIZE:
                evicted_key, (evicted_agent, _) = _agent_cache.popitem(last=False)
                logger.info("Evicting agent for session %s", evicted_key)

        return agent

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------
    async def verify_token(request: Request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = auth_header[7:]
        if not secrets.compare_digest(token, auth_token):
            raise HTTPException(status_code=401, detail="Invalid token")

    # ------------------------------------------------------------------
    # Request/Response models
    # ------------------------------------------------------------------
    class ChatRequest(BaseModel):
        message: str
        session_id: str = "default"

    class ChatResponse(BaseModel):
        response: str
        session_id: str
        model: str = ""
        input_tokens: int = 0
        output_tokens: int = 0

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str = ""
        model: str = ""

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health():
        try:
            from hermes_constants import HERMES_VERSION
            version = HERMES_VERSION
        except Exception:
            version = "unknown"

        model = "unknown"
        try:
            m, _ = _resolve_runtime_kwargs()
            model = m
        except Exception:
            pass

        return HealthResponse(status="ok", version=version, model=model)

    @app.post("/api/v1/chat", dependencies=[Depends(verify_token)])
    async def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session_lock = _get_or_create_lock(req.session_id)
        async with session_lock:
            try:
                agent = _get_agent(req.session_id)
                # Run synchronous agent in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: agent.run_conversation(
                        user_message=req.message,
                        task_id=req.session_id,
                    ),
                )

                return ChatResponse(
                    response=result.get("final_response", ""),
                    session_id=req.session_id,
                    model=result.get("model", ""),
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                )
            except Exception as e:
                logger.error("Chat error for session %s: %s", req.session_id, e)
                raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/chat/{session_id}/history", dependencies=[Depends(verify_token)])
    async def chat_history(session_id: str):
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home

        try:
            db_path = get_hermes_home() / "state.db"
            db = SessionDB(db_path=db_path)
            messages = db.get_messages(session_id)
            db.close()

            return {
                "session_id": session_id,
                "messages": [
                    {
                        "role": m.get("role", ""),
                        "content": m.get("content", ""),
                        "timestamp": m.get("created_at", ""),
                    }
                    for m in (messages or [])
                    if m.get("role") in ("user", "assistant")
                ],
            }
        except Exception as e:
            logger.error("History error for session %s: %s", session_id, e)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(host: str = "0.0.0.0", port: int = 7420):
    """Start the Deskmote HTTP server."""
    import uvicorn

    token = ensure_token()
    logger.info("Deskmote server starting on %s:%d", host, port)
    logger.info("Auth token file: %s", get_token_path())

    app = create_app(auth_token=token)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
