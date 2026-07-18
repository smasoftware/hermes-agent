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


_DEFAULT_DESKMOTE_TOOLSETS = ["deskmote", "terminal", "file", "web", "skills", "memory", "browser"]


def _resolve_enabled_toolsets() -> list[str]:
    """Resolve toolsets for the Deskmote platform from config.yaml.

    Honours ``platform_toolsets.deskmote`` verbatim (the Deskmote app writes
    it via its Hermes setup flow) so toolsets like ``browser`` can be enabled
    per-host without a code change. Reads the raw config key instead of
    ``_get_platform_tools`` because that helper filters to CONFIGURABLE_TOOLSETS
    and would silently drop registry-registered toolsets like ``deskmote``.
    """
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        toolsets = (config.get("platform_toolsets") or {}).get("deskmote")
        if isinstance(toolsets, list) and toolsets:
            return [str(ts) for ts in toolsets]
    except Exception:
        logger.warning("Failed to read platform_toolsets.deskmote; using defaults", exc_info=True)
    return list(_DEFAULT_DESKMOTE_TOOLSETS)


def _resolve_runtime_kwargs() -> tuple[str, dict]:
    """Resolve LLM provider credentials from Hermes config."""
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = load_config()
    model_config = config.get("model", {})
    if isinstance(model_config, str):
        model_config = {"default": model_config}

    model = model_config.get("default", "anthropic/claude-opus-4-20250514")
    provider = model_config.get("provider", "auto")

    runtime = resolve_runtime_provider(
        requested=provider if provider != "auto" else None,
    )

    return model, {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
    }


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
    _conversation_history: dict[str, list] = {}
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
            enabled_toolsets=_resolve_enabled_toolsets(),
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
        cwd: str = ""  # Working directory for OpenWolf context

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

    def _load_openwolf_context(cwd: str) -> Optional[str]:
        """Load OpenWolf context files from a project directory."""
        if not cwd:
            return None

        wolf_dir = Path(cwd) / ".wolf"
        if not wolf_dir.is_dir():
            return None

        context_parts = []

        # Read cerebrum (conventions, learnings, do-not-repeat)
        cerebrum = wolf_dir / "cerebrum.md"
        if cerebrum.exists():
            try:
                content = cerebrum.read_text(errors="replace")[:4000]
                context_parts.append(f"## Project Conventions (cerebrum.md)\n{content}")
            except Exception:
                pass

        # Read anatomy (file descriptions)
        anatomy = wolf_dir / "anatomy.md"
        if anatomy.exists():
            try:
                content = anatomy.read_text(errors="replace")[:6000]
                context_parts.append(f"## Project Structure (anatomy.md)\n{content}")
            except Exception:
                pass

        # Read memory (session log)
        memory = wolf_dir / "memory.md"
        if memory.exists():
            try:
                content = memory.read_text(errors="replace")[-2000:]  # Last 2000 chars
                context_parts.append(f"## Recent Activity (memory.md)\n{content}")
            except Exception:
                pass

        if not context_parts:
            return None

        return (
            "# OpenWolf Project Context\n"
            f"Working directory: {cwd}\n\n"
            + "\n\n".join(context_parts)
        )

    @app.post("/api/v1/chat", dependencies=[Depends(verify_token)])
    async def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session_lock = _get_or_create_lock(req.session_id)
        async with session_lock:
            try:
                agent = _get_agent(req.session_id)

                # Get existing conversation history for this session
                history = _conversation_history.get(req.session_id, [])

                # Build system message with OpenWolf context if available
                system_msg = None
                wolf_context = _load_openwolf_context(req.cwd)
                if wolf_context:
                    system_msg = (
                        "You are Hermes, an AI assistant integrated with Deskmote. "
                        "You have access to the user's project context below. "
                        "Use it to give informed, project-specific answers. "
                        "Follow the conventions and avoid the mistakes listed in Do-Not-Repeat.\n\n"
                        + wolf_context
                    )

                # Run synchronous agent in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: agent.run_conversation(
                        user_message=req.message,
                        system_message=system_msg,
                        conversation_history=history if history else None,
                        task_id=req.session_id,
                    ),
                )

                # Store updated conversation history for next turn
                if result and result.get("messages"):
                    _conversation_history[req.session_id] = result["messages"]

                return ChatResponse(
                    response=result.get("final_response") or "" if result else "",
                    session_id=req.session_id,
                    model=(result.get("model") or "") if result else "",
                    input_tokens=(result.get("input_tokens") or 0) if result else 0,
                    output_tokens=(result.get("output_tokens") or 0) if result else 0,
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

    # ------------------------------------------------------------------
    # Alerts endpoint (prompt/error scanner)
    # ------------------------------------------------------------------
    _active_alerts: list[dict] = []
    _last_scan: float = 0.0
    _SCAN_INTERVAL = 10.0  # seconds

    def _scan_prompts() -> list[dict]:
        """Scan tmux panes for pending prompts. No LLM calls — pure regex."""
        import re
        patterns = [
            r'\[Y/n\]', r'\[y/N\]', r'\(yes/no\)', r'\(y/n\)',
            r'Continue\?', r'Proceed\?', r'Are you sure\?',
            r'Password:', r'password:', r'passphrase',
            r'ERROR', r'FAILED', r'FATAL', r'panic:',
            r'Permission denied', r'Do you want to',
        ]
        prompt_re = re.compile('|'.join(patterns), re.IGNORECASE)

        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []
        except Exception:
            return []

        sessions = [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
        alerts = []
        for name in sessions:
            try:
                cap = subprocess.run(
                    ["tmux", "capture-pane", "-t", name, "-p", "-S", "-5"],
                    capture_output=True, text=True, timeout=5,
                )
                if cap.returncode != 0:
                    continue
                last_lines = cap.stdout.strip().split("\n")[-5:]
                text = "\n".join(last_lines)
                matches = prompt_re.findall(text)
                if matches:
                    alerts.append({
                        "tmux_session": name,
                        "prompt": matches[0],
                        "context": text.strip(),
                        "timestamp": time.time(),
                    })
            except Exception:
                continue
        return alerts

    @app.get("/api/v1/alerts", dependencies=[Depends(verify_token)])
    async def get_alerts():
        """Return pending terminal alerts. Scans at most every 10s."""
        nonlocal _active_alerts, _last_scan
        now = time.time()
        if now - _last_scan >= _SCAN_INTERVAL:
            loop = asyncio.get_event_loop()
            _active_alerts = await loop.run_in_executor(None, _scan_prompts)
            _last_scan = now
        return {"alerts": _active_alerts, "count": len(_active_alerts)}

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
        # Default keep-alive is 5s: macOS URLSession reuses idle connections
        # and POSTs onto the closed socket -> NSURLErrorNetworkConnectionLost
        # (-1005) in the Deskmote app. Keep connections open longer than any
        # realistic think-time between chat messages.
        timeout_keep_alive=75,
    )
