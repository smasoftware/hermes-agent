#!/usr/bin/env python3
"""
Deskmote Tool Module — tmux workspace awareness for Deskmote integration.

Gives the Hermes agent visibility into the user's Deskmote workspace:
- List active tmux sessions (Deskmote terminal tabs)
- Capture visible output from a tmux pane
- Send keystrokes to a tmux pane

All tools run locally via subprocess (Hermes runs on the same host as tmux).
"""

import json
import shutil
import subprocess
import logging

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def _run_tmux(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a tmux command and return (returncode, stdout, stderr)."""
    cmd = ["tmux"] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "tmux command timed out"
    except FileNotFoundError:
        return 1, "", "tmux is not installed"


# ---------------------------------------------------------------------------
# Tool: deskmote_list_sessions
# ---------------------------------------------------------------------------

def handle_list_sessions(args: dict, **kw) -> str:
    """List all active tmux sessions with their windows."""
    fmt = "#{session_name}\t#{session_windows}\t#{session_created}\t#{session_attached}"
    rc, stdout, stderr = _run_tmux(["list-sessions", "-F", fmt])
    if rc != 0:
        if "no server running" in stderr or "no sessions" in stderr.lower():
            return tool_result(sessions=[], message="No tmux sessions active.")
        return tool_error(f"tmux list-sessions failed: {stderr.strip()}")

    sessions = []
    for line in stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            sessions.append({
                "name": parts[0],
                "windows": int(parts[1]),
                "created": parts[2],
                "attached": parts[3] == "1",
            })

    return tool_result(sessions=sessions)


LIST_SESSIONS_SCHEMA = {
    "name": "deskmote_list_sessions",
    "description": (
        "List all active tmux sessions on this host. Each session corresponds "
        "to a Deskmote terminal tab. Returns session names, window counts, and "
        "whether each session is currently attached."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Tool: deskmote_read_pane
# ---------------------------------------------------------------------------

def handle_read_pane(args: dict, **kw) -> str:
    """Capture visible output from a tmux pane."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required (tmux session name or session:window.pane)")

    lines = args.get("lines", 50)
    if lines < 1:
        lines = 50
    if lines > 500:
        lines = 500

    # capture-pane -p prints to stdout, -S sets start line (negative = scrollback)
    capture_args = ["capture-pane", "-t", target, "-p", "-S", str(-lines)]
    rc, stdout, stderr = _run_tmux(capture_args)
    if rc != 0:
        return tool_error(f"Failed to capture pane '{target}': {stderr.strip()}")

    # Trim trailing empty lines
    content = stdout.rstrip("\n")

    return tool_result(target=target, lines_captured=lines, content=content)


READ_PANE_SCHEMA = {
    "name": "deskmote_read_pane",
    "description": (
        "Capture the visible terminal output from a tmux pane. Use this to see "
        "what's currently on screen in a Deskmote terminal tab — command output, "
        "errors, logs, etc. Use deskmote_list_sessions first to find session names."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "tmux target: session name (e.g. 'deskmote-abc123') or "
                    "session:window.pane (e.g. 'deskmote-abc123:0.0'). "
                    "Use deskmote_list_sessions to find available sessions."
                ),
            },
            "lines": {
                "type": "integer",
                "description": "Number of lines to capture from scrollback (default 50, max 500).",
                "default": 50,
            },
        },
        "required": ["target"],
    },
}


# ---------------------------------------------------------------------------
# Tool: deskmote_send_keys
# ---------------------------------------------------------------------------

def handle_send_keys(args: dict, **kw) -> str:
    """Send keystrokes to a tmux pane."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required (tmux session name)")

    keys = args.get("keys", "")
    if not keys:
        return tool_error("keys is required (text or keystrokes to send)")

    send_enter = args.get("send_enter", True)

    # Send the keys
    send_args = ["send-keys", "-t", target, keys]
    if send_enter:
        send_args.append("Enter")

    rc, stdout, stderr = _run_tmux(send_args)
    if rc != 0:
        return tool_error(f"Failed to send keys to '{target}': {stderr.strip()}")

    return tool_result(
        target=target,
        keys_sent=keys,
        enter_sent=send_enter,
        message=f"Keys sent to {target}",
    )


SEND_KEYS_SCHEMA = {
    "name": "deskmote_send_keys",
    "description": (
        "Send keystrokes to a tmux pane — like typing into a Deskmote terminal "
        "tab. Use this to run commands, respond to prompts, or interact with "
        "programs running in a specific session. By default, Enter is sent after "
        "the keys."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "tmux target: session name or session:window.pane. "
                    "Use deskmote_list_sessions to find available sessions."
                ),
            },
            "keys": {
                "type": "string",
                "description": (
                    "Text or keystrokes to send. For special keys use tmux "
                    "key names: 'C-c' (Ctrl+C), 'C-d' (Ctrl+D), 'Up', 'Down', "
                    "'Tab', 'Escape', etc."
                ),
            },
            "send_enter": {
                "type": "boolean",
                "description": "Whether to press Enter after the keys (default true).",
                "default": True,
            },
        },
        "required": ["target", "keys"],
    },
}


# ---------------------------------------------------------------------------
# Tool: deskmote_get_workspace
# ---------------------------------------------------------------------------

def handle_get_workspace(args: dict, **kw) -> str:
    """Get the full Deskmote workspace (connections, sessions, tabs) from the API."""
    import os
    import httpx

    api_key = os.getenv("DESKMOTE_API_KEY", "")
    api_url = os.getenv("DESKMOTE_API_URL", "https://api.deskmote.io")

    if not api_key:
        return tool_error("DESKMOTE_API_KEY not set — configure in Deskmote Settings > AI Assistant")

    try:
        resp = httpx.get(
            f"{api_url}/api/v1/customer/hermes/workspace",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return tool_error(f"API error {resp.status_code}: {resp.text[:200]}")
        return tool_result(**resp.json())
    except Exception as e:
        return tool_error(f"Failed to query workspace: {e}")


GET_WORKSPACE_SCHEMA = {
    "name": "deskmote_get_workspace",
    "description": (
        "Get the full Deskmote workspace from the cloud API — all connections (hosts), "
        "sessions, and tabs with their friendly names. Use this to map tmux session "
        "names to Host → Session → Shell names for clearer responses. "
        "Returns connections and workspace tree."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Tool: deskmote_check_prompts
# ---------------------------------------------------------------------------

# Patterns that indicate a terminal needs attention
_PROMPT_PATTERNS = [
    r'\[Y/n\]', r'\[y/N\]', r'\(yes/no\)', r'\(y/n\)',
    r'Continue\?', r'Proceed\?', r'Are you sure\?',
    r'Press any key', r'Press Enter', r'Press RETURN',
    r'Password:', r'password:', r'passphrase',
    r'Enter .*:', r'Confirm .*:',
    r'ERROR', r'FAILED', r'FATAL', r'panic:',
    r'Permission denied', r'Access denied',
    r'Do you want to', r'Would you like to',
    r'Overwrite.*\?', r'Replace.*\?', r'Delete.*\?',
]

import re
_PROMPT_RE = re.compile('|'.join(_PROMPT_PATTERNS), re.IGNORECASE)


def handle_check_prompts(args: dict, **kw) -> str:
    """Scan all tmux sessions for pending prompts/errors that need attention."""
    rc, stdout, stderr = _run_tmux(["list-sessions", "-F", "#{session_name}"])
    if rc != 0:
        return tool_result(alerts=[], message="No tmux sessions active.")

    sessions = [s.strip() for s in stdout.strip().splitlines() if s.strip()]
    alerts = []

    for session_name in sessions:
        # Capture last 5 lines of each pane
        cap_rc, cap_out, _ = _run_tmux(["capture-pane", "-t", session_name, "-p", "-S", "-5"])
        if cap_rc != 0 or not cap_out.strip():
            continue

        last_lines = cap_out.strip().split("\n")[-5:]
        text = "\n".join(last_lines)

        matches = _PROMPT_RE.findall(text)
        if matches:
            alerts.append({
                "tmux_session": session_name,
                "prompt": matches[0],
                "context": text.strip(),
            })

    if not alerts:
        return tool_result(alerts=[], message="No pending prompts or errors found.")

    return tool_result(
        alerts=alerts,
        message=f"Found {len(alerts)} session(s) needing attention.",
    )


CHECK_PROMPTS_SCHEMA = {
    "name": "deskmote_check_prompts",
    "description": (
        "Scan all tmux sessions for pending prompts, confirmation dialogs, "
        "password requests, or errors that need user attention. Returns a list "
        "of alerts with the tmux session name and the prompt text. Use "
        "deskmote_get_workspace to map tmux names to friendly Host/Session/Shell names."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_deskmote_available() -> bool:
    """Deskmote tools require tmux to be installed on the host."""
    return shutil.which("tmux") is not None


def check_workspace_available() -> bool:
    """Workspace tool requires DESKMOTE_API_KEY."""
    import os
    return bool(os.getenv("DESKMOTE_API_KEY"))


# ---------------------------------------------------------------------------
# Register all tools
# ---------------------------------------------------------------------------

registry.register(
    name="deskmote_list_sessions",
    toolset="deskmote",
    schema=LIST_SESSIONS_SCHEMA,
    handler=handle_list_sessions,
    check_fn=check_deskmote_available,
    description="List active tmux sessions (Deskmote terminal tabs)",
    emoji="🖥️",
)

registry.register(
    name="deskmote_read_pane",
    toolset="deskmote",
    schema=READ_PANE_SCHEMA,
    handler=handle_read_pane,
    check_fn=check_deskmote_available,
    description="Capture visible output from a tmux pane",
    emoji="👁️",
)

registry.register(
    name="deskmote_send_keys",
    toolset="deskmote",
    schema=SEND_KEYS_SCHEMA,
    handler=handle_send_keys,
    check_fn=check_deskmote_available,
    description="Send keystrokes to a tmux pane",
    emoji="⌨️",
)

registry.register(
    name="deskmote_get_workspace",
    toolset="deskmote",
    schema=GET_WORKSPACE_SCHEMA,
    handler=handle_get_workspace,
    check_fn=check_workspace_available,
    description="Get Deskmote workspace (hosts, sessions, tabs) from API",
    emoji="🌐",
)

registry.register(
    name="deskmote_check_prompts",
    toolset="deskmote",
    schema=CHECK_PROMPTS_SCHEMA,
    handler=handle_check_prompts,
    check_fn=check_deskmote_available,
    description="Scan all sessions for pending prompts or errors",
    emoji="🔔",
)
