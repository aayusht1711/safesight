#!/usr/bin/env python3
"""
SafeSight AI — Auto-Bootstrapping Engine Manager
===================================================
Shared module that transparently ensures the FastAPI backend is running
before any test script executes API requests.

Workflow:
  1. Active Connection Probe  — quick GET ping to the server.
  2. Auto-Bootstrap           — if offline, spawn uvicorn in background.
  3. Retry Polling             — poll every 0.5s (up to 10 retries / 5s).
  4. Clean Teardown            — atexit + signal handler kills child process.
  5. Failsafe Fallback         — if still offline after retries, returns False.

Usage:
    from tests.engine_manager import ensure_backend

    server_live = ensure_backend(
        host="127.0.0.1",
        port=8000,
        console=console,       # rich.console.Console instance
    )
    if not server_live:
        # fall back to mock pipeline
        ...
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

# ── Module State ────────────────────────────────────────────────────────────
_managed_process: subprocess.Popen | None = None
_already_registered_cleanup: bool = False


# ── Internal Helpers ────────────────────────────────────────────────────────

def _ping_server(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Perform a quick, non-blocking HTTP GET probe to the server root.
    Returns True if the server responds (any HTTP status), False otherwise.
    """
    try:
        import httpx
        resp = httpx.get(f"http://{host}:{port}/", timeout=timeout)
        return True  # Any response means server is alive
    except Exception:
        pass

    # Fallback: try with requests if httpx is unavailable
    try:
        import requests
        resp = requests.get(f"http://{host}:{port}/", timeout=timeout)
        return True
    except Exception:
        pass

    # Fallback: raw socket connect (no HTTP, just TCP)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _spawn_server(host: str, port: int) -> subprocess.Popen | None:
    """
    Spawn uvicorn as a background subprocess.
    Uses the same Python interpreter that's running the test script.

    Returns the Popen handle, or None if launch failed immediately.
    """
    project_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable

    cmd = [
        python_exe, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
    ]

    try:
        # Suppress server stdout/stderr so it doesn't interleave with test output.
        # On Windows, CREATE_NEW_PROCESS_GROUP allows clean termination.
        kwargs: dict = {
            "cwd": str(project_root),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # On Unix, start in its own process group for clean SIGTERM
            kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **kwargs)

        # Quick sanity check — if it crashes immediately, poll() returns exit code
        time.sleep(0.15)
        if proc.poll() is not None:
            return None  # Process died immediately

        return proc

    except Exception:
        return None


def _register_cleanup(proc: subprocess.Popen) -> None:
    """
    Register an atexit handler + signal handler to kill the managed server
    when the test script exits (normally or via CTRL+C).
    """
    global _already_registered_cleanup

    def _cleanup() -> None:
        global _managed_process
        if _managed_process is not None and _managed_process.poll() is None:
            try:
                if sys.platform == "win32":
                    # On Windows, terminate the process tree
                    _managed_process.terminate()
                else:
                    # On Unix, send SIGTERM to the process group
                    os.killpg(os.getpgid(_managed_process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass

            try:
                _managed_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _managed_process.kill()
            finally:
                _managed_process = None

    if not _already_registered_cleanup:
        atexit.register(_cleanup)

        # Handle CTRL+C gracefully
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            _cleanup()
            # Re-raise original handler (or default KeyboardInterrupt)
            if callable(original_sigint) and original_sigint not in (
                signal.SIG_IGN, signal.SIG_DFL
            ):
                original_sigint(signum, frame)
            else:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sigint_handler)
        _already_registered_cleanup = True


# ── Public API ──────────────────────────────────────────────────────────────

def ensure_backend(
    host: str = "127.0.0.1",
    port: int = 8000,
    console: Console | None = None,
    max_retries: int = 10,
    poll_interval: float = 0.5,
) -> bool:
    """
    Ensure the FastAPI backend is running. Returns True if the server is
    reachable, False if all attempts (including auto-bootstrap) failed.

    Parameters
    ----------
    host : str
        Server bind address (default 127.0.0.1).
    port : int
        Server port (default 8000).
    console : rich.console.Console or None
        Rich console for styled output. Falls back to plain print if None.
    max_retries : int
        Max number of polling attempts after spawning server (default 10).
    poll_interval : float
        Seconds between polling attempts (default 0.5).

    Returns
    -------
    bool
        True if backend is live and ready; False if offline (caller should
        fall back to mock pipeline).
    """
    global _managed_process

    # ── 1. Active Connection Probe ───────────────────────────────────────
    if _ping_server(host, port, timeout=0.5):
        if console:
            console.print(
                f"[bold green]✔ Backend already running[/bold green] "
                f"[dim]at http://{host}:{port}[/dim]"
            )
        return True

    # ── 2. Automatic Background Server Launch ────────────────────────────
    if console:
        try:
            from rich.status import Status
            spinner = Status(
                "[bold yellow]⚠️  Backend offline. Auto-bootstrapping SafeSight AI Engine...[/bold yellow]",
                console=console,
                spinner="dots",
                spinner_style="bold yellow",
            )
            spinner.start()
        except ImportError:
            spinner = None
            console.print("[yellow]⚠️  Backend offline. Auto-bootstrapping SafeSight AI Engine...[/yellow]")
    else:
        spinner = None
        print("⚠  Backend offline. Auto-bootstrapping SafeSight AI Engine...")

    proc = _spawn_server(host, port)

    if proc is None:
        # Server binary failed to launch
        if spinner:
            spinner.stop()
        if console:
            from rich.panel import Panel
            from rich import box
            console.print(
                Panel(
                    "[bold red]❌ Failed to spawn uvicorn process.[/bold red]\n\n"
                    "[yellow]Possible causes:[/yellow]\n"
                    "  • uvicorn is not installed in the active Python environment\n"
                    "  • Port conflict (another process on port {port})\n"
                    "  • Missing project dependencies\n\n"
                    "[cyan]Falling back to local deterministic mock engine...[/cyan]",
                    title="[bold red]Engine Bootstrap Failed[/bold red]",
                    border_style="red",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        return False

    # Track the managed process for cleanup
    _managed_process = proc
    _register_cleanup(proc)

    # ── 3. Retry Polling Loop ────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        time.sleep(poll_interval)

        # Check if the process died during startup
        if proc.poll() is not None:
            if spinner:
                spinner.stop()
            if console:
                from rich.panel import Panel
                from rich import box
                console.print(
                    Panel(
                        f"[bold red]❌ Server process exited prematurely (code: {proc.returncode}).[/bold red]\n\n"
                        "[yellow]The server may have crashed during model loading or encountered\n"
                        "a configuration error. Check server logs for details.[/yellow]\n\n"
                        "[cyan]Falling back to local deterministic mock engine...[/cyan]",
                        title="[bold red]Engine Bootstrap Failed[/bold red]",
                        border_style="red",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )
            _managed_process = None
            return False

        if _ping_server(host, port, timeout=0.5):
            if spinner:
                spinner.stop()
            if console:
                console.print(
                    f"[bold green]✔ Engine auto-bootstrapped successfully[/bold green] "
                    f"[dim]in {attempt * poll_interval:.1f}s "
                    f"(http://{host}:{port}, PID: {proc.pid})[/dim]"
                )
                console.print(
                    "[dim yellow]ℹ  Server will be auto-terminated when this script exits.[/dim yellow]"
                )
            return True

        # Update spinner text with progress
        if spinner:
            elapsed = attempt * poll_interval
            remaining = (max_retries - attempt) * poll_interval
            spinner.update(
                f"[bold yellow]⚠️  Waiting for engine startup... "
                f"({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)[/bold yellow]"
            )

    # ── 4. Failsafe Fallback ─────────────────────────────────────────────
    if spinner:
        spinner.stop()

    # Kill the unresponsive process
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
    _managed_process = None

    if console:
        from rich.panel import Panel
        from rich import box
        max_wait = max_retries * poll_interval
        console.print(
            Panel(
                f"[bold red]❌ Server did not respond within {max_wait:.0f}s timeout.[/bold red]\n\n"
                "[yellow]The SafeSight backend was spawned but did not become responsive\n"
                "within the maximum wait window. This typically happens when:[/yellow]\n"
                "  • Heavy ML models (CLIP, DeBERTa, Whisper) are loading for the first time\n"
                "  • Insufficient system RAM/GPU memory\n"
                "  • Ollama service is not running\n\n"
                "[cyan]Falling back to local deterministic mock engine...[/cyan]",
                title="[bold yellow]⚠ Engine Startup Timeout — Fallback Activated[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    return False


def is_server_managed() -> bool:
    """Return True if this script spawned and is managing the backend process."""
    return _managed_process is not None and _managed_process.poll() is None
