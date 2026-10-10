#!/usr/bin/env python3
"""
SafeSight AI — Interactive Terminal Interface (REPL)
=====================================================
A persistent, developer-friendly command-line environment for analyzing
media files on the fly.

Features:
  - Persistent input loop powered by prompt_toolkit
  - OS-resilient drag-and-drop path resolver (Windows PowerShell & macOS Finder)
  - Shorthand filename lookup across project directories
  - Filesystem tab-completion via PathCompleter
  - Live AI Forensics & Content Moderation evaluation with rich dashboards
  - Graceful backend health probe and auto-bootstrap integration

Usage:
    python app/cli_interactive.py
    # or
    python -m app.cli_interactive
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── Force UTF-8 encoding on Windows to prevent UnicodeEncodeError ────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Rich UI Library ─────────────────────────────────────────────────────────
try:
    from rich import box
    from rich.console import Console
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.status import Status
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: 'rich' library is required. Run: pip install rich>=13.7.0")
    sys.exit(1)

# Global rich console
console = Console(force_terminal=True, color_system="auto")

# ── prompt_toolkit Library ──────────────────────────────────────────────────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import PathCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.output import create_output
    from prompt_toolkit.output.plain_text import PlainTextOutput
    from prompt_toolkit.styles import Style
except ImportError:
    console.print(
        "[bold red]ERROR: 'prompt_toolkit' is required for the interactive REPL.[/bold red]\n"
        "[yellow]Please run: pip install prompt_toolkit>=3.0.36[/yellow]"
    )
    sys.exit(1)

# Project component imports
from app.pipeline import run_multimodal_forensics
from app.ui.splash import display_splash_screen
from tests.engine_manager import ensure_backend, is_server_managed
from tests.test_single import _execute_mock_request, display_result

# ─────────────────────────────────────────────────────────────────────────────
# Constants & MIME Mapping
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_API_URL = f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/moderate"

MIME_MAP = {
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    # Video
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac"}
VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# prompt_toolkit style
PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "shield": "ansimagenta",
        "placeholder": "#777777 italic",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. OS-Resilient Drag-and-Drop Path Resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_file_path(user_input: str) -> Path | None:
    """
    Clean and resolve media file paths from terminal drag-and-drop or shorthand.

    Handles:
      - Stripping quotes (' and ") and whitespace
      - Windows PowerShell execution prefix '& ' or '& "..."'
      - macOS / Linux escaped spaces (e.g. 'my\\ image.jpg' -> 'my image.jpg')
      - file:// URI schemes
      - Direct relative and absolute path verification
      - Shorthand filename search across project root, sample_images,
        samples, tests/assets, and tests
    """
    if not user_input or not user_input.strip():
        return None

    raw = user_input.strip()

    # A. Strip Windows PowerShell execution prefix '& ' if present
    if raw.startswith("& "):
        raw = raw[2:].strip()
    elif raw.startswith("&") and len(raw) > 1 and raw[1] in ('"', "'"):
        raw = raw[1:].strip()

    # B. Strip leading and trailing quotes (single or double)
    while (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # C. Handle file:// URI scheme
    if raw.startswith("file://"):
        from urllib.parse import unquote, urlparse
        parsed = urlparse(raw)
        raw = unquote(parsed.path)
        if sys.platform == "win32" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]

    # D. Convert escaped spaces (like 'space\\ name.png' in macOS/Linux) to plain spaces
    raw = re.sub(r"\\ ", " ", raw)

    if not raw:
        return None

    # E. Hierarchy 1: Check if cleaned input is a valid absolute/relative path
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    # Check relative to project root
    candidate_root = (_PROJECT_ROOT / raw).resolve()
    if candidate_root.is_file():
        return candidate_root

    # F. Hierarchy 2: Shorthand filename lookup
    # If the input is just a filename (e.g. 'image5.png'), search project folders
    target_name = candidate.name.lower()
    search_dirs = [
        _PROJECT_ROOT,
        _PROJECT_ROOT / "sample_images",
        _PROJECT_ROOT / "samples",
        _PROJECT_ROOT / "tests" / "assets",
        _PROJECT_ROOT / "tests",
        Path.cwd(),
    ]

    # First check direct children of search directories
    for s_dir in search_dirs:
        if s_dir.is_dir():
            direct = s_dir / candidate.name
            if direct.is_file():
                return direct.resolve()

    # Recursively search (skipping heavy / hidden directories)
    for s_dir in search_dirs:
        if s_dir.is_dir():
            for item in s_dir.rglob("*"):
                # Prune .venv, .git, cache, node_modules
                if any(p.startswith(".") or p in (".venv", "venv", "__pycache__", "node_modules") for p in item.parts):
                    continue
                if item.is_file() and item.name.lower() == target_name:
                    return item.resolve()

    # G. Hierarchy 3: Not found -> clean Rich warning
    console.print(
        "[bold yellow]⚠️ File not found. Try dragging and dropping it into this window.[/bold yellow]"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backend Health Probe & Diagnostic Cards
# ─────────────────────────────────────────────────────────────────────────────

def is_backend_online(host: str = DEFAULT_API_HOST, port: int = DEFAULT_API_PORT) -> bool:
    """Perform a fast, non-blocking check to verify if FastAPI is reachable."""
    try:
        import httpx
        r = httpx.get(f"http://{host}:{port}/", timeout=0.5)
        return r.status_code in (200, 404, 405)
    except Exception:
        pass

    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        res = sock.connect_ex((host, port))
        sock.close()
        return res == 0
    except Exception:
        return False


def display_offline_card(host: str = DEFAULT_API_HOST, port: int = DEFAULT_API_PORT) -> None:
    """Display a rich diagnostic card when the FastAPI backend is offline."""
    console.print()
    card_content = Text()
    card_content.append("❌ FastAPI backend is offline at ", style="bold red")
    card_content.append(f"http://{host}:{port}\n\n", style="bold white")

    card_content.append("To enable live AI Forensics & Vision-Language analysis:\n", style="yellow")
    card_content.append("  1. Open a new terminal tab and run:\n", style="white")
    card_content.append("     uvicorn app.main:app --host 127.0.0.1 --port 8000\n", style="bold green")
    card_content.append("  2. Or run the server launcher script:\n", style="white")
    card_content.append("     python run_server.py\n", style="bold green")
    card_content.append("  3. Or run the native batch script:\n", style="white")
    card_content.append("     .\\run_native.bat\n\n", style="bold green")

    card_content.append(
        "💡 Tip: You can also type 'start' in this console to auto-bootstrap the backend,\n"
        "or continue dropping files to test with the local deterministic engine.",
        style="dim cyan",
    )

    panel = Panel(
        card_content,
        title="[bold yellow]⚠️  Backend Server Offline[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multimodal Forensics Dashboard & Execution
# ─────────────────────────────────────────────────────────────────────────────

def display_multimodal_forensics_card(
    forensic_data: dict[str, Any],
    file_path: Path,
) -> None:
    """
    Render executive multimodal forensics report:
      - Modality Processed (IMAGE / AUDIO / VIDEO)
      - Exact Spliced/AI Component
      - Coordinate / Timestamp Table
      - Direct file URI and path to visual evidence proof
    """
    modality = str(forensic_data.get("modality", "IMAGE")).upper()
    verdict = str(forensic_data.get("verdict", "AUTHENTIC")).upper()
    ai_component = str(forensic_data.get("ai_component", "None (Authentic)"))
    ai_pct = float(forensic_data.get("ai_percentage", 0.0))
    conf = float(forensic_data.get("confidence", 1.0))
    evidence_path = forensic_data.get("evidence_file_path")
    explanation = str(forensic_data.get("forensic_explanation", ""))
    items = forensic_data.get("coordinates_or_timestamps", [])

    is_tampered = "AUTHENTIC" not in verdict and "GENUINE" not in verdict
    status_color = "red" if is_tampered else "green"
    status_icon = "⚠️" if is_tampered else "🛡️"

    console.print()
    banner = Panel(
        Text.assemble(
            ("SafeSight AI Enterprise Multimodal Forensics · ", "bold white"),
            (f"{modality} FORENSICS\n", f"bold {'cyan' if modality=='IMAGE' else ('magenta' if modality=='AUDIO' else 'yellow')}"),
            (f"Status: {status_icon} {verdict}   ", f"bold {status_color}"),
            (f"· AI Involvement: {ai_pct:.1f}%   ", "bold white"),
            (f"· Confidence: {conf*100:.1f}%\n", "bold cyan"),
            (f"Forensic Assessment: {explanation}", "dim italic white"),
        ),
        title=f"[bold {status_color}]{status_icon}  MULTIMODAL FORENSIC REPORT · {modality}[/bold {status_color}]",
        border_style=status_color,
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(banner)

    summary_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    summary_table.add_column("Parameter", style="bold white", width=25)
    summary_table.add_column("Forensic Value", style="bold yellow")

    mod_color = "cyan" if modality == "IMAGE" else ("magenta" if modality == "AUDIO" else "yellow")
    comp_color = "red" if is_tampered else "green"
    summary_table.add_row("Modality Processed", f"[{mod_color} bold]{modality}[/{mod_color} bold]")
    summary_table.add_row("Exact Spliced/AI Component", f"[{comp_color} bold]{ai_component}[/{comp_color} bold]")
    summary_table.add_row("Authenticity Verdict", f"[{status_color} bold]{verdict}[/{status_color} bold]")
    summary_table.add_row("Input Media File", str(file_path.name))
    console.print(summary_table)

    if modality == "IMAGE" and items:
        # Anomaly Breakdown Panel
        breakdown_text = Text()
        for idx, r in enumerate(items, 1):
            lbl = r.get("region_label") or r.get("label", "Tampered Region")
            clean_lbl = str(lbl).replace("_", " ").title() if "_" in str(lbl) else str(lbl)
            r_conf = float(r.get("confidence", 0.0))
            reason = r.get("forensic_explanation") or r.get("forensic_reasoning", "Localized anomaly.")
            cat = r.get("category", "")
            cat_str = f" [{cat}]" if cat else ""

            breakdown_text.append(f"• Region {idx}: ", style="bold cyan")
            breakdown_text.append(f"{clean_lbl}", style="bold yellow")
            breakdown_text.append(f"{cat_str}  ·  Confidence: ", style="dim")
            breakdown_text.append(f"{r_conf * 100:.1f}%\n", style="bold red" if r_conf > 0.85 else "bold yellow")
            breakdown_text.append("  Reason: ", style="bold white")
            breakdown_text.append(f"{reason}\n", style="italic white")
            if idx < len(items):
                breakdown_text.append("\n")

        breakdown_panel = Panel(
            breakdown_text,
            title="[bold yellow]🔍  Anomaly Breakdown[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(breakdown_panel)

        coord_table = Table(
            title="[bold red]🎯  Localized Tampering Coordinates[/bold red]",
            box=box.ROUNDED,
            border_style="bold red",
            header_style="bold red",
            expand=True,
            show_lines=True,
        )
        coord_table.add_column("Region", style="bold yellow", no_wrap=True)
        coord_table.add_column("Confidence", justify="right", style="bold white", no_wrap=True)
        coord_table.add_column("Bounding Box (ymin, xmin, ymax, xmax)", justify="center", style="cyan", no_wrap=True)
        coord_table.add_column("Anomaly Details", style="white")

        for r in items:
            lbl = r.get("region_label") or r.get("label", "tampered_region")
            clean_lbl = str(lbl).replace("_", " ").title() if "_" in str(lbl) else str(lbl)
            r_conf = float(r.get("confidence", 0.0))
            box_2d = r.get("box_2d", [0, 0, 0, 0])
            reason = r.get("forensic_explanation") or r.get("forensic_reasoning", "Localized anomaly.")
            coord_table.add_row(
                str(clean_lbl),
                f"{r_conf * 100:.1f}%",
                f"[{box_2d[0]}, {box_2d[1]}, {box_2d[2]}, {box_2d[3]}]",
                str(reason),
            )
        console.print(coord_table)

    elif modality == "AUDIO" and items:
        aud_table = Table(
            title="[bold magenta]⏱️  Spliced Audio & Voice Clone Timestamps[/bold magenta]",
            box=box.ROUNDED,
            border_style="bold magenta",
            header_style="bold magenta",
            expand=True,
            show_lines=True,
        )
        aud_table.add_column("Segment Type", style="bold yellow", no_wrap=True)
        aud_table.add_column("Confidence", justify="right", style="bold white", no_wrap=True)
        aud_table.add_column("Start Time (s)", justify="center", style="cyan", no_wrap=True)
        aud_table.add_column("End Time (s)", justify="center", style="cyan", no_wrap=True)
        aud_table.add_column("Duration (s)", justify="right", style="bold white", no_wrap=True)
        aud_table.add_column("Forensic Reasoning", style="white")

        for s in items:
            s_type = s.get("type", "voice_clone")
            s_conf = float(s.get("confidence", 0.0))
            s_start = float(s.get("start_sec", 0.0))
            s_end = float(s.get("end_sec", 0.0))
            s_dur = max(0.0, s_end - s_start)
            reason = s.get("reasoning", "Acoustic anomaly detected.")
            aud_table.add_row(
                str(s_type),
                f"{s_conf * 100:.1f}%",
                f"{s_start:.2f}s",
                f"{s_end:.2f}s",
                f"{s_dur:.2f}s",
                str(reason),
            )
        console.print(aud_table)

    elif modality == "VIDEO" and items:
        vid_table = Table(
            title="[bold yellow]🎬  Manipulated Deepfake Frame Intervals[/bold yellow]",
            box=box.ROUNDED,
            border_style="bold yellow",
            header_style="bold yellow",
            expand=True,
            show_lines=True,
        )
        vid_table.add_column("Anomaly Type", style="bold yellow", no_wrap=True)
        vid_table.add_column("Confidence", justify="right", style="bold white", no_wrap=True)
        vid_table.add_column("Frame Range", justify="center", style="cyan", no_wrap=True)
        vid_table.add_column("Bounding Box (ymin, xmin, ymax, xmax)", justify="center", style="dim cyan", no_wrap=True)
        vid_table.add_column("Anomaly Details", style="white")

        for m in items:
            m_anom = m.get("anomaly", "face_swap_boundary_mismatch")
            m_conf = float(m.get("confidence", 0.0))
            f_start = m.get("frame_start", 0)
            f_end = m.get("frame_end", 0)
            b_box = m.get("box_2d", [0, 0, 0, 0])
            b_str = f"[{b_box[0]}, {b_box[1]}, {b_box[2]}, {b_box[3]}]"
            vid_table.add_row(
                str(m_anom),
                f"{m_conf * 100:.1f}%",
                f"Frames {f_start} - {f_end}",
                b_str,
                "Boundary chromaticity mismatch / motion discontinuity.",
            )
        console.print(vid_table)

    if evidence_path:
        ev_p = Path(evidence_path).resolve()
        console.print(f"[bold cyan]🖼️ Visual Proof Generated:[/bold cyan] [bold green]{ev_p}[/bold green]")
        console.print(
            Padding(
                Text.assemble(
                    ("🖼️  Visual Forensic Evidence Proof: ", "bold cyan"),
                    (ev_p.as_uri(), "bold underline green"),
                    (f"\n   Local Path: {str(ev_p)}", "dim white"),
                ),
                (1, 0, 0, 1),
            )
        )


def evaluate_media_file(
    file_path: Path,
    target_url: str = DEFAULT_API_URL,
    text_caption: str | None = None,
) -> None:
    """
    Evaluate media file through SafeSight AI Multimodal Forensics Suite and API.
    """
    try:
        # 1. Run local Multimodal Forensic Suite
        with Status(
            f"[bold cyan]🔍 Evaluating [magenta]{file_path.name}[/magenta] through Multimodal Forensic Engine...[/bold cyan]",
            console=console,
            spinner="dots",
        ):
            forensic_res = run_multimodal_forensics(file_path=file_path, text_context=text_caption)

        # 2. Render dedicated Multimodal Forensics Card
        display_multimodal_forensics_card(forensic_res, file_path)

    except Exception as exc:
        console.print(f"[bold red]❌ Forensic analysis error:[/bold red] {exc}")

    # 3. If caption is provided or backend is active, also check moderation API
    if text_caption or is_backend_online(DEFAULT_API_HOST, DEFAULT_API_PORT):
        backend_live = is_backend_online(DEFAULT_API_HOST, DEFAULT_API_PORT)
        result_data = None
        suffix = file_path.suffix.lower()
        mime_type = MIME_MAP.get(suffix, "application/octet-stream")
        field_key = "audio" if suffix in AUDIO_EXTS else ("video" if suffix in VIDEO_EXTS else "image")

        if backend_live:
            try:
                import httpx as http_client
                with Status("[dim cyan]Syncing with content moderation gate...[/dim cyan]", console=console, spinner="dots"):
                    files = {field_key: (file_path.name, file_path.read_bytes(), mime_type)}
                    data_form = {"text": text_caption} if text_caption else {}
                    r = http_client.post(target_url, data=data_form, files=files, timeout=60.0)
                    if r.status_code == 200:
                        result_data = r.json()
            except Exception:
                pass

        if result_data and text_caption:
            display_result(data=result_data, input_text=text_caption, input_file=str(file_path))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive Terminal Loop (REPL)
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_repl() -> None:
    """
    Main interactive REPL loop.
    """
    # Clear screen and display modern splash screen
    display_splash_screen(
        clear=True,
        local_server=f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}",
        api_docs=f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/docs",
    )

    console.print(
        "[bold cyan]SafeSight AI Interactive Media Inspector[/bold cyan]\n"
        "[dim]Drag & drop any media file from your File Explorer/Finder, type a filename, or hit TAB for path completion.[/dim]\n"
        "[dim]Type [bold white]'help'[/bold white] for commands, [bold white]'start'[/bold white] to boot server, or [bold white]'exit'[/bold white] to quit.[/dim]\n"
    )

    # Initial backend connectivity notice
    if not is_backend_online(DEFAULT_API_HOST, DEFAULT_API_PORT):
        display_offline_card(DEFAULT_API_HOST, DEFAULT_API_PORT)

    # Set up prompt_toolkit session with filesystem autocompletion
    completer = PathCompleter(expanduser=True)

    try:
        output = create_output()
        session: PromptSession[str] = PromptSession(
            completer=completer,
            output=output,
            style=PROMPT_STYLE,
        )
    except Exception:
        # Fallback for environments without standard Windows console screen buffer
        output = PlainTextOutput(sys.stdout)
        session = PromptSession(
            completer=completer,
            output=output,
            style=PROMPT_STYLE,
        )

    # Interactive REPL Loop
    while True:
        try:
            user_input = session.prompt(
                HTML("<ansicyan><b>SafeSight AI</b></ansicyan> 🛡️ <b>&gt;</b> "),
                placeholder=HTML("<style color='#777777'><i>Drop a file, type a filename, or type 'exit'</i></style>"),
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]👋 Exiting SafeSight AI Interactive Console. Stay safe![/bold cyan]")
            break

        cleaned_input = user_input.strip()

        if not cleaned_input:
            continue

        # Handle built-in commands
        cmd = cleaned_input.lower()
        if cmd in ("exit", "quit", "q"):
            console.print("[bold cyan]👋 Exiting SafeSight AI Interactive Console. Stay safe![/bold cyan]")
            break

        if cmd in ("clear", "cls"):
            console.clear()
            display_splash_screen(clear=False)
            continue

        if cmd == "help":
            console.print()
            help_text = (
                "[bold white]Available Commands & Actions:[/bold white]\n\n"
                "  • [bold cyan]Drag & Drop:[/bold cyan] Drag any image, video, or audio file directly into this terminal window.\n"
                "  • [bold cyan]Shorthand:[/bold cyan]   Type a filename (e.g. [green]nature_sample.jpg[/green]) — SafeSight will find it automatically.\n"
                "  • [bold cyan]Tab-Complete:[/bold cyan] Type a folder path and press [green]TAB[/green] to autocomplete files.\n"
                "  • [bold cyan]start:[/bold cyan]        Auto-bootstrap the FastAPI backend server in the background.\n"
                "  • [bold cyan]status:[/bold cyan]       Check whether the backend server and Ollama are reachable.\n"
                "  • [bold cyan]clear / cls:[/bold cyan]  Clear the screen and redisplay the splash screen.\n"
                "  • [bold cyan]exit / quit:[/bold cyan]  Close the interactive console.\n"
            )
            console.print(Panel(help_text, title="[bold cyan]SafeSight AI Help[/bold cyan]", box=box.ROUNDED))
            console.print()
            continue

        if cmd == "status":
            online = is_backend_online(DEFAULT_API_HOST, DEFAULT_API_PORT)
            if online:
                console.print(f"[bold green]✔ FastAPI backend is LIVE[/bold green] at http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}")
            else:
                console.print(f"[bold red]❌ FastAPI backend is OFFLINE[/bold red] at http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}")
            continue

        if cmd == "start":
            console.print("[cyan]Initiating backend bootstrap...[/cyan]")
            ensure_backend(
                host=DEFAULT_API_HOST,
                port=DEFAULT_API_PORT,
                console=console,
            )
            continue

        # ── Resolve Media File ───────────────────────────────────────────────
        resolved_path = resolve_file_path(cleaned_input)

        if resolved_path is not None:
            console.print(f"[dim green]✔ Resolved file:[/dim green] [bold white]{resolved_path}[/bold white]")
            evaluate_media_file(resolved_path)
            console.print("[dim]─" * 60 + "[/dim]\n")


def main() -> None:
    """CLI Entrypoint."""
    try:
        run_interactive_repl()
    except Exception as exc:
        console.print(f"[bold red]Unexpected error in REPL session:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
