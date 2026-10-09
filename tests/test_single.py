#!/usr/bin/env python3
"""
SafeSight AI — The Beautiful CLI Single Test Runner
=====================================================
A modern, polished, and developer-friendly terminal runner for SafeSight AI
multimodal content moderation and deepfake forensics.

Accepts an optional file path (--file / --image / positional) and text string.
Formats API and local pipeline verification results using Rich panels, tables,
and color-coded status themes.

Usage:
    # Flag syntax (recommended)
    python tests/test_single.py --file sample_images/nature_sample.jpg --text "Mountain sunrise"
    python tests/test_single.py --file sample_images/meme_sample.jpg
    python tests/test_single.py --text "I hate everyone, death to them"

    # Positional syntax
    python tests/test_single.py sample_images/nature_sample.jpg "Mountain sunrise"
    python tests/test_single.py "Check this message"

    # Specify custom URL (default: http://localhost:8000/moderate)
    python tests/test_single.py --file my_image.jpg --text "My text" --url http://localhost:8000/moderate

    # Force local deterministic mock pipeline (offline testing)
    python tests/test_single.py --text "I hate you" --mock

    # Raw JSON output mode
    python tests/test_single.py --text "Hello world" --json
"""

from __future__ import annotations

import argparse
import datetime
import json
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

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Rich UI Library ─────────────────────────────────────────────────────────
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns
    from rich.text import Text
    from rich.padding import Padding
    from rich import box
    from app.ui.splash import display_splash_screen
    from tests.engine_manager import ensure_backend
except ImportError:
    print(
        "ERROR: 'rich' library is required for SafeSight CLI.\n"
        "Please run: pip install rich>=13.7.0"
    )
    sys.exit(1)

# Initialize global Rich console
console = Console(force_terminal=True, color_system="auto")


# ─────────────────────────────────────────────────────────────────────────────
# Theme & Color Configuration
# ─────────────────────────────────────────────────────────────────────────────

STATUS_THEME = {
    "APPROVED": {
        "icon": "🟢",
        "color": "green",
        "border_style": "bold green",
        "title": "APPROVED",
        "subtitle": "Safe for Publication · No Severe Policy Violations Detected",
        "badge": "[bold white on green]  APPROVED  [/bold white on green]",
    },
    "HUMAN_AUDIT_REQUIRED": {
        "icon": "🟡",
        "color": "yellow",
        "border_style": "bold yellow",
        "title": "HUMAN AUDIT REQUIRED",
        "subtitle": "Borderline / Compound Risk · Human Review Recommended",
        "badge": "[bold black on yellow]  AUDIT REQUIRED  [/bold black on yellow]",
    },
    "FLAGGED": {
        "icon": "🔴",
        "color": "red",
        "border_style": "bold red",
        "title": "FLAGGED",
        "subtitle": "Policy Violations Detected · Content Quarantined / Blocked",
        "badge": "[bold white on red]  FLAGGED  [/bold white on red]",
    },
}

FORENSICS_VERDICT_CONFIG = {
    "AUTHENTIC": {
        "color": "green",
        "icon": "🛡️",
        "style": "bold green",
        "desc": "Original media capture · No tampering artifacts",
    },
    "HUMAN_ORIGINAL": {
        "color": "green",
        "icon": "🛡️",
        "style": "bold green",
        "desc": "Natural human expression · Authentic variance",
    },
    "BENIGN_AI_ASSISTED": {
        "color": "cyan",
        "icon": "🎨",
        "style": "bold cyan",
        "desc": "Non-harmful AI enhancement or editing tools",
    },
    "HYBRID_HUMAN_AI_EDITED": {
        "color": "yellow",
        "icon": "✂️",
        "style": "bold yellow",
        "desc": "Composite content with mixed human and generative segments",
    },
    "SYNTHETIC_GENERATED": {
        "color": "red",
        "icon": "⚠️",
        "style": "bold red",
        "desc": "Fully AI-generated synthetic media (Diffusion/GAN)",
    },
    "FULLY_AI_GENERATED": {
        "color": "red",
        "icon": "⚠️",
        "style": "bold red",
        "desc": "Fully synthetic generative content",
    },
    "MALICIOUS_TAMPERING": {
        "color": "red",
        "icon": "🚨",
        "style": "bold red",
        "desc": "Deepfake, face-swap, audio splice, or inpainting tampering",
    },
}

SEVERITY_STYLES = {
    "HIGH": "[bold red]HIGH[/bold red]",
    "MEDIUM": "[bold yellow]MEDIUM[/bold yellow]",
    "LOW": "[bold cyan]LOW[/bold cyan]",
}


# ─────────────────────────────────────────────────────────────────────────────
# UI Rendering Functions
# ─────────────────────────────────────────────────────────────────────────────

def _render_progress_bar(percentage: float, width: int = 28, color: str = "red") -> Text:
    """Render a clean text-based progress meter for AI involvement."""
    pct = max(0.0, min(100.0, percentage))
    filled = int(round((pct / 100.0) * width))
    empty = width - filled
    bar_text = Text()
    bar_text.append("█" * filled, style=color)
    bar_text.append("░" * empty, style="dim")
    bar_text.append(f"  {pct:.1f}%", style="bold white")
    return bar_text


def display_result(
    data: dict[str, Any],
    input_text: str | None = None,
    input_file: str | None = None,
) -> None:
    """
    Render moderation and forensics results in a modern, polished terminal layout.

    Layout Structure:
      A. Main top Panel acting as a status banner.
      B. Borderless 'Assessment Summary' Table showing input parameters, decision, confidence.
      C. 'Forensics Verdict' section dynamically color-coded with AI involvement percentage.
      D. Red border 'Policy Violations' Table (displayed only if violations exist).
      E. Three Panels presenting the 'System Reasoning Breakdown' (Textual, Visual, Bimodal).
    """
    decision = str(data.get("decision", "FLAGGED")).upper()
    theme = STATUS_THEME.get(decision, STATUS_THEME["FLAGGED"])

    confidence = float(data.get("confidence_score", 0.0))
    tier = str(data.get("pipeline_tier_used", "TIER1"))
    latency = float(data.get("total_latency_ms", 0.0))
    media_types = data.get("media_types_processed", [])
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # A. MAIN TOP PANEL — Status Banner
    # ─────────────────────────────────────────────────────────────────────────
    banner_content = Text()
    banner_content.append("SafeSight AI ", style="bold white")
    banner_content.append("· Multimodal Content Moderation & Deepfake Forensics Pipeline\n", style="dim")
    banner_content.append(f"Status: ", style="dim")
    banner_content.append(f"{theme['icon']} {theme['title']}  ", style=f"bold {theme['color']}")
    banner_content.append(f"· Pipeline: ", style="dim")
    banner_content.append(f"{tier}  ", style="bold magenta")
    banner_content.append(f"· Latency: ", style="dim")
    banner_content.append(f"{latency:.1f} ms  ", style="bold cyan")
    banner_content.append(f"· Verified: ", style="dim")
    banner_content.append(f"{now_str}", style="dim white")

    top_banner = Panel(
        banner_content,
        title=f"[bold {theme['color']}]{theme['icon']}  {theme['title']}  {theme['icon']}[/bold {theme['color']}]",
        subtitle=f"[dim]{theme['subtitle']}[/dim]",
        border_style=theme["border_style"],
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(top_banner)

    # ─────────────────────────────────────────────────────────────────────────
    # B. BORDERLESS 'ASSESSMENT SUMMARY' TABLE
    # ─────────────────────────────────────────────────────────────────────────
    summary_title = Text("ASSESSMENT SUMMARY", style="bold white")
    summary_rule = Text("─" * 40, style="dim cyan")
    console.print(Padding(Text.assemble(summary_title, "  ", summary_rule), (1, 0, 0, 1)))

    summary_table = Table(
        box=None,
        show_header=False,
        pad_edge=False,
        padding=(0, 3),
        expand=True,
    )
    summary_table.add_column("Parameter", style="bold cyan", width=22, no_wrap=True)
    summary_table.add_column("Value", style="bold white")

    # Input parameters
    if input_text:
        preview_text = input_text if len(input_text) <= 90 else f"{input_text[:87]}..."
        summary_table.add_row("Input Text", f'[italic]"{preview_text}"[/italic]')
    else:
        summary_table.add_row("Input Text", "[dim]None provided[/dim]")

    if input_file:
        p = Path(input_file)
        size_kb = (p.stat().st_size / 1024) if p.exists() else 0.0
        summary_table.add_row("Input Media File", f"{p.name} [dim]({size_kb:.1f} KB, path: {p})[/dim]")
    else:
        summary_table.add_row("Input Media File", "[dim]None provided[/dim]")

    if media_types:
        modality_str = ", ".join(str(m).upper() for m in media_types)
        summary_table.add_row("Modalities Processed", f"[bold white]{modality_str}[/bold white]")

    # Decision, confidence, tier, latency
    decision_badge = f"[{theme['color']} bold]{theme['icon']}  {decision}[/{theme['color']} bold]"
    conf_pct = f"[bold white]{confidence * 100:.1f}%[/bold white]"
    if confidence >= 0.90:
        conf_badge = f"{conf_pct} [green](Very High Confidence)[/green]"
    elif confidence >= 0.75:
        conf_badge = f"{conf_pct} [yellow](Moderate Confidence)[/yellow]"
    else:
        conf_badge = f"{conf_pct} [dim red](Low / Borderline)[/dim red]"

    summary_table.add_row("Model Decision", decision_badge)
    summary_table.add_row("Confidence Rating", conf_badge)
    summary_table.add_row("Pipeline Tier Used", f"[magenta]{tier}[/magenta]")
    summary_table.add_row("Engine Latency", f"[cyan]{latency:.1f} ms[/cyan]")

    console.print(Padding(summary_table, (0, 0, 1, 2)))

    # ─────────────────────────────────────────────────────────────────────────
    # C. 'FORENSICS VERDICT' SECTION
    # ─────────────────────────────────────────────────────────────────────────
    auth_report = data.get("authenticity_report") or {}
    raw_verdict = auth_report.get("verdict", "AUTHENTIC")
    verdict_str = (
        raw_verdict.value if hasattr(raw_verdict, "value")
        else str(raw_verdict).split(".")[-1].upper()
    )

    # Determine dynamic color based on authenticity vs tampering
    # Green for AUTHENTIC / HUMAN_ORIGINAL
    # Red for MALICIOUS_TAMPERING / SYNTHETIC_GENERATED
    f_config = FORENSICS_VERDICT_CONFIG.get(
        verdict_str,
        {
            "color": "red" if ("TAMPER" in verdict_str or "SYNTH" in verdict_str or "AI" in verdict_str) else "green",
            "icon": "⚠️" if ("TAMPER" in verdict_str or "SYNTH" in verdict_str) else "🛡️",
            "style": f"bold {'red' if ('TAMPER' in verdict_str or 'SYNTH' in verdict_str) else 'green'}",
            "desc": "Automated forensic assessment",
        }
    )
    f_color = f_config["color"]
    f_icon = f_config["icon"]
    f_desc = f_config["desc"]

    ai_pct = float(auth_report.get("ai_involvement_percentage", 0.0))
    annotations = auth_report.get("tamper_annotations", [])

    forensics_text = Text()
    forensics_text.append(f"{f_icon} Authenticity Verdict: ", style="bold white")
    forensics_text.append(f"{verdict_str} ", style=f_config["style"])
    forensics_text.append(f"({f_desc})\n\n", style="dim")

    forensics_text.append("Calculated AI Involvement: ", style="bold cyan")
    forensics_text.append_text(_render_progress_bar(ai_pct, width=28, color=f_color))
    forensics_text.append("\n")

    if annotations:
        forensics_text.append("\nForensic Observations & Artifact Annotations:\n", style="bold yellow")
        for ann in annotations:
            forensics_text.append(f"  • {ann}\n", style="white")

    forensics_panel = Panel(
        forensics_text,
        title=f"[bold {f_color}]{f_icon}  Forensics & Deepfake Authenticity Verdict[/bold {f_color}]",
        border_style=f"bold {f_color}",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(forensics_panel)

    # ─────────────────────────────────────────────────────────────────────────
    # D. RED BORDER 'POLICY VIOLATIONS' TABLE (ONLY if violations exist)
    # ─────────────────────────────────────────────────────────────────────────
    violations = data.get("detected_violations", [])
    if violations:
        console.print()
        viol_table = Table(
            title="[bold red]🚨  Detected Policy Violations[/bold red]",
            title_justify="left",
            border_style="bold red",
            header_style="bold red",
            box=box.ROUNDED,
            expand=True,
            show_lines=True,
        )
        viol_table.add_column("Category", style="bold red", no_wrap=True)
        viol_table.add_column("Policy Code", style="yellow", no_wrap=True)
        viol_table.add_column("Severity", justify="center", no_wrap=True)
        viol_table.add_column("Confidence", justify="right", style="bold white", no_wrap=True)
        viol_table.add_column("Action / Enforcement", style="dim white")

        for v in violations:
            cat_val = v.get("category", "UNKNOWN")
            if hasattr(cat_val, "value"):
                cat_val = cat_val.value
            elif "." in str(cat_val):
                cat_val = str(cat_val).split(".")[-1]

            code_val = v.get("policy_code", "N/A")

            sev_val = str(v.get("severity", "HIGH")).upper()
            if hasattr(sev_val, "value"):
                sev_val = sev_val.value
            elif "." in str(sev_val):
                sev_val = str(sev_val).split(".")[-1]
            sev_badge = SEVERITY_STYLES.get(sev_val, f"[white]{sev_val}[/white]")

            c_val = float(v.get("confidence", 0.0))
            action_desc = "Quarantine & Review" if sev_val == "HIGH" else "Flag for Audit"

            viol_table.add_row(
                str(cat_val),
                str(code_val),
                sev_badge,
                f"{c_val * 100:.1f}%",
                action_desc,
            )

        console.print(viol_table)
    else:
        # Subtle clean indicator that no policy violations were found
        console.print(Padding(Text("✔ No safety policy violations detected in provided content.", style="dim green"), (0, 0, 0, 1)))

    # ─────────────────────────────────────────────────────────────────────────
    # E. THREE PANELS: SYSTEM REASONING BREAKDOWN
    # ─────────────────────────────────────────────────────────────────────────
    reasoning = data.get("reasoning", {}) or {}
    text_exp = reasoning.get("text_explanation") or "No textual signals analyzed or clean text content."
    visual_exp = reasoning.get("visual_explanation") or "No image provided or visual signals within nominal safe thresholds."
    fusion_exp = reasoning.get("bimodal_synergy") or "Cross-modal signals evaluated. No conflicting or amplifying toxicity identified."

    panel_textual = Panel(
        Text(text_exp, style="white"),
        title="[bold yellow]📝  Textual Analysis[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    panel_visual = Panel(
        Text(visual_exp, style="white"),
        title="[bold magenta]👁️   Visual Analysis[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    panel_fusion = Panel(
        Text(fusion_exp, style="white"),
        title="[bold cyan]⚡  Bimodal Context Fusion[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    console.print()
    reasoning_title = Text("🧠  SYSTEM REASONING BREAKDOWN", style="bold white")
    reasoning_rule = Text("─" * 32, style="dim magenta")
    console.print(Padding(Text.assemble(reasoning_title, "  ", reasoning_rule), (0, 0, 1, 1)))

    # If terminal is comfortably wide, render side-by-side columns; otherwise stack
    if console.width >= 120:
        console.print(Columns([panel_textual, panel_visual, panel_fusion], expand=True, equal=True))
    else:
        console.print(panel_textual)
        console.print(panel_visual)
        console.print(panel_fusion)

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Request Execution (Live API with Automatic Offline Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_candidate_endpoints(target_url: str) -> list[str]:
    """Resolve prioritized candidate URLs for the SafeSight API."""
    clean = target_url.rstrip("/")
    if clean.endswith("/moderate/multimodal") or clean.endswith("/api/v1/moderate/multimodal"):
        return [clean]
    if clean.endswith("/moderate"):
        return [clean, f"{clean}/multimodal", f"{clean.rsplit('/moderate', 1)[0]}/api/v1/moderate/multimodal"]
    if clean.endswith("/moderate/text"):
        return [clean, f"{clean.rsplit('/moderate/text', 1)[0]}/api/v1/moderate/text"]
    if clean.endswith("/moderate/image"):
        return [clean, f"{clean.rsplit('/moderate/image', 1)[0]}/api/v1/moderate/image"]

    return [
        f"{clean}/moderate",
        f"{clean}/api/v1/moderate/multimodal",
        f"{clean}/moderate/multimodal",
    ]


def _execute_api_request(
    target_url: str,
    text: str | None,
    file_path: str | None,
) -> dict[str, Any]:
    """
    Send request to live FastAPI server with automatic graceful fallback
    to local deterministic engine if the server is offline.
    """
    try:
        import httpx as http_client
    except ImportError:
        try:
            import requests as http_client  # type: ignore
        except ImportError:
            console.print("[yellow]⚠ Neither httpx nor requests installed; running offline engine directly.[/yellow]")
            return _execute_mock_request(text, file_path)

    form_data: dict[str, str] = {}
    if text:
        form_data["text"] = text

    file_bytes: bytes | None = None
    file_name: str = ""
    mime_type: str = "application/octet-stream"

    if file_path:
        p = Path(file_path)
        if not p.exists():
            console.print(f"[bold red]❌ Error:[/bold red] Media file not found: [yellow]{file_path}[/yellow]")
            sys.exit(1)

        suffix = p.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".mp4": "video/mp4",
        }
        mime_type = mime_map.get(suffix, "application/octet-stream")
        file_name = p.name
        file_bytes = p.read_bytes()

    candidate_urls = _resolve_candidate_endpoints(target_url)

    for url in candidate_urls:
        files: dict[str, Any] = {}
        if file_bytes is not None:
            if suffix in (".mp3", ".wav", ".flac", ".ogg", ".aac"):
                field_key = "audio"
            elif suffix in (".mp4", ".webm", ".avi", ".mov", ".mkv"):
                field_key = "video"
            else:
                field_key = "image"
            files = {field_key: (file_name, file_bytes, mime_type)}

        t0 = time.perf_counter()
        try:
            response = http_client.post(url, data=form_data, files=files if files else None, timeout=180.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if response.status_code == 200:
                console.print(f"[dim green]✔ Connected to live backend at {url} (HTTP 200 in {elapsed_ms:.1f}ms)[/dim green]")
                return response.json()
            elif response.status_code in (404, 405):
                continue
            else:
                console.print(f"[bold red]❌ API Error (HTTP {response.status_code}):[/bold red] {response.text}")
                sys.exit(1)

        except Exception as exc:
            exc_name = type(exc).__name__
            if "Connect" in exc_name or "Connection" in exc_name or "Timeout" in exc_name:
                console.print(f"[bold yellow]⚠ FastAPI backend offline at {url} ({exc_name}).[/bold yellow]")
                console.print("[cyan]ℹ Falling back gracefully to SafeSight deterministic local verification engine...[/cyan]\n")
                return _execute_mock_request(text, file_path)
            raise

    # If all candidate endpoints returned 404/405, fall back to mock
    console.print(f"[yellow]⚠ Endpoints not found at {target_url}. Falling back to local offline engine...[/yellow]\n")
    return _execute_mock_request(text, file_path)


def _execute_mock_request(text: str | None, file_path: str | None) -> dict[str, Any]:
    """Execute the local deterministic pipeline without server dependency."""
    from tests.mock_pipeline import mock_moderate

    t0 = time.perf_counter()
    result = mock_moderate(text=text, image_path=file_path, simulate_latency=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    dumped = result.model_dump()
    dumped["total_latency_ms"] = round(latency_ms, 2)
    return dumped


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parsing & Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _is_file_path(val: str) -> bool:
    """Check if value is an existing file or has common media extension."""
    p = Path(val)
    if p.is_file():
        return True
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".mp4", ".wav", ".mp3")
    return p.suffix.lower() in exts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_single",
        description="SafeSight AI — Single Request CLI Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional arguments: support `python tests/test_single.py path/to/file.jpg "caption"`
    parser.add_argument(
        "positional_args",
        nargs="*",
        help="Positional inputs: [file_path] [text_string] or [text_string]",
    )
    # Flag arguments
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to media file (image, audio, or video)",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Alias for --file",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text string or prompt to moderate",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/moderate",
        help="API endpoint URL (default: http://localhost:8000/moderate)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Legacy alias for --url (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force local deterministic offline mock pipeline",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="raw_json",
        help="Output raw JSON response",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve file and text inputs
    file_path = args.file or args.image
    text = args.text

    # Parse positional inputs if flags weren't fully provided
    pos = args.positional_args
    if pos:
        if len(pos) == 1:
            if _is_file_path(pos[0]):
                if not file_path:
                    file_path = pos[0]
            else:
                if not text:
                    text = pos[0]
        elif len(pos) >= 2:
            if _is_file_path(pos[0]):
                if not file_path:
                    file_path = pos[0]
                if not text:
                    text = pos[1]
            elif _is_file_path(pos[1]):
                if not text:
                    text = pos[0]
                if not file_path:
                    file_path = pos[1]
            else:
                if not file_path:
                    file_path = pos[0]
                if not text:
                    text = pos[1]

    # Validate that at least one input is provided
    if not text and not file_path:
        console.print(
            Panel(
                "[bold red]Error: No input provided.[/bold red]\n\n"
                "Please provide at least one text string or media file path.\n\n"
                "[bold cyan]Examples:[/bold cyan]\n"
                '  python tests/test_single.py --file sample_images/nature_sample.jpg --text "Nature photo"\n'
                '  python tests/test_single.py --text "Check this content"\n'
                '  python tests/test_single.py "Check this message"',
                title="[bold red]Invalid Command Usage[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        sys.exit(1)

    target_url = args.host if (args.host and not args.url) else args.url

    # Display splash screen unless raw json is requested
    if not args.raw_json:
        server_base = target_url.rsplit("/moderate", 1)[0] if "/moderate" in target_url else target_url
        display_splash_screen(
            clear=True,
            local_server=server_base,
            api_docs=f"{server_base}/docs",
        )

    # Execute request
    if args.mock:
        console.print("[dim cyan]ℹ Running in offline mock mode (--mock flag provided)...[/dim cyan]")
        result_data = _execute_mock_request(text, file_path)
    else:
        # ── Auto-Bootstrapping Engine Manager ────────────────────────────
        # Extract host/port from the target URL for the engine probe
        from urllib.parse import urlparse
        parsed = urlparse(target_url)
        probe_host = parsed.hostname or "127.0.0.1"
        probe_port = parsed.port or 8000

        console.print()
        server_live = ensure_backend(
            host=probe_host,
            port=probe_port,
            console=console,
        )
        console.print()

        if server_live:
            result_data = _execute_api_request(target_url, text, file_path)
        else:
            # Failsafe: server could not be reached even after auto-bootstrap
            console.print(
                "[cyan]ℹ Falling back gracefully to SafeSight deterministic local verification engine...[/cyan]\n"
            )
            result_data = _execute_mock_request(text, file_path)

    # Render output
    if args.raw_json:
        console.print_json(json.dumps(result_data, indent=2))
    else:
        display_result(result_data, input_text=text, input_file=file_path)


if __name__ == "__main__":
    main()
