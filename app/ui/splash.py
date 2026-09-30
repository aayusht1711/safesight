"""
SafeSight AI — Terminal Splash Screen & Dashboard
===================================================
A modern, polished CLI home dashboard and boot screen inspired by top-tier
developer tools (Supabase, Stripe, Astro CLI).

Renders centered ASCII art branding, project tagline, and an aligned
system runtime status card using the Rich library.
"""

from __future__ import annotations

import sys
from typing import Any

# Ensure safe UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    # Graceful fallback if rich is somehow missing
    def display_splash_screen(*args: Any, **kwargs: Any) -> None:
        print("=" * 70)
        print("  SAFESIGHT AI — Real-Time Multimodal Moderation & Forensics")
        print("=" * 70)
        print("  Version:      1.0.0 (Production Prototype)")
        print("  Server:       http://localhost:8000")
        print("  API Docs:     http://localhost:8000/docs")
        print("  Status:       Live Engine [Ollama: Qwen2.5-VL-7B]")
        print("=" * 70 + "\n")
else:
    console = Console(force_terminal=True, color_system="auto")

    # Pure ASCII art logo for SAFESIGHT AI (width: 66 chars, 100% terminal safe)
    ASCII_LOGO = r"""   ___   _   ___  ___  ___  ___  ___  _  _  ___       _   ___ 
  / __| /_\ | __|| __|/ __||_ _|/ __|| || | |_ _|     /_\ |_ _|
  \__ \/ _ \| _| | _| \__ \ | | | (_|| __ |  | |     / _ \ | | 
  |___//_/ \_\|_|  |___||___/|___| \___||_||_|  |_|     /_/ \_\|___|"""

    TAGLINE = "Real-Time Multimodal Content Moderation & Digital Forensics Engine"

    def display_splash_screen(
        clear: bool = True,
        version: str = "1.0.0 (Production Prototype)",
        local_server: str = "http://localhost:8000",
        api_docs: str = "http://localhost:8000/docs",
        github_repo: str = "https://github.com/aayusht1711/safesight",
        system_status: str = "🟢 Live Engine [Ollama: Qwen2.5-VL-7B]",
        brand_color: str = "bold royal_blue1",
    ) -> None:
        """
        Display the SafeSight AI CLI splash screen and system runtime status panel.

        Args:
            clear: Whether to clear the terminal screen prior to rendering.
            version: System version string.
            local_server: Local server URL.
            api_docs: Swagger / OpenAPI docs URL.
            github_repo: Repository URL.
            system_status: Formatted engine status string.
            brand_color: Primary brand color used for logo and panel border.
        """
        if clear:
            try:
                console.clear()
            except Exception:
                pass

        console.print()

        # 1. Centered ASCII Art Logo
        logo_text = Text(ASCII_LOGO, style=brand_color)
        console.print(Align.center(logo_text))

        # 2. Dimmed Project Tagline
        tagline_text = Text(TAGLINE, style="dim white")
        console.print(Align.center(tagline_text))
        console.print()

        # 3. Dynamic System Status Panel
        status_table = Table(
            box=None,
            show_header=False,
            pad_edge=False,
            padding=(0, 2),
        )
        status_table.add_column("Key", style="bold white", width=18, justify="left")
        status_table.add_column("Value", style="green", justify="left")

        status_table.add_row("Version", f"[green]{version}[/green]")
        status_table.add_row("Local Server", f"[green]{local_server}[/green]")
        status_table.add_row("API Docs", f"[green]{api_docs}[/green]")
        status_table.add_row("GitHub Repo", f"[green]{github_repo}[/green]")
        status_table.add_row("System Status", f"[green]{system_status}[/green]")

        status_panel = Panel(
            Align.center(status_table),
            title=f"[{brand_color}] SYSTEM RUNTIME STATUS [/{brand_color}]",
            subtitle="[dim]SafeSight AI · Developer Console[/dim]",
            border_style=brand_color,
            box=box.ROUNDED,
            padding=(1, 3),
        )

        console.print(status_panel)
        console.print()


if __name__ == "__main__":
    display_splash_screen()
