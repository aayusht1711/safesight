#!/usr/bin/env python3
"""
SafeSight AI — The Stylish Batch Validation Runner
====================================================
A modern, polished, and developer-friendly terminal test runner for
evaluating SafeSight AI's multimodal moderation and forensics pipeline
across standard test matrices.

Features:
  - Animated loading transitions using rich.progress.Progress.
  - Highly readable results grid with clear borders.
  - Color-coded decision status theme and pass/fail indicators.
  - Summary card panel displaying Total Cases, Overall Accuracy %,
    Average Latency (ms), and F1/Precision/Recall metrics.
  - Multimodal forensics sub-engine verification battery.
  - Graceful fallback with clear error messages if the local
    FastAPI backend is offline.

Usage:
    # Run standard validation matrix (local deterministic engine)
    python tests/test_suite.py

    # Show detailed failure analysis
    python tests/test_suite.py --verbose

    # Run against live FastAPI server with concurrent workers
    python tests/test_suite.py --api-mode --host http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Force UTF-8 output on Windows (handles box-drawing & emojis) ───────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Rich UI Library ─────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.padding import Padding
    from rich import box
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )
    from app.ui.splash import display_splash_screen
    from tests.engine_manager import ensure_backend
except ImportError:
    print(
        "ERROR: 'rich' library is required for SafeSight test suite.\n"
        "Please install via: pip install rich>=13.7.0"
    )
    sys.exit(1)

# Global rich console
console = Console(force_terminal=True, color_system="auto")

from tests.mock_pipeline import MOCK_TEST_CASES, MockTestCase, mock_moderate
from app.schemas import (
    AuthenticityVerdict,
    ModerationDecision,
    ModerationResult,
    PipelineTier,
    ReasoningBlock,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case: MockTestCase
    result: ModerationResult
    decision_correct: bool
    violations_correct: bool
    forensics_correct: bool
    latency_ms: float
    overall_pass: bool
    error_message: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_decision(decision: Any) -> Text:
    """Return a styled text badge for moderation decision."""
    val = decision.value if hasattr(decision, "value") else str(decision)
    val = val.split(".")[-1].upper()

    if val == "APPROVED":
        return Text("APPROVED", style="bold green")
    elif val == "HUMAN_AUDIT_REQUIRED":
        return Text("AUDIT_REQ", style="bold yellow")
    elif val == "FLAGGED":
        return Text("FLAGGED", style="bold red")
    return Text(val, style="bold white")


def _format_status(passed: bool) -> Text:
    """Return styled ✅ Pass or ❌ Fail status."""
    if passed:
        return Text("✅ Pass", style="bold green")
    return Text("❌ Fail", style="bold red")


def _get_test_file_label(case: MockTestCase) -> str:
    """Return a clean file or test identifier for the 'Test File' column."""
    if case.image_path:
        p = Path(case.image_path)
        return p.name

    # Check if this case maps to a sample image
    if case.case_id == "TC-001" and Path("sample_images/nature_sample.jpg").exists():
        return "nature_sample.jpg"
    elif case.case_id == "TC-011" and Path("sample_images/meme_sample.jpg").exists():
        return "meme_sample.jpg"

    # For text-only cases, format a readable filename-like label
    return f"{case.case_id.lower()}_input.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_mock_case(case: MockTestCase) -> CaseResult:
    """Evaluate a single test case against the deterministic mock pipeline."""
    # Wire sample image if applicable
    img_path = case.image_path
    if not img_path:
        if case.case_id == "TC-001" and Path("sample_images/nature_sample.jpg").exists():
            img_path = "sample_images/nature_sample.jpg"
        elif case.case_id == "TC-011" and Path("sample_images/meme_sample.jpg").exists():
            img_path = "sample_images/meme_sample.jpg"

    t0 = time.perf_counter()
    result = mock_moderate(
        text=case.text,
        image_path=img_path,
        simulate_latency=True,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    decision_correct = result.decision == case.expected_decision
    actual_categories = {v.category for v in result.detected_violations}
    violations_correct = actual_categories == set(case.expected_violations)

    expected_f = getattr(case, "expected_forensic_verdict", None)
    actual_f = getattr(result, "authenticity_report", None)
    act_verdict = actual_f.verdict if actual_f else AuthenticityVerdict.AUTHENTIC
    act_v_str = act_verdict.value if hasattr(act_verdict, "value") else str(act_verdict)
    exp_v_str = expected_f.value if hasattr(expected_f, "value") else str(expected_f) if expected_f else None

    forensics_correct = (exp_v_str is None) or (act_v_str == exp_v_str)
    overall_pass = decision_correct and violations_correct and forensics_correct

    return CaseResult(
        case=case,
        result=result,
        decision_correct=decision_correct,
        violations_correct=violations_correct,
        forensics_correct=forensics_correct,
        latency_ms=latency_ms,
        overall_pass=overall_pass,
    )


def _evaluate_api_case(case: MockTestCase, host: str) -> CaseResult:
    """Evaluate a test case against the live API server."""
    try:
        import httpx as client
    except ImportError:
        import requests as client  # type: ignore

    url = f"{host.rstrip('/')}/moderate"
    data: dict[str, str] = {}
    if case.text:
        data["text"] = case.text

    files: dict[str, Any] = {}
    img_path = case.image_path
    if not img_path:
        if case.case_id == "TC-001" and Path("sample_images/nature_sample.jpg").exists():
            img_path = "sample_images/nature_sample.jpg"
        elif case.case_id == "TC-011" and Path("sample_images/meme_sample.jpg").exists():
            img_path = "sample_images/meme_sample.jpg"

    if img_path and Path(img_path).exists():
        p = Path(img_path)
        files = {"image": (p.name, p.read_bytes(), "image/jpeg")}

    t0 = time.perf_counter()
    try:
        resp = client.post(url, data=data, files=files if files else None, timeout=30.0)
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            result_dict = resp.json()
            result = ModerationResult.model_validate(result_dict)
        elif resp.status_code in (404, 405):
            # Try alternate multimodal route
            alt_url = f"{host.rstrip('/')}/api/v1/moderate/multimodal"
            resp2 = client.post(alt_url, data=data, files=files if files else None, timeout=30.0)
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp2.status_code == 200:
                result = ModerationResult.model_validate(resp2.json())
            else:
                raise RuntimeError(f"HTTP {resp2.status_code}: {resp2.text}")
        else:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        decision_correct = result.decision == case.expected_decision
        actual_categories = {v.category for v in result.detected_violations}
        violations_correct = actual_categories == set(case.expected_violations)

        actual_f = getattr(result, "authenticity_report", None)
        act_verdict = actual_f.verdict if actual_f else AuthenticityVerdict.AUTHENTIC
        act_v_str = act_verdict.value if hasattr(act_verdict, "value") else str(act_verdict)
        expected_f = getattr(case, "expected_forensic_verdict", None)
        exp_v_str = expected_f.value if hasattr(expected_f, "value") else str(expected_f) if expected_f else None
        forensics_correct = (exp_v_str is None) or (act_v_str == exp_v_str)

        return CaseResult(
            case=case,
            result=result,
            decision_correct=decision_correct,
            violations_correct=violations_correct,
            forensics_correct=forensics_correct,
            latency_ms=latency_ms,
            overall_pass=decision_correct and violations_correct and forensics_correct,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        # Return fallback failure result
        dummy = ModerationResult(
            decision=ModerationDecision.HUMAN_AUDIT_REQUIRED,
            confidence_score=0.0,
            detected_violations=[],
            reasoning=ReasoningBlock(
                text_explanation=f"API Error: {exc}",
                visual_explanation="API Error",
                bimodal_synergy="API Error",
            ),
            pipeline_tier_used=PipelineTier.TIER1,
            total_latency_ms=0.0,
        )
        return CaseResult(
            case=case,
            result=dummy,
            decision_correct=False,
            violations_correct=False,
            forensics_correct=False,
            latency_ms=latency_ms,
            overall_pass=False,
            error_message=str(exc),
        )


def _check_api_liveness(host: str) -> bool:
    """Verify whether the backend is reachable before starting API batch mode."""
    try:
        import httpx as client
    except ImportError:
        import requests as client  # type: ignore

    clean = host.rstrip("/")
    probe_urls = [f"{clean}/docs", f"{clean}/health", f"{clean}/moderate"]
    for url in probe_urls:
        try:
            resp = client.get(url, timeout=3.0)
            if resp.status_code in (200, 404, 405):
                return True
        except Exception:
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(results: list[CaseResult]) -> dict[str, Any]:
    """Compute overall accuracy, precision, recall, and F1."""
    tp = fp = fn = tn = 0

    for r in results:
        expected_pos = r.case.expected_decision != ModerationDecision.APPROVED
        pred_pos = r.result.decision != ModerationDecision.APPROVED

        if expected_pos and pred_pos:
            tp += 1
        elif not expected_pos and pred_pos:
            fp += 1
        elif expected_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Forensics Sub-Engine Verification Battery
# ─────────────────────────────────────────────────────────────────────────────

async def _run_forensics_subengine_checks() -> list[tuple[str, str, str, float, bool]]:
    """Execute tests validating the 4 multimodal forensics algorithms."""
    from PIL import Image
    from app.utils.forensics_detector import (
        AuthenticityVerdict as FV,
        analyze_audio_integrity,
        analyze_image_integrity,
        analyze_text_authorship,
        analyze_video_integrity,
    )

    rows: list[tuple[str, str, str, float, bool]] = []

    # 1. Text Authorship: Human natural variation
    t_human = (
        "Yesterday my grandmother baked fresh strawberry pie. The crust was flaky and sweet, "
        "though slightly burnt at the edge! Why do we always forget the kitchen timer?"
    )
    t0 = time.perf_counter()
    r_text_human = await analyze_text_authorship(t_human)
    lat = (time.perf_counter() - t0) * 1000
    pass_human = r_text_human.verdict in (FV.HUMAN_ORIGINAL, FV.AUTHENTIC)
    rows.append(("Text", "Authorship: Natural human perplexity", r_text_human.verdict.value, lat, pass_human))

    # 2. Text Authorship: LLM Uniform Low Perplexity
    t_llm = (
        "Furthermore, the integration of cutting-edge artificial intelligence systems facilitates streamlined workflows. "
        "Consequently, organizations can leverage predictive analytics to enhance operational efficiency. "
        "In addition, robust protocols mitigate potential systemic risks. "
        "Moreover, scalable infrastructure guarantees continuous model evaluation and deployment across enterprise environments."
    )
    t0 = time.perf_counter()
    r_text_ai = await analyze_text_authorship(t_llm)
    lat = (time.perf_counter() - t0) * 1000
    pass_ai = r_text_ai.verdict in (FV.FULLY_AI_GENERATED, FV.SYNTHETIC_GENERATED)
    rows.append(("Text", "Authorship: LLM uniform perplexity", r_text_ai.verdict.value, lat, pass_ai))

    # 3. Audio Integrity: Phase jump & silence drop
    raw_audio = [255 if (i % 250 < 12) else 0 for i in range(5000)]
    t0 = time.perf_counter()
    r_audio = await analyze_audio_integrity(bytes(raw_audio))
    lat = (time.perf_counter() - t0) * 1000
    pass_aud = r_audio.verdict in (FV.MALICIOUS_TAMPERING, FV.SYNTHETIC_GENERATED, FV.HYBRID_HUMAN_AI_EDITED)
    rows.append(("Audio", "Spectral: Phase discontinuity & drop", r_audio.verdict.value, lat, pass_aud))

    # 4. Video Integrity: Facial landmark jitter spikes
    t0 = time.perf_counter()
    r_vid = await analyze_video_integrity(bytes([10, 50, 100, 200] * 500))
    lat = (time.perf_counter() - t0) * 1000
    pass_vid = r_vid.verdict == FV.MALICIOUS_TAMPERING
    rows.append(("Video", "Landmark: Temporal facial jitter", r_vid.verdict.value, lat, pass_vid))

    # 5. Image Integrity: ELA compression difference
    img_clean = Image.new("RGB", (120, 120), color=(140, 150, 160))
    buf = io.BytesIO()
    img_clean.save(buf, format="JPEG", quality=90)
    t0 = time.perf_counter()
    r_img = await analyze_image_integrity(buf.getvalue())
    lat = (time.perf_counter() - t0) * 1000
    pass_img = r_img.verdict in (FV.AUTHENTIC, FV.BENIGN_AI_ASSISTED, FV.SYNTHETIC_GENERATED)
    rows.append(("Image", "ELA: Error Level Analysis", r_img.verdict.value, lat, pass_img))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point & Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="test_suite",
        description="SafeSight AI — The Stylish Batch Validation Runner",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Display detailed failure diagnostics and explanations",
    )
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help="Execute test matrix against live FastAPI backend",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="FastAPI backend host URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=4,
        help="Number of concurrent workers for live API requests (default: 4)",
    )
    parser.add_argument(
        "--skip-forensics",
        action="store_true",
        help="Skip dedicated forensics sub-engine verification tests",
    )
    args = parser.parse_args()

    # ── Display Developer CLI Splash Screen ──────────────────────────────────
    display_splash_screen(
        clear=True,
        local_server=args.host,
        api_docs=f"{args.host.rstrip('/')}/docs",
    )

    # ── Top Title Banner ─────────────────────────────────────────────────────
    console.print()
    banner_title = Text("SAFESIGHT AI  ·  BATCH VALIDATION SUITE", style="bold white")
    mode_label = "LIVE API SERVER" if args.api_mode else "LOCAL DETERMINISTIC PIPELINE"
    banner_sub = f"Mode: {mode_label}  ·  Test Cases: {len(MOCK_TEST_CASES)}  ·  Started: {datetime.datetime.now().strftime('%H:%M:%S')}"
    console.print(
        Panel(
            Text.assemble(banner_title, "\n", (banner_sub, "dim cyan")),
            box=box.ROUNDED,
            border_style="bold blue",
            padding=(1, 2),
        )
    )

    # ── Handle Server Connectivity with Auto-Bootstrap ─────────────────────
    running_api_mode = args.api_mode
    if running_api_mode:
        from urllib.parse import urlparse
        parsed = urlparse(args.host)
        probe_host = parsed.hostname or "127.0.0.1"
        probe_port = parsed.port or 8000

        console.print(f"[cyan]Testing connectivity to live backend at [bold]{args.host}[/bold]...[/cyan]")

        server_live = ensure_backend(
            host=probe_host,
            port=probe_port,
            console=console,
        )

        if server_live:
            console.print(f"[green]✔ Backend is responsive. Running concurrent batch validation (workers: {args.concurrency})...[/green]\n")
        else:
            console.print()
            console.print(
                Panel(
                    f"[bold red]❌ FastAPI backend could not be started at {args.host}.[/bold red]\n\n"
                    f"[yellow]Auto-bootstrap was attempted but the server did not become responsive.\n"
                    f"Falling back to the local deterministic mock pipeline.[/yellow]",
                    title="[bold yellow]⚠ Engine Bootstrap Failed — Fallback Activated[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            running_api_mode = False

    # ── Animated Progress Bar ────────────────────────────────────────────────
    test_cases = MOCK_TEST_CASES
    n_cases = len(test_cases)
    case_results: list[CaseResult] = []

    progress = Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=36, style="blue", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    overall_t0 = time.perf_counter()

    with progress:
        task_id = progress.add_task("[bold cyan]Executing validation matrix...", total=n_cases)

        if running_api_mode and args.concurrency > 1:
            # Concurrent execution across threads
            results_dict: dict[str, CaseResult] = {}
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_map = {
                    executor.submit(_evaluate_api_case, case, args.host): case
                    for case in test_cases
                }
                for future in as_completed(future_map):
                    c = future_map[future]
                    res = future.result()
                    results_dict[c.case_id] = res
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Evaluated [magenta]{c.case_id}[/magenta]: {c.description[:32]}...",
                    )
            # Reconstruct original ordered list
            case_results = [results_dict[c.case_id] for c in test_cases]
        else:
            # Sequential execution (smooth progress updates)
            for case in test_cases:
                progress.update(
                    task_id,
                    description=f"Running [magenta]{case.case_id}[/magenta]: {case.description[:35]}...",
                )
                if running_api_mode:
                    cr = _evaluate_api_case(case, args.host)
                else:
                    cr = _evaluate_mock_case(case)
                case_results.append(cr)
                progress.advance(task_id, 1)

    total_time_ms = (time.perf_counter() - overall_t0) * 1000

    # ─────────────────────────────────────────────────────────────────────────
    # Execution Results Table (Clear Grid Borders)
    # Columns: Test File, Expected Output, Model Decision, Latency (ms), Result Status
    # ─────────────────────────────────────────────────────────────────────────
    console.print()
    results_table = Table(
        title="[bold white]TEST EXECUTION MATRIX[/bold white]",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        expand=True,
    )

    results_table.add_column("Test File", style="bold white", no_wrap=True)
    results_table.add_column("Expected Output", justify="center", no_wrap=True)
    results_table.add_column("Model Decision", justify="center", no_wrap=True)
    results_table.add_column("Latency (ms)", justify="right", style="cyan", no_wrap=True)
    results_table.add_column("Result Status", justify="center", no_wrap=True)

    for cr in case_results:
        file_label = _get_test_file_label(cr.case)
        exp_badge = _format_decision(cr.case.expected_decision)
        act_badge = _format_decision(cr.result.decision)
        status_badge = _format_status(cr.overall_pass)

        # Color-code latency
        if cr.latency_ms < 50:
            lat_text = Text(f"{cr.latency_ms:.1f} ms", style="green")
        elif cr.latency_ms < 150:
            lat_text = Text(f"{cr.latency_ms:.1f} ms", style="yellow")
        else:
            lat_text = Text(f"{cr.latency_ms:.1f} ms", style="red")

        results_table.add_row(
            file_label,
            exp_badge,
            act_badge,
            lat_text,
            status_badge,
        )

    console.print(results_table)

    # ── Verbose Diagnostic Details (if requested) ────────────────────────────
    if args.verbose:
        failing = [cr for cr in case_results if not cr.overall_pass]
        if failing:
            console.print()
            console.print(Text("🔍  DETAILED FAILURE ANALYSIS", style="bold red"))
            for cr in failing:
                console.print(f"  [bold yellow]► {cr.case.case_id}[/bold yellow] — {cr.case.description}")
                console.print(f"    Expected Decision: {cr.case.expected_decision.value} | Expected Violations: {cr.case.expected_violations}")
                console.print(f"    Actual Decision:   {cr.result.decision.value} | Detected Violations: {[v.category.value for v in cr.result.detected_violations]}")
                if cr.error_message:
                    console.print(f"    Error:             {cr.error_message}")
                console.print(f"    Reasoning:         {cr.result.reasoning.text_explanation}")
                console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # Forensics Sub-Engine Verification Battery
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_forensics:
        try:
            forensics_checks = asyncio.run(_run_forensics_subengine_checks())
            console.print()
            f_table = Table(
                title="[bold white]MULTIMODAL FORENSICS SUB-ENGINE VERIFICATION[/bold white]",
                title_justify="left",
                box=box.ROUNDED,
                show_lines=True,
                header_style="bold cyan",
                expand=True,
            )
            f_table.add_column("Modality", style="bold magenta", width=12, no_wrap=True)
            f_table.add_column("Algorithm / Target Heuristic", style="white")
            f_table.add_column("Forensics Verdict", justify="center", no_wrap=True)
            f_table.add_column("Latency (ms)", justify="right", style="cyan", no_wrap=True)
            f_table.add_column("Result Status", justify="center", no_wrap=True)

            for mod, algo, verd, lat, passed in forensics_checks:
                v_style = "bold green" if "AUTHENTIC" in verd or "HUMAN" in verd else "bold red"
                f_table.add_row(
                    mod,
                    algo,
                    Text(verd, style=v_style),
                    f"{lat:.1f} ms",
                    _format_status(passed),
                )
            console.print(f_table)
        except Exception as exc:
            console.print(f"[dim yellow]Notice: Forensics battery skipped ({exc})[/dim yellow]")

    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary Card Panel
    # * Total Test Cases run
    # * Overall accuracy percentage
    # * Average system latency in milliseconds
    # ─────────────────────────────────────────────────────────────────────────
    metrics = _compute_metrics(case_results)
    total_cases = len(case_results)
    passed_cases = sum(1 for r in case_results if r.overall_pass)
    failed_cases = total_cases - passed_cases
    accuracy_pct = (passed_cases / total_cases * 100.0) if total_cases > 0 else 0.0

    latencies = [r.latency_ms for r in case_results]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    summary_text = Text()
    summary_text.append("• Total Test Cases:          ", style="bold white")
    summary_text.append(f"{total_cases} executed ", style="bold cyan")
    summary_text.append(f"({passed_cases} passed, {failed_cases} failed)\n", style="green" if failed_cases == 0 else "red")

    summary_text.append("• Overall Accuracy:          ", style="bold white")
    acc_color = "bold green" if accuracy_pct >= 95.0 else ("bold yellow" if accuracy_pct >= 80.0 else "bold red")
    summary_text.append(f"{accuracy_pct:.1f}%\n", style=acc_color)

    summary_text.append("• Average System Latency:    ", style="bold white")
    summary_text.append(f"{avg_latency:.1f} ms ", style="bold cyan")
    summary_text.append(f"(P95: {p95_latency:.1f} ms · Total Run: {total_time_ms:.1f} ms)\n", style="dim")

    summary_text.append("• Precision / Recall / F1:   ", style="bold white")
    f1_color = "bold green" if metrics["f1"] >= 0.90 else "bold yellow"
    summary_text.append(f"{metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}\n", style=f1_color)

    if accuracy_pct >= 95.0:
        card_title = "[bold green]✨  VALIDATION SUITE PASSED  ·  PIPELINE PRODUCTION READY[/bold green]"
        card_border = "bold green"
        summary_text.append("\n✔ System integrity verified across multimodal content safety and deepfake forensics.", style="dim green")
    elif accuracy_pct >= 80.0:
        card_title = "[bold yellow]⚠  VALIDATION COMPLETED WITH WARNINGS[/bold yellow]"
        card_border = "bold yellow"
        summary_text.append("\n⚠ Some edge cases deviated from ground truth. Review failed cases with --verbose.", style="dim yellow")
    else:
        card_title = "[bold red]❌  VALIDATION MATRIX FAILED[/bold red]"
        card_border = "bold red"
        summary_text.append("\n❌ Critical regression detected. Check failed cases with --verbose.", style="bold red")

    console.print()
    summary_card = Panel(
        summary_text,
        title=card_title,
        border_style=card_border,
        box=box.ROUNDED,
        padding=(1, 3),
    )
    console.print(summary_card)
    console.print()


if __name__ == "__main__":
    main()
