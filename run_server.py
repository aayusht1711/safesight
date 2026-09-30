#!/usr/bin/env python3
"""
SafeSight AI — Server Entry Point
====================================
Run with:
    python run_server.py                    # defaults from .env
    python run_server.py --reload           # hot-reload for development
    python run_server.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import get_settings

settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the SafeSight AI server")
    parser.add_argument("--host", default=settings.app_host)
    parser.add_argument("--port", type=int, default=settings.app_port)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (use 1 for GPU to avoid OOM)",
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=settings.app_log_level.lower(),
        access_log=False,  # Handled by our middleware
    )


if __name__ == "__main__":
    main()
