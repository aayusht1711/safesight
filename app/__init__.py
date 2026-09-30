"""
SafeSight AI — App Package Init
================================
Intentionally minimal — avoids eager framework imports so that
schemas, processors, and pipeline modules can be imported in
lightweight test/CI environments without FastAPI installed.

To launch the server, import directly from app.main:
    from app.main import create_app, app
"""
