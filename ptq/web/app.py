from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_WEB_DIR = Path(__file__).parent


def setup_logging(*, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ptq").setLevel(level)
    if not debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def create_debug_app() -> FastAPI:
    """Factory for uvicorn --reload (no args)."""
    return create_app(debug=True)


def create_app(*, debug: bool = False) -> FastAPI:
    setup_logging(debug=debug)
    app = FastAPI(title="ptq — PyTorch Job Queue")

    app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

    from ptq.web.routes import router

    app.include_router(router)
    return app
