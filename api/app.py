"""FastAPI application and lifespan"""
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from agent.factory import AgentLockedError, AgentNotFoundError
from memory.block_crud import BlockNotFoundError, DuplicateBlockError, DuplicatePositionError
from api.routes import router
from api.schemas import HealthResponse
from db.connection import create_sqlite_engine, init_db
from utils.integrity_checker import INTEGRITY_LOCKFILE_NAME

logger = logging.getLogger(__name__)

DB_PATH = os.environ["AGENT_HOME_DB_PATH"]


_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1")
_ALLOWED_ORIGINS: tuple[str, ...] = _ALLOWED_HOSTS


class OriginValidationMiddleware:
    """Reject browser-originated cross-site requests by validating the Origin header.

    Browsers send an Origin header on cross-origin requests. If it is present and does
    not resolve to an allowed host, the request is rejected with a 403. Requests without
    an Origin header (direct API calls from toad, curl, scripts) are always allowed.

    This prevents a malicious web page from making requests to the server on behalf of
    a user who happens to visit it while the server is running locally — a relevant concern
    as endpoints become more capable.

    Note: TrustedHostMiddleware handles DNS rebinding; this middleware handles CSRF.
    Together they cover the two main browser-based attack vectors.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        origin = Headers(scope=scope).get("origin")
        if origin is not None and urlparse(origin).hostname not in _ALLOWED_ORIGINS:
            await Response(status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _handle_background_task_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """
    The user is not notified about exceptions in background tasks, and they will typically occur
    in contexts where agents may be running unmonitored, and critically, may be kicking off new runs for each other
    in a degraded state.
    Kill server to prevent dammage
    """
    exc = context.get("exception")
    logger.critical("Unhandled exception in background task, shutting down: %s", exc, exc_info=exc)
    os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lockfile = Path(DB_PATH).parent / INTEGRITY_LOCKFILE_NAME
    if lockfile.exists():
        msg = (
            f"Integrity check failed — server startup blocked. "
            f"Inspect {lockfile.parent / 'integrity_checker_results.txt'} and resolve all issues, "
            f"then delete {lockfile} to allow the server to start."
        )
        logger.critical(msg)
        raise RuntimeError(msg)
    asyncio.get_running_loop().set_exception_handler(_handle_background_task_exception)
    engine = create_sqlite_engine(DB_PATH)
    try:
        await init_db(engine)
        app.state.engine = engine
        yield
    finally:
        await engine.dispose()


# App-level handlers commonize exception → HTTP response mapping. Without them, each route or
# dep that raises these exceptions would need its own mapping, making it easy for behavior to
# drift across the codebase. Handlers here apply consistently regardless of raise site.
# TODO: commonize with other exception formatting in the codebase
def _exc_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


async def agent_not_found_handler(request: Request, exc: AgentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _exc_detail(exc)})


async def agent_locked_handler(request: Request, exc: AgentLockedError) -> JSONResponse:
    return JSONResponse(status_code=423, content={"detail": _exc_detail(exc)})


async def block_not_found_handler(request: Request, exc: BlockNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _exc_detail(exc)})


async def duplicate_block_handler(request: Request, exc: DuplicateBlockError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _exc_detail(exc)})


async def duplicate_position_handler(request: Request, exc: DuplicatePositionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _exc_detail(exc)})


# Since this app is intended for self hosters, we want exception details to pass on to the client
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": _exc_detail(exc)})


def _create_app() -> FastAPI:
    """Factory function for creating the FastAPI app. Enables fresh instances per test."""
    app = FastAPI(lifespan=lifespan)
    app.state.agent_app_state_reg = {}
    app.include_router(router)
    app.add_exception_handler(AgentNotFoundError, agent_not_found_handler)
    app.add_exception_handler(AgentLockedError, agent_locked_handler)
    app.add_exception_handler(BlockNotFoundError, block_not_found_handler)
    app.add_exception_handler(DuplicateBlockError, duplicate_block_handler)
    app.add_exception_handler(DuplicatePositionError, duplicate_position_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    # Prevent DNS rebinding (validates Host header)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_ALLOWED_HOSTS)
    # Prevent CSRF (validates Origin header on browser-originated requests)
    app.add_middleware(OriginValidationMiddleware)

    @app.get("/health")
    async def health() -> HealthResponse:
        # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
        return HealthResponse(status="ok")
    
    return app


app = _create_app()
