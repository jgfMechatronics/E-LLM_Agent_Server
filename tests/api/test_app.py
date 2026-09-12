import os

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from collections.abc import AsyncGenerator
from unittest.mock import patch, AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from agent.factory import AgentLockedError, AgentNotFoundError
from memory.block_crud import BlockNotFoundError, DuplicateBlockError, DuplicatePositionError
from api.app import _create_app
from api.routes import router
from tests.conftest import TEST_BASE_URL


def test_create_app_includes_router():
    """Sanity check that _create_app() wires up the routes."""
    app = _create_app()
    
    app_paths = {r.path for r in app.routes}
    router_paths = {r.path for r in router.routes}
    
    assert router_paths.issubset(app_paths)
    assert "/health" in app_paths


class TestLifespan:

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        # Fresh app instance per test - no state contamination
        self.app = _create_app()
        
        # set up mocks and handle patching
        self.mock_db_engine = MagicMock()
        self.mock_db_engine.dispose = AsyncMock()

        with (patch('api.app.create_sqlite_engine') as mock_create_engine,  # sync function
              patch('api.app.init_db', new_callable=AsyncMock) as mock_init_db):
            self.mock_create_engine = mock_create_engine
            self.mock_init_db = mock_init_db
            self.mock_create_engine.return_value = self.mock_db_engine
            yield
    
    async def startup_and_shutdown_lifespan(self) -> None:
        try:
            async with LifespanManager(self.app):  # Triggers ASGI lifespan startup/shutdown
                pass
        finally:
            # lifespan shutdown should have disposed engine
            self.mock_db_engine.dispose.assert_called_once()

    async def test_happy_path(self):
        await self.startup_and_shutdown_lifespan()

        expected_db_path = os.environ["AGENT_HOME_DB_PATH"]
        
        self.mock_create_engine.assert_called_once_with(expected_db_path)
        self.mock_init_db.assert_called_once_with(self.mock_db_engine)

        assert self.app.state.engine is self.mock_db_engine
        assert self.app.state.agent_app_state_reg == {}
        # teardown asserts cleanup activity

    async def test_init_db_failure_still_disposes(self):
        """If init_db raises, engine.dispose() should still be called for cleanup."""
        self.mock_init_db.side_effect = RuntimeError("DB init failed")

        with pytest.raises(RuntimeError, match="DB init failed"):
            await self.startup_and_shutdown_lifespan()
        # dispose assertion happens in startup_and_shutdown_lifespan's finally block

    async def test_lockfile_blocks_startup(self, tmp_path):
        """Server refuses to start if INTEGRITY_CHECK_FAILED lockfile exists beside the DB."""
        from utils.integrity_checker import INTEGRITY_LOCKFILE_NAME
        db_path = tmp_path / "agent_home.sqlite"
        (tmp_path / INTEGRITY_LOCKFILE_NAME).touch()

        with patch('api.app.DB_PATH', str(db_path)):
            self.app = _create_app()
            with pytest.raises(RuntimeError, match="Integrity check failed"):
                async with LifespanManager(self.app):
                    pass

    async def test_no_lockfile_allows_startup(self, tmp_path):
        """Server starts normally when no lockfile is present."""
        with patch('api.app.DB_PATH', str(tmp_path / "agent_home.sqlite")):
            self.app = _create_app()
            await self.startup_and_shutdown_lifespan()


class _BaseAppClientTest:
    """Shared setup for tests that need an AsyncClient wrapping a fresh _create_app() instance."""

    def _add_test_routes(self) -> None:
        """Override to register additional test routes on self.app before client creation."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_client(self) -> AsyncGenerator[None, None]:
        self.app = _create_app()
        self._add_test_routes()
        async with AsyncClient(
            transport=ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url=TEST_BASE_URL,
        ) as client:
            self.client = client
            yield


class TestExceptionHandlers(_BaseAppClientTest):
    """App-level exception handlers map domain exceptions to HTTP responses."""

    def _add_test_routes(self) -> None:
        @self.app.get("/test-not-found")
        async def _raise_not_found():
            raise AgentNotFoundError("agent 'x' not found")

        @self.app.get("/test-locked")
        async def _raise_locked():
            raise AgentLockedError("agent 'x' is locked")

        @self.app.get("/test-block-not-found")
        async def _raise_block_not_found():
            raise BlockNotFoundError("block 'y' not found")

        @self.app.get("/test-duplicate-block")
        async def _raise_duplicate_block():
            raise DuplicateBlockError("block 'z' already exists")

        @self.app.get("/test-duplicate-position")
        async def _raise_duplicate_position():
            raise DuplicatePositionError("position 0 already held")

        @self.app.get("/test-unexpected")
        async def _raise_unexpected():
            raise RuntimeError("something broke")

    @pytest.mark.parametrize("path,expected_status,error_msg", [
        ("/test-not-found", 404, "AgentNotFoundError: agent 'x' not found"),
        ("/test-locked", 423, "AgentLockedError: agent 'x' is locked"),
        ("/test-block-not-found", 404, "BlockNotFoundError: block 'y' not found"),
        ("/test-duplicate-block", 422, "DuplicateBlockError: block 'z' already exists"),
        ("/test-duplicate-position", 422, "DuplicatePositionError: position 0 already held"),
        ("/test-unexpected", 500, "RuntimeError: something broke"),
    ])
    async def test_maps_domain_exception_to_http(
        self, path: str, expected_status: int, error_msg: str
    ):
        """Domain exceptions map to appropriate HTTP status codes with detail string."""
        response = await self.client.get(path)
        assert response.status_code == expected_status
        assert response.json()["detail"] == error_msg


class TestTrustedHost(_BaseAppClientTest):
    """TrustedHostMiddleware rejects requests with unexpected Host headers."""

    @pytest.mark.parametrize("host,expected_status", [
        ("localhost", 200),
        ("127.0.0.1", 200),
        ("localhost:8000", 200),  # port stripped before comparison — all ports on allowed hosts pass
        ("evil.com", 400),
        ("notlocalhost", 400),
        ("localhost.but_actually_evil", 400)
    ])
    async def test_host_validation(self, host: str, expected_status: int) -> None:
        response = await self.client.get("/health", headers={"host": host})
        assert response.status_code == expected_status


class TestOriginValidation(_BaseAppClientTest):
    """OriginValidationMiddleware rejects browser cross-site requests via Origin header."""

    @pytest.mark.parametrize("origin,expected_status", [
        (None, 200),                              # no Origin — direct API call, always allowed
        ("http://localhost", 200),
        ("http://localhost:8000", 200),           # port is ignored in host comparison
        ("http://127.0.0.1", 200),
        ("http://127.0.0.1:8000", 200),
        ("http://evil.com", 403),
        ("http://notlocalhost", 403),
        ("http://localhost.evil.com", 403),       # subdomain of allowed host — must not pass
        ("http://not.localhost", 403),
        ("null", 403),                            # If the origin field is present it MUST be populated with a known valid origin.
        ("", 403)
    ])
    async def test_origin_validation(self, origin: str | None, expected_status: int) -> None:
        headers = {"origin": origin} if origin is not None else {}
        response = await self.client.get("/health", headers=headers)
        assert response.status_code == expected_status

    async def test_valid_origin_still_rejected_on_bad_host(self) -> None:
        response = await self.client.get(
            "/health", headers={"host": "evil.com", "origin": "http://localhost:8000"}
        )
        assert response.status_code == 400

    async def test_valid_host_still_rejected_on_bad_origin(self) -> None:
        response = await self.client.get(
            "/health", headers={"host": "localhost", "origin": "http://evil.com"}
        )
        assert response.status_code == 403
