"""
HTTP route tests

Tests the FastAPI routes using httpx AsyncClient. Uses dependency_overrides
to inject mock factories, avoiding real DB lookups in route tests.

Fixtures are defined here (not in conftest) because only this file uses them.

Fixtures from conftest used here:
- session: Test DB session (function-scoped, rolled back after each test)
- agent_record: Pre-created agent for tests that need an existing agent
- agent_with_blocks: Agent with memory blocks attached

handle_message test is currently in agent.test_runner.py as those tests are currently entangled with the runner
"""
# Standard library
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

# Local
from agent.factory import AgentNotFoundError, LOCK_TIMEOUT_FAST
from agent.types import AgentAppState, AgentConfig, AgentDeps, BlockSettings
from api.fastapi_deps import get_agent_deps
from agent.crud import create_agent_record
from conftest import make_deps, SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord, utcnow
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse
from memory.block_crud import BlockNotFoundError, ContentExceedsLimitError, DuplicateBlockError, InvalidBlockOrderListError


# --- Test Classes ---

class TestCreateAgent:
    """POST /agents — create a new agent."""

    _NAME = "test-agent"
    _MODEL = "claude-sonnet-4-20250514"
    _VALID_BODY: dict = {
        "name": _NAME,
        "system_instructions": "Be helpful.",
        "config": {
            "model_name": _MODEL,
            "tool_names": [],
            "soft_compaction_limit": 1000,
        },
    }

    @pytest.fixture(autouse=True)
    def mock_create_agent_deps(self):
        with patch("api.routes.create_agent_record", new_callable=AsyncMock) as mock_create:
            self.mock_create_agent_record = mock_create
            yield

    async def test_creates_agent_and_returns_metadata(self, client: AsyncClient) -> None:
        """Creating an agent returns full metadata and 201 status."""
        expected_id = str(uuid4())
        DATETIME_NOW = utcnow()

        mock_record = Mock()
        mock_record.id = expected_id
        mock_record.name = self._NAME
        mock_record.agent_config.model_name = self._MODEL
        mock_record.created_at = DATETIME_NOW
        mock_record.updated_at = DATETIME_NOW
        self.mock_create_agent_record.return_value = mock_record

        expected_metadata = AgentMetadataResponse(
            id=expected_id,
            name=self._NAME,
            model=self._MODEL,
            created_at=DATETIME_NOW,
            updated_at=DATETIME_NOW,
        )

        response = await client.post("/agents", json=self._VALID_BODY)

        assert response.status_code == 201
        self.mock_create_agent_record.assert_called_once()
        assert AgentMetadataResponse.model_validate(response.json()) == expected_metadata

    async def test_returns_500_when_create_agent_fails(self, client: AsyncClient):
        """Route propagates unexpected exceptions to the app-level handler, returning 500."""
        self.mock_create_agent_record.side_effect = RuntimeError("DB failure")
        response = await client.post("/agents", json=self._VALID_BODY)
        assert response.status_code == 500
        assert response.json()["detail"] == "RuntimeError: DB failure"

    async def test_returns_400_for_invalid_config(self, client: AsyncClient):
        """Missing required fields result in 400 before route logic is reached."""
        response = await client.post(
            "/agents",
            json={"name": "incomplete"},  # missing system_instructions and config
        )
        assert response.status_code in (400, 422)  # FastAPI validation error


class TestCreateAgentNameUniqueness:
    """POST /agents — duplicate name handling."""

    async def test_returns_409_for_duplicate_name(self, client: AsyncClient):
        """Creating a second agent with an already-used name returns 409."""
        first = await client.post("/agents", json=TestCreateAgent._VALID_BODY)
        assert first.status_code == 201

        second = await client.post("/agents", json=TestCreateAgent._VALID_BODY)
        assert second.status_code == 409
        assert second.json()["detail"] == f"Agent name already in use: {TestCreateAgent._NAME!r}"


class TestGetConfig:
    """GET /agents/{agent_id}/config — agent config."""

    async def test_returns_config(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns the agent's AgentConfig as JSON."""
        response = await client.get(f"/agents/{agent_record.id}/config")

        assert response.status_code == 200
        assert AgentConfig.model_validate(response.json()) == agent_record.agent_config

    # 404 tested via parametrized TestNotFound


class TestGetSystemInstructions:
    """GET /agents/{agent_id}/system-instructions — agent system instructions."""

    async def test_returns_system_instructions(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns the agent's system instructions wrapped in a response object."""
        response = await client.get(f"/agents/{agent_record.id}/system-instructions")

        assert response.status_code == 200
        assert response.json() == {"system_instructions": agent_record.system_instructions}

    # 404 tested via parametrized TestNotFound


class _PutEndpointBase:
    """Base for PUT endpoint tests that patch a crud function and override get_agent_deps."""
    crud_patch_target: str  # subclasses define

    @staticmethod
    def _make_deps_override(agent_deps: AgentDeps):
        """Returns an async dep generator that yields agent_deps, for overriding get_agent_deps in tests."""
        async def _dep():
            yield agent_deps
        return _dep

    @pytest_asyncio.fixture(autouse=True)
    async def _setup(self, app: FastAPI, agent_deps: AgentDeps):
        app.dependency_overrides[get_agent_deps] = _PutEndpointBase._make_deps_override(agent_deps)
        self.agent_deps = agent_deps
        with patch(self.crud_patch_target, new_callable=AsyncMock) as mock:
            self.mock_crud_fcn = mock
            yield
        app.dependency_overrides.pop(get_agent_deps)


class TestPutConfig(_PutEndpointBase):
    """PUT /agents/{agent_id}/config — replace agent config."""
    crud_patch_target = "api.routes.replace_agent_config"

    @pytest.fixture()
    def mutated_config_copy(self, agent_record: AgentRecord):
        original_config = agent_record.agent_config
        return original_config.model_copy(update={"retries": original_config.retries + 1})



    async def test_calls_replace_agent_config_with_correct_args(
        self, client: AsyncClient, agent_record: AgentRecord, mutated_config_copy: AgentConfig
    ):
        """Calls replace_agent_config with the request body config, not the one already on agent_deps."""
        self.mock_crud_fcn.return_value = mutated_config_copy

        await client.put(f"/agents/{agent_record.id}/config", json=mutated_config_copy.model_dump())

        self.mock_crud_fcn.assert_called_once_with(self.agent_deps, mutated_config_copy)

    async def test_returns_200_with_echoed_config(
        self, client: AsyncClient, agent_record: AgentRecord, mutated_config_copy: AgentConfig
    ):
        """Echoes the value returned by replace_agent_config, not just the input."""
        sent_config = agent_record.agent_config
        # Return a different config to confirm we echo the crud result, not the raw input
        self.mock_crud_fcn.return_value = mutated_config_copy

        response = await client.put(
            f"/agents/{agent_record.id}/config",
            json=sent_config.model_dump(),
        )

        assert response.status_code == 200
        assert AgentConfig.model_validate(response.json()) == mutated_config_copy

    async def test_returns_422_for_invalid_config(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Returns 422 when config fails AgentConfig validation."""
        invalid_config = {"model_name": 12345}  # model_name should be string

        response = await client.put(
            f"/agents/{agent_record.id}/config",
            json=invalid_config,
        )

        assert response.status_code == 422
        # Crud function should not be called when validation fails
        self.mock_crud_fcn.assert_not_called()

    # 404/423 checked in common tests (TestNotFound, TestAgentLocked)


class TestPutSystemInstructions(_PutEndpointBase):
    """PUT /agents/{agent_id}/system-instructions — replace system instructions."""
    crud_patch_target = "api.routes.replace_system_instructions"

    async def test_calls_replace_system_instructions_with_correct_args(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Calls replace_system_instructions with the agent's deps and the instructions string."""
        instructions = "instructions updated by route"
        self.mock_crud_fcn.return_value = instructions

        await client.put(
            f"/agents/{agent_record.id}/system-instructions",
            json={"system_instructions": instructions},
        )

        self.mock_crud_fcn.assert_called_once_with(self.agent_deps, instructions)

    async def test_returns_200_with_echoed_instructions(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Echoes the value returned by replace_system_instructions, not just the input."""
        original = agent_record.system_instructions
        mutated = "mutated instructions"
        self.mock_crud_fcn.return_value = mutated

        response = await client.put(
            f"/agents/{agent_record.id}/system-instructions",
            json={"system_instructions": original},
        )

        assert response.status_code == 200
        assert response.json() == {"system_instructions": mutated}

    # 404/423 checked in common tests (TestNotFound, TestAgentLocked)


class TestListAgents:
    """GET /agents — list all agents on the server."""

    @pytest.mark.parametrize("n_agents", list(range(4)))
    async def test_returns_all_agents(
        self, client: AsyncClient, session: AsyncSession, n_agents: int
    ):
        """Returns all agents as AgentMetadataResponse objects; empty list when none exist."""
        expected = []
        for i in range(n_agents):
            record = await create_agent_record(
                session, name=f"agent-{i}", system_instructions="", config=SAMPLE_AGENT_CONFIG
            )
            expected.append(AgentMetadataResponse.from_record(record))

        response = await client.get("/agents")

        assert response.status_code == 200
        result = sorted(
            [AgentMetadataResponse(**item) for item in response.json()],
            key=lambda r: str(r.id),
        )
        assert result == sorted(expected, key=lambda r: str(r.id))


class TestGetAgent:
    """GET /agents/{agent_id} — agent metadata."""
    
    async def test_returns_agent_metadata(self, client: AsyncClient, agent_record: AgentRecord):
        """
        Returns agent metadata: name, model, created_at, updated_at.
        TODO: Should this assert that calls the appropriate internal function?
        Might be an impl detail we *don't* want to test actually
        """
        response = await client.get(f"/agents/{agent_record.id}")
        metadata = AgentMetadataResponse.model_validate(response.json())
        expected_metadata = AgentMetadataResponse(
            id=agent_record.id,
            name=agent_record.name,
            model=agent_record.agent_config.model_name,
            created_at=agent_record.created_at,
            updated_at=agent_record.updated_at,
        )

        assert response.status_code == 200
        assert metadata == expected_metadata
    
    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestGetMemoryBlocks:
    """GET /agents/{agent_id}/memory/blocks — memory blocks."""
    
    async def test_returns_memory_blocks(self, client: AsyncClient, agent_with_blocks: dict):
        """Returns blocks in position order with all schema fields present."""
        agent = agent_with_blocks["agent"]
        blocks = agent_with_blocks["blocks"]

        response = await client.get(f"/agents/{agent.id}/memory/blocks")

        assert response.status_code == 200
        actual = CoreMemoryResponse.model_validate(response.json())
        expected = CoreMemoryResponse(blocks=[
            MemoryBlockResponse(
                label=block.label,
                description=block.description,
                content=block.content,
                char_limit=block.char_limit,
                updated_at=block.updated_at,
            )
            for block in blocks
        ])
        assert actual == expected

    async def test_returns_empty_blocks_list_when_no_blocks(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns empty blocks list when agent has no memory blocks."""
        response = await client.get(f"/agents/{agent_record.id}/memory/blocks")

        assert response.status_code == 200
        data = response.json()
        assert data["blocks"] == []

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestGetMemoryBlock:
    """GET /agents/{agent_id}/memory/blocks/{label} — single memory block."""

    async def test_returns_single_block(self, client: AsyncClient, agent_with_blocks: dict):
        """Returns the requested memory block."""
        agent = agent_with_blocks["agent"]
        block = agent_with_blocks["blocks"][0]

        response = await client.get(f"/agents/{agent.id}/memory/blocks/{block.label}")

        assert response.status_code == 200
        actual = MemoryBlockResponse.model_validate(response.json())
        expected = MemoryBlockResponse.from_record(block)
        assert actual == expected

    async def test_returns_404_for_nonexistent_block(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns 404 when block label doesn't exist."""
        response = await client.get(f"/agents/{agent_record.id}/memory/blocks/nonexistent")

        assert response.status_code == 404

    # 404 for unknown agent tested via parametrized test_get_endpoints_return_404_for_unknown_agent


@pytest.mark.xfail(reason="get_messages endpoint format TBD — will be reworked once coding CLI/harness is selected")
class TestGetMessages:
    """
    GET /agents/{agent_id}/messages — conversation history.
    TODO: This is OK for now but we will likely rework the endpoint after defining what is most useful for the frontend in terms of message format
    """

    @pytest.fixture(autouse=True)
    def mock_message_loaders(self):
        """Patch message-loading functions for all TestGetMessages tests.

        Provides self.mock_load_messages for loader-routing assertions.
        """
        with (
            patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load,
        ):
            mock_load.return_value = []
            self.mock_load_messages = mock_load
            yield

    async def test_default_loads_context_window_and_returns_messages(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Without ?full=true: calls load_messages with context_window_start as start_seq_id."""
        expected_messages = [{"role": "user", "content": "test"}]
        self.mock_load_messages.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_seq_id=agent_record.context_window_start
        )

    async def test_full_true_returns_complete_history(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """With ?full=true: calls load_messages with start_seq_id=0 for full history."""
        expected_messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
        self.mock_load_messages.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages?full=true")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_seq_id=0
        )

    async def test_returns_reasonable_format(self):
        # TODO: finalize MessageItem format, constrain MessageResponse (or whatever it is) to be list[MessageItem]
        pytest.fail()

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestHealthCheck:
    """GET /health — service health."""
    
    async def test_returns_200_ok(self, client: AsyncClient):
        """Health endpoint returns 200 with status."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.xfail(reason="TODO: requires DB integration in app lifespan — need to determine how to simulate unreachable DB")
    async def test_returns_503_when_db_unreachable(self, client: AsyncClient):
        """Health endpoint should return 503 when the DB is unreachable."""
        response = await client.get("/health")
        assert response.status_code == 503


# --- Shared test data for parametrized PUT endpoint tests ---
_VALID_CONFIG_BODY = {
    "model_name": "claude-sonnet-4-20250514",
    "tool_names": [],
    "soft_compaction_limit": 1000,
}
_PUT_ENDPOINT_PARAMS = [
    ("/agents/{agent_id}/config", _VALID_CONFIG_BODY),
    ("/agents/{agent_id}/system-instructions", {"system_instructions": "some instructions"}),
    # Memory block routes
    ("/agents/{agent_id}/memory/blocks/some-label/content", {"content": "new content"}),
    ("/agents/{agent_id}/memory/blocks/some-label/settings", {"description": "new desc"}),
    ("/agents/{agent_id}/memory/blocks/order", ["label1", "label2"]),
]


class TestNotFound:
    """404 behavior for unknown agent_id across all endpoints."""

    @pytest.mark.parametrize("path", [
        "/agents/{agent_id}",
        "/agents/{agent_id}/memory/blocks",
        "/agents/{agent_id}/memory/blocks/some-label",
        "/agents/{agent_id}/messages",
        "/agents/{agent_id}/config",
        "/agents/{agent_id}/system-instructions",
        "/agents/{agent_id}/memory/blocks/some-label/settings",
    ])
    async def test_get_endpoints_return_404_for_unknown_agent(self, client: AsyncClient, path: str):
        """All GET endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.get(url)
        assert response.status_code == 404

    @pytest.mark.parametrize("path,body", _PUT_ENDPOINT_PARAMS)
    async def test_put_endpoints_return_404_for_unknown_agent(
        self, client: AsyncClient, path: str, body
    ):
        """All PUT endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.put(url, json=body)
        assert response.status_code == 404

    @pytest.mark.parametrize("path,body", [
        ("/agents/{agent_id}/memory/blocks", {"label": "test", "content": "content"}),
        # TODO: Add more POST endpoints as they're created
    ])
    async def test_post_endpoints_return_404_for_unknown_agent(
        self, client: AsyncClient, path: str, body
    ):
        """All POST endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.post(url, json=body)
        assert response.status_code == 404


class TestAgentLocked:
    """423 behavior when agent has an active run in progress."""

    @pytest.mark.parametrize("path,body", _PUT_ENDPOINT_PARAMS)
    async def test_put_endpoints_return_423_when_agent_locked(
        self, app: FastAPI, client: AsyncClient, agent_record: AgentRecord, path: str, body
    ):
        """All write endpoints that modify agent state return 423 when agent is locked."""
        # Set up locked state for this agent
        agent_state = AgentAppState()
        await agent_state.lock.acquire()
        app.state.agent_app_state_reg[agent_record.id] = agent_state

        url = path.format(agent_id=agent_record.id)
        response = await client.put(url, json=body)

        assert response.status_code == 423
        assert response.json()["detail"] == f"AgentLockedError: Agent {agent_record.id!r} did not become available within {LOCK_TIMEOUT_FAST}s"


class _MemoryBlockEndpointBase:
    """Base for memory block endpoint tests that patch a crud function and override get_agent_deps.
    
    Subclasses must define:
    - crud_patch_target: str — the crud function to patch (e.g. "api.routes.create_block")
    - crud_attr_name: str — attribute name for the mock (e.g. "mock_create_block")
    
    Provides:
    - self.agent_record: The agent from agent_with_blocks
    - self.blocks: The pre-existing blocks from agent_with_blocks
    - self.mock_session: A mock session
    - self.<crud_attr_name>: The mocked crud function
    """
    crud_patch_target: str
    crud_attr_name: str

    @pytest.fixture(autouse=True)
    def _setup(self, app: FastAPI, agent_with_blocks: dict):
        """Common setup: override get_agent_deps, patch crud function, cleanup."""
        self.agent_record = agent_with_blocks["agent"]
        self.blocks = agent_with_blocks["blocks"]
        self.mock_session = Mock()

        async def _mock_dep():
            yield make_deps(self.mock_session, self.agent_record)

        app.dependency_overrides[get_agent_deps] = _mock_dep

        with patch(self.crud_patch_target, new_callable=AsyncMock) as mock:
            setattr(self, self.crud_attr_name, mock)
            yield

        app.dependency_overrides.pop(get_agent_deps)


class TestCreateMemoryBlock(_MemoryBlockEndpointBase):
    """POST /agents/{agent_id}/memory/blocks — create a memory block."""
    crud_patch_target = "api.routes.create_block"
    crud_attr_name = "mock_create_block"

    _VALID_BODY = {
        "label": "notes",
        "content": "Some content.",
        "description": "A scratch pad.",
        "char_limit": 5000,
    }
    _MOCK_UPDATED_AT = datetime(2026, 1, 1, 12, 0, 0)

    async def test_calls_create_block_and_returns_201(self, client: AsyncClient):
        """Successful creation calls create_block and returns 201 with block data."""
        mock_block_record = MemoryBlockRecord(
            agent_id="dummy", position=0, updated_at=self._MOCK_UPDATED_AT, **self._VALID_BODY
        )
        self.mock_create_block.return_value = mock_block_record

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 201
        self.mock_create_block.assert_called_once()
        assert MemoryBlockResponse.model_validate(response.json()) == MemoryBlockResponse.from_record(mock_block_record)

    # 404 tested via parametrized TestNotFound
    # 422 for Duplicate block handled by app level handler + test

    async def test_returns_422_for_invalid_settings(self, client: AsyncClient):
        """Returns 422 when BlockSettings validation fails (e.g., char_limit <= 0)."""
        invalid_body = {**self._VALID_BODY, "char_limit": 0}

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=invalid_body,
        )

        assert response.status_code == 422

    async def test_returns_500_for_unexpected_error(self, client: AsyncClient):
        """
        Route propagates unexpected exceptions to the app-level handler, returning 500.
        Caught by an app level exception handler
        """
        self.mock_create_block.side_effect = RuntimeError("DB failure")

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "RuntimeError: DB failure"


class TestUpdateBlockContent(_MemoryBlockEndpointBase):
    """PUT /agents/{agent_id}/memory/blocks/{label}/content — update block content."""
    crud_patch_target = "api.routes.update_block"
    crud_attr_name = "mock_update_block"

    _UPDATED_CONTENT = "This is the new content."
    _MOCK_UPDATED_AT = datetime(2026, 9, 10, 12, 0, 0)

    async def test_calls_update_block_and_returns_200(self, client: AsyncClient):
        """Successful update calls update_block and returns 200 with updated block."""
        target_block = self.blocks[0]
        # Mutate fixture to represent updated state (not persisted, just mock return value)
        target_block.content = self._UPDATED_CONTENT
        target_block.updated_at = self._MOCK_UPDATED_AT
        self.mock_update_block.return_value = target_block

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/{target_block.label}/content",
            json={"content": self._UPDATED_CONTENT},
        )

        assert response.status_code == 200
        self.mock_update_block.assert_called_once()
        call_args = self.mock_update_block.call_args
        assert call_args.args[0].agent_id == self.agent_record.id
        assert call_args.args[1] == target_block.label
        assert call_args.args[2] == self._UPDATED_CONTENT
        assert MemoryBlockResponse.model_validate(response.json()) == MemoryBlockResponse.from_record(target_block)

    async def test_returns_404_for_unknown_label(self, client: AsyncClient):
        """Returns 404 when block label doesn't exist for this agent."""
        self.mock_update_block.side_effect = BlockNotFoundError("block not found")

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/nonexistent-label/content",
            json={"content": "new content"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_returns_422_for_content_over_limit(self, client: AsyncClient):
        """Returns 422 when new content exceeds char_limit."""
        self.mock_update_block.side_effect = ContentExceedsLimitError("new content exceeds char limit")

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/{self.blocks[0].label}/content",
            json={"content": "x" * 100000},
        )

        assert response.status_code == 422
        assert "char limit" in response.json()["detail"].lower()


class TestGetBlockSettings(_MemoryBlockEndpointBase):
    """GET /agents/{agent_id}/memory/blocks/{label}/settings — get block settings."""
    crud_patch_target = "api.routes.get_block"
    crud_attr_name = "mock_get_block"

    async def test_returns_settings_for_existing_block(self, client: AsyncClient):
        """Returns 200 with settings schema for existing block."""
        target_block = self.blocks[0]
        self.mock_get_block.return_value = target_block

        response = await client.get(
            f"/agents/{self.agent_record.id}/memory/blocks/{target_block.label}/settings",
        )

        assert response.status_code == 200
        data = response.json()
        assert data == {
            "label": target_block.label,
            "description": target_block.description,
            "char_limit": target_block.char_limit,
            "position": target_block.position,
        }

    async def test_returns_404_for_unknown_label(self, client: AsyncClient):
        """Returns 404 when block label doesn't exist for this agent."""
        self.mock_get_block.return_value = None

        response = await client.get(
            f"/agents/{self.agent_record.id}/memory/blocks/nonexistent-label/settings",
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestUpdateBlockSettings(_MemoryBlockEndpointBase):
    """PUT /agents/{agent_id}/memory/blocks/{label}/settings — update block settings."""
    crud_patch_target = "api.routes.update_block_settings"
    crud_attr_name = "mock_update_block_settings"

    async def test_calls_update_block_settings_and_returns_200(self, client: AsyncClient):
        """Successful update calls update_block_settings and returns 200 with helper's output.
        
        Input and output intentionally differ to verify route returns the helper's result,
        not just echoing the request. In practice they'd usually match, but the route's job
        is to pass through whatever the helper returns.
        """
        target_block = self.blocks[0]
        original_label = target_block.label
        
        # Request body — what the client sends
        request_settings = {
            "label": "renamed-block",
            "description": "Updated description.",
            "char_limit": 30000,
            "position": 5,
        }
        
        # Helper's return — intentionally different to prove route returns this, not request
        target_block.label = request_settings["label"]
        target_block.description = "Helper changed this description."  # Different!
        target_block.char_limit = request_settings["char_limit"]
        target_block.position = 99  # Different!
        self.mock_update_block_settings.return_value = target_block

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/{original_label}/settings",
            json=request_settings,
        )

        assert response.status_code == 200
        # Verify route called helper with correct args
        self.mock_update_block_settings.assert_called_once()
        call_args = self.mock_update_block_settings.call_args
        assert call_args.args[0].agent_id == self.agent_record.id  # deps
        assert call_args.args[1] == original_label  # current label from URL
        assert call_args.args[2] == BlockSettings(**request_settings)  # settings object
        # Verify route returns helper's output (which differs from request)
        assert response.json() == BlockSettings.from_record(target_block).model_dump()

    async def test_returns_404_for_unknown_label(self, client: AsyncClient):
        """Returns 404 when block label doesn't exist for this agent."""
        self.mock_update_block_settings.side_effect = BlockNotFoundError("block not found")

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/nonexistent-label/settings",
            json={"label": "nonexistent-label", "description": "new desc", "char_limit": 5000, "position": 0},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_returns_422_for_invalid_settings(self, client: AsyncClient):
        """Returns 422 when BlockSettings validation fails (FastAPI request body validation)."""
        target_block = self.blocks[0]
        invalid_body = {"label": target_block.label, "description": "", "char_limit": 0, "position": 0}

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/{target_block.label}/settings",
            json=invalid_body,
        )

        assert response.status_code == 422  # FastAPI validation


class TestReorderBlocks(_MemoryBlockEndpointBase):
    """Tests for PUT /agents/{agent_id}/memory/blocks/order"""

    crud_patch_target = "api.routes.reorder_blocks"
    crud_attr_name = "mock_reorder_blocks"

    async def test_calls_reorder_blocks_and_returns_204(self, client: AsyncClient):
        """Successful reorder calls reorder_blocks helper and returns 204 No Content."""
        new_order = ["human", "persona", "system"]
        self.mock_reorder_blocks.return_value = None  # reorder_blocks returns None

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/order",
            json=new_order,
        )

        assert response.status_code == 204
        assert response.content == b""  # No content
        # Verify route called helper with correct args
        self.mock_reorder_blocks.assert_called_once()
        call_args = self.mock_reorder_blocks.call_args
        assert call_args.args[0].agent_id == self.agent_record.id  # deps
        assert call_args.args[1] == new_order  # labels in order

    async def test_returns_422_for_invalid_label_list(self, client: AsyncClient):
        """Returns 422 when label list doesn't match agent's blocks."""
        self.mock_reorder_blocks.side_effect = InvalidBlockOrderListError("missing labels: {'system'}")

        response = await client.put(
            f"/agents/{self.agent_record.id}/memory/blocks/order",
            json=["persona", "human"],  # missing "system". Note the input here doesn't really matter as the helper is mocked
        )

        assert response.status_code == 422
        assert "missing labels" in response.json()["detail"]
