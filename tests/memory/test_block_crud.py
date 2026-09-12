"""
Tests for block CRUD (memory/block_crud.py)

Read operations take (session, agent_id) — no lock required.
Write operations take (deps) — proves caller holds per-agent lock.

TODO: These should be grouped into test classes for consistency with rest of code base
"""
import asyncio

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import BlockSettings
from conftest import make_deps, SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord

from memory import block_crud
from memory.block_crud import (
    get_blocks,
    get_block,
    update_block,
    update_block_settings,
    create_block,
    delete_block,
    reorder_blocks,
    BlockNotFoundError,
    ContentExceedsLimitError,
    DuplicateBlockError,
    DuplicatePositionError,
    InvalidBlockOrderListError,
)


# --- Fixtures ---

@pytest_asyncio.fixture
async def multi_tenant_agents_with_core_memory(session: AsyncSession):
    """
    Two agents, each with their own memory blocks.
    
    Agent A has: persona (pos 0), human (pos 1), system (pos 2)
    Agent B has: persona (pos 0), notes (pos 1)
    
    Returns dict with agents and their blocks for easy test access.
    """
    # Create two agents
    agent_a = AgentRecord(name="agent-a", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Agent A instructions")
    agent_b = AgentRecord(name="agent-b", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Agent B instructions")
    session.add_all([agent_a, agent_b])
    await session.flush()
    
    # Agent A's blocks (out of order to verify position sorting, varied char_limits for limit tests)
    block_a_human = MemoryBlockRecord(agent_id=agent_a.id, label="human", content="Human info for A", description="", char_limit=500, position=1)
    block_a_persona = MemoryBlockRecord(agent_id=agent_a.id, label="persona", content="Persona for A", description="", char_limit=1000, position=0)
    block_a_system = MemoryBlockRecord(agent_id=agent_a.id, label="system", content="System for A", description="", char_limit=2000, position=2)
    
    # Agent B's blocks
    block_b_persona = MemoryBlockRecord(agent_id=agent_b.id, label="persona", content="Persona for B", description="", char_limit=2000, position=0)
    block_b_notes = MemoryBlockRecord(agent_id=agent_b.id, label="notes", content="Notes for B", description="", char_limit=2000, position=1)
    
    all_blocks = [block_a_human, block_a_persona, block_a_system, block_b_persona, block_b_notes]
    session.add_all(all_blocks)
    await session.flush()
    
    # Refresh to get server-generated values (created_at, updated_at)
    for block in all_blocks:
        await session.refresh(block)
    
    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "blocks_a": [block_a_persona, block_a_human, block_a_system],  # in position order
        "blocks_b": [block_b_persona, block_b_notes],
    }


@pytest_asyncio.fixture
async def multi_tenant_with_deps(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    """
    Extends multi_tenant fixture with AgentDeps for both agents.
    
    For testing that write operations respect agent_id boundaries.
    """
    data = multi_tenant_agents_with_core_memory
    deps_a = make_deps(session, data["agent_a"])
    deps_b = make_deps(session, data["agent_b"])
    return {**data, "deps_a": deps_a, "deps_b": deps_b}


# --- Shared read operation tests ---

@pytest.mark.parametrize("fn,args,expected", [
    pytest.param(get_blocks, (), [], id="get_blocks_returns_empty_list"),
    pytest.param(get_block, ("any_label",), None, id="get_block_returns_none"),
])
async def test_read_ops_handle_nonexistent_agent_gracefully(session: AsyncSession, fn, args, expected):
    """Both read functions should return empty/None for nonexistent agent_id, not raise."""
    nonexistent_agent_id = "agent-does-not-exist"
    result = await fn(session, nonexistent_agent_id, *args)
    assert result == expected


# --- get_blocks tests ---

async def test_get_blocks_in_order_from_correct_agent(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_id = multi_tenant_agents_with_core_memory["agent_a"].id
    expected_blocks = multi_tenant_agents_with_core_memory["blocks_a"]
    got_blocks = await get_blocks(session, agent_id)
    assert got_blocks == expected_blocks


async def test_get_blocks_returns_empty_list_for_agent_with_no_blocks(
    session: AsyncSession, multi_tenant_agents_with_core_memory: dict
):
    _ = multi_tenant_agents_with_core_memory  # this fixture being included populates DB with agents that HAVE blocks
    
    # Create a third agent with no blocks to test isolation
    agent = AgentRecord(name="empty-agent", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Empty")
    session.add(agent)
    await session.flush()
    
    got_blocks = await get_blocks(session, agent.id)
    assert got_blocks == []


# --- get_block tests ---

async def test_get_block_returns_correct_block_by_label(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_a = multi_tenant_agents_with_core_memory["agent_a"]
    expected_block = multi_tenant_agents_with_core_memory["blocks_a"][1]  # human block at position 1
    
    got_block = await get_block(session, agent_a.id, expected_block.label)
    assert got_block == expected_block


async def test_get_block_returns_none_for_nonexistent_label(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_a = multi_tenant_agents_with_core_memory["agent_a"]
    
    got_block = await get_block(session, agent_a.id, "nonexistent")
    assert got_block is None


async def test_get_block_isolates_by_agent(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    """Agent A's 'system' block should not be returned when querying Agent B."""
    agent_b = multi_tenant_agents_with_core_memory["agent_b"]
    
    # Agent A has 'system' block, Agent B does not — verify no cross-agent leakage
    got_block = await get_block(session, agent_b.id, "system")
    assert got_block is None


# --- update_block tests ---

async def test_update_block_modifies_content_and_updated_at(multi_tenant_with_deps: dict):
    """update_block should change content and bump updated_at timestamp."""
    deps = multi_tenant_with_deps["deps_a"]
    persona = multi_tenant_with_deps["blocks_a"][0]
    original_updated_at = persona.updated_at
    
    new_content = "Updated persona content."
    result = await update_block(deps, "persona", new_content)

    assert result.content == new_content
    assert result.updated_at > original_updated_at
    assert (await get_block(deps.session, deps.agent_id, persona.label)) == result # sanity check


async def test_update_block_enforces_char_limit(multi_tenant_with_deps: dict):
    """update_block should reject content exceeding the block's char_limit."""
    deps = multi_tenant_with_deps["deps_a"]
    human_block = multi_tenant_with_deps["blocks_a"][1]
    
    oversized_content = "x" * (human_block.char_limit + 1)
    
    with pytest.raises(ContentExceedsLimitError, match="new content exceeds char limit"):
        await update_block(deps, "human", oversized_content)


async def test_update_block_accepts_prefetched_block(multi_tenant_with_deps: dict, mocker):
    """update_block with block param skips fetch and uses provided block."""
    deps = multi_tenant_with_deps["deps_a"]
    persona = multi_tenant_with_deps["blocks_a"][0]
    
    spy = mocker.spy(block_crud, "get_block")
    
    new_content = "Updated via prefetched block."
    result = await update_block(deps, "persona", new_content, block=persona)
    
    assert result.content == new_content
    assert result is persona  # Same object, not a fresh fetch
    spy.assert_not_called()


class TestUpdateBlockSettings:
    """Tests for update_block_settings helper."""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, multi_tenant_with_deps: dict):
        """Common setup: extract deps and target block."""
        self.deps = multi_tenant_with_deps["deps_a"]
        self.blocks = multi_tenant_with_deps["blocks_a"]
        self.target = self.blocks[0]  # persona at position 0
    
    async def test_modifies_all_fields(self):
        """update_block_settings should update label, description, char_limit, and position."""
        original_label = self.target.label
        
        new_settings = BlockSettings(
            label="persona_renamed",
            description="New description",
            char_limit=5000,
            position=99,
        )
        result = await update_block_settings(self.deps, original_label, new_settings)
        
        # Return value matches expected settings
        assert BlockSettings.from_record(result) == new_settings
        
        # Old label gone, new label exists with correct settings
        assert await get_block(self.deps.session, self.deps.agent_id, original_label) is None
        fetched = await get_block(self.deps.session, self.deps.agent_id, "persona_renamed")
        assert BlockSettings.from_record(fetched) == new_settings

    async def test_position_none_keeps_current(self):
        """When settings.position is None, should keep the block's current position."""
        original_position = self.target.position
        
        new_settings = BlockSettings(
            label=self.target.label,
            description="Changed description",
            char_limit=self.target.char_limit,
            position=None,  # Explicitly None — should keep current
        )
        result = await update_block_settings(self.deps, self.target.label, new_settings)
        
        # Position preserved, other fields updated
        expected = new_settings.model_copy(update={"position": original_position})
        assert BlockSettings.from_record(result) == expected
        
        # DB matches
        fetched = await get_block(self.deps.session, self.deps.agent_id, self.target.label)
        assert BlockSettings.from_record(fetched) == expected

    async def test_duplicate_label_raises(self):
        """Renaming to an existing label should raise DuplicateBlockError."""
        # Try to rename "persona" to "human" (which exists)
        new_settings = BlockSettings(label="human", description="", char_limit=20000)
        
        with pytest.raises(DuplicateBlockError):
            await update_block_settings(self.deps, "persona", new_settings)

    async def test_duplicate_position_raises(self):
        """Setting position to one already used should raise DuplicatePositionError."""
        # "persona" is at position 0, "human" is at position 1
        # Try to move "human" to position 0
        new_settings = BlockSettings(label="human", description="", char_limit=20000, position=0)
        
        with pytest.raises(DuplicatePositionError):
            await update_block_settings(self.deps, "human", new_settings)

    async def test_rejects_char_limit_below_content_length(self):
        """Reducing char_limit below current content length should raise ContentExceedsLimitError."""
        # First, put some content in the block
        await update_block(self.deps, self.target.label, "x" * 100)
        
        # Now try to reduce char_limit below content length — should fail
        new_settings = BlockSettings(
            label=self.target.label,
            description=self.target.description,
            char_limit=10,  # Less than 100 chars of content
        )
        
        with pytest.raises(ContentExceedsLimitError):
            await update_block_settings(self.deps, self.target.label, new_settings)


# --- create_block tests ---

async def test_create_block_inserts_with_defaults(multi_tenant_with_deps: dict):
    """create_block with minimal args should use correct defaults."""
    deps = multi_tenant_with_deps["deps_a"]
    
    result = await create_block(deps, BlockSettings(label="notes"), content="Some notes")

    assert result.label == "notes"
    assert result.content == "Some notes"
    assert result.description == ""  # default
    assert result.char_limit == 20000  # default
    assert result.agent_id == deps.agent_id
    
    # Verify it's actually in the database
    fetched = await get_block(deps.session, deps.agent_id, "notes")
    assert result == fetched


async def test_create_block_with_duplicate_label_raises(multi_tenant_with_deps: dict):
    """create_block should fail when label already exists for this agent."""
    deps = multi_tenant_with_deps["deps_a"]
    
    # "persona" already exists from fixture
    with pytest.raises(DuplicateBlockError, match="already exists"):
        await create_block(deps, BlockSettings(label="persona"), content="Duplicate!")


async def test_create_block_auto_assigns_position_at_end(multi_tenant_with_deps: dict):
    """create_block without explicit position should append after existing blocks."""
    deps = multi_tenant_with_deps["deps_a"]
    existing_blocks = multi_tenant_with_deps["blocks_a"]
    max_existing_position = max(b.position for b in existing_blocks)
    
    result = await create_block(deps, BlockSettings(label="notes"))
    
    assert result.position == max_existing_position + 1


async def test_create_block_on_agent_with_no_blocks(session: AsyncSession, agent_record: AgentRecord):
    """create_block on agent with no blocks should assign position 0."""
    deps = make_deps(session, agent_record)
    result = await create_block(deps, BlockSettings(label="first_block"))
    assert result.position == 0


async def test_create_block_with_explicit_position(multi_tenant_with_deps: dict):
    """create_block with explicit position should use that position."""
    deps = multi_tenant_with_deps["deps_a"]
    
    result = await create_block(deps, BlockSettings(label="notes", position=99))
    
    assert result.position == 99


async def test_create_block_with_duplicate_position_raises(multi_tenant_with_deps: dict):
    """create_block should fail when position already exists for this agent."""
    deps = multi_tenant_with_deps["deps_a"]
    
    # Position 0 already taken by "persona" from fixture
    with pytest.raises(IntegrityError):
        await create_block(deps, BlockSettings(label="notes", position=0))


# --- delete_block tests ---

async def test_delete_block_removes_block(multi_tenant_with_deps: dict):
    """delete_block should remove the block from the database."""
    deps = multi_tenant_with_deps["deps_a"]
    
    # Verify block exists before delete
    before = await get_block(deps.session, deps.agent_id, "persona")
    assert before is not None
    
    await delete_block(deps, "persona")

    # Verify block is gone
    after = await get_block(deps.session, deps.agent_id, "persona")
    assert after is None


# --- Shared: operations on nonexistent blocks ---

@pytest.mark.parametrize("operation,args", [
    pytest.param(update_block, ("nonexistent", "content"), id="update_block"),
    pytest.param(update_block_settings, ("nonexistent", BlockSettings(label="new")), id="update_block_settings"),
    pytest.param(delete_block, ("nonexistent",), id="delete_block"),
])
async def test_write_op_raises_on_nonexistent_block(multi_tenant_with_deps: dict, operation, args):
    """Write operations should raise BlockNotFoundError when block doesn't exist."""
    deps = multi_tenant_with_deps["deps_a"]
    
    with pytest.raises(BlockNotFoundError, match="not found"):
        await operation(deps, *args)


# --- reorder_blocks tests ---

async def test_reorder_blocks_assigns_positions_by_list_order(multi_tenant_with_deps: dict):
    """reorder_blocks should assign positions 0, 1, 2... based on list order."""
    deps = multi_tenant_with_deps["deps_a"]
    # Fixture has: persona (pos 0), human (pos 1), system (pos 2)
    # Reverse the order
    intended_order = ["system", "human", "persona"]
    await reorder_blocks(deps, intended_order)

    # Fetch fresh from DB to verify
    blocks = await get_blocks(deps.session, deps.agent_id)
    labels_in_order = [b.label for b in blocks]
    
    assert labels_in_order == intended_order
    assert blocks[0].position == 0
    assert blocks[1].position == 1
    assert blocks[2].position == 2


@pytest.mark.parametrize("incomplete_list,error_match", [
    pytest.param(["persona", "human"], "missing", id="missing_block"),  # missing "system"
    pytest.param(["persona", "human", "system", "nonexistent"], "unknown", id="unknown_label"),
    pytest.param(["persona", "nonexistent"], "missing.*unknown", id="missing_and_unknown"),
])
async def test_reorder_blocks_validates_label_list(multi_tenant_with_deps: dict, incomplete_list, error_match):
    """reorder_blocks should reject lists that don't exactly match agent's blocks."""
    deps = multi_tenant_with_deps["deps_a"]
    
    with pytest.raises(InvalidBlockOrderListError, match=error_match):
        await reorder_blocks(deps, incomplete_list)


# --- Multi-tenant isolation (common to all write ops) ---

async def test_write_operations_respect_agent_isolation(multi_tenant_with_deps: dict):
    """
    Write operations should only affect blocks for the specified agent_id.
    
    This tests that deps.agent_id properly scopes all write operations,
    preventing cross-agent contamination.
    """
    data = multi_tenant_with_deps
    deps_a = data["deps_a"]
    deps_b = data["deps_b"]
    
    # Snapshot Agent B's state before we modify Agent A
    b_blocks_before = await get_blocks(deps_b.session, deps_b.agent_id)
    b_snapshot_before = [(b.label, b.content, b.position) for b in b_blocks_before]
    
    # Perform all write operations on Agent A
    await update_block(deps_a, "persona", "Modified A's persona")
    await update_block_settings(deps_a, "human", BlockSettings(label="human_renamed", description="new desc", char_limit=5000))
    await create_block(deps_a, BlockSettings(label="new_block"), content="New for A")
    await delete_block(deps_a, "system")  # Agent A has system block
    await reorder_blocks(deps_a, ["human_renamed", "persona", "new_block"])
    # All writes commit via deps_a's session, which expires ALL records in the session (including deps_b's)
    # In prod, seperate agents have seperate sessions
    await deps_b.session.refresh(deps_b._agent_record)

    # Verify Agent B is completely unaffected
    b_blocks_after = await get_blocks(deps_b.session, deps_b.agent_id)
    b_snapshot_after = [(b.label, b.content, b.position) for b in b_blocks_after]
    
    assert b_snapshot_after == b_snapshot_before, "Agent B was modified by operations on Agent A"


# --- Commit behavior (common to all write ops) ---

@pytest.mark.parametrize("write_op,call_args,returns_record", [
    pytest.param(update_block, ("persona", "new content"), True, id="update_block"),
    pytest.param(update_block_settings, ("persona", BlockSettings(label="persona_new", description="new")), True, id="update_block_settings"),
    pytest.param(create_block, (BlockSettings(label="new_block"),), True, id="create_block"),
    pytest.param(delete_block, ("persona",), False, id="delete_block"),
    pytest.param(reorder_blocks, (["system", "human", "persona"],), False, id="reorder_blocks"),
])
async def test_write_ops_commit_and_refresh_by_default(multi_tenant_with_deps, write_op, call_args, returns_record):
    """Write ops with commit=True (default) should commit and refresh returned objects."""
    deps = multi_tenant_with_deps["deps_a"]

    result = await write_op(deps, *call_args)

    if returns_record:
        # Accessing attributes proves refresh happened (would raise MissingGreenlet otherwise)
        _ = result.label
        _ = result.content

    # No pending changes proves commit happened
    assert not deps.session.new
    assert not deps.session.dirty
    assert not deps.session.deleted


@pytest.mark.parametrize("write_op,call_args,fetch_record,change_applied", [
    pytest.param(
        update_block,
        ("persona", "modified"),
        lambda deps: get_block(deps.session, deps.agent_id, "persona"),
        lambda block: block.content == "modified",
        id="update_block",
    ),
    pytest.param(
        update_block_settings,
        ("persona", BlockSettings(label="persona_renamed", description="new desc")),
        lambda deps: get_block(deps.session, deps.agent_id, "persona_renamed"),
        lambda block: block is not None and block.description == "new desc",
        id="update_block_settings",
    ),
    pytest.param(
        create_block,
        (BlockSettings(label="new_block"),),
        lambda deps: get_block(deps.session, deps.agent_id, "new_block"),
        lambda block: block is not None,
        id="create_block",
    ),
    pytest.param(
        delete_block,
        ("persona",),
        lambda deps: get_block(deps.session, deps.agent_id, "persona"),
        lambda block: block is None,
        id="delete_block",
    ),
    pytest.param(
        reorder_blocks,
        (["system", "human", "persona"],),
        lambda deps: get_blocks(deps.session, deps.agent_id),
        lambda blocks: [b.label for b in blocks] == ["system", "human", "persona"],
        id="reorder_blocks",
    ),
])
async def test_write_ops_flush_only_when_commit_false(
    multi_tenant_with_deps, write_op, call_args, fetch_record, change_applied
):
    """Write ops with commit=False should flush but not commit — rollback undoes changes."""
    deps = multi_tenant_with_deps["deps_a"]

    # Start a savepoint so rollback doesn't undo fixture data
    savepoint = await deps.session.begin_nested()

    await write_op(deps, *call_args, commit=False)

    # Change is visible within transaction (flushed)
    current = await fetch_record(deps)
    assert change_applied(current), "Change should be visible after flush"

    # Explicitly rollback the savepoint
    await savepoint.rollback()

    # After rollback, change should be undone
    reverted = await fetch_record(deps)
    assert not change_applied(reverted), "Change should be undone after rollback"
