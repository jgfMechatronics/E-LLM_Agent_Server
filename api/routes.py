"""
API routes
NOTE: Read only routes should take a session only and do not need to read or acquire the agent lock
R/W routes should take deps from get_agent_deps which acquires and holds the lock

TODO: Our current "read-only" access pattern isn't truly read-only. Read operations
take a full AsyncSession and may return mutable ORM objects still connected to the DB.
This works but violates principle of least privilege — callers that only need to read
have full write access. Worth revisiting when we have bandwidth.

TODO: We have some exception catching and mapping that doesn't use "raise ... from e", probably some places we want to add the chaining.
"""
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import ValidationError
from pydantic_ai import Agent, AgentRunResultEvent
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import agent_exists, create_agent_record, get_agent_record, get_all_agents, replace_agent_config, replace_system_instructions
from agent.types import AgentAppState, AgentConfig, AgentDeps, BlockSettings
from agent.runner import run_stateful_agent
from api.fastapi_deps import get_session_dep, get_agent_and_deps, get_agent_app_state_reg, get_agent_deps
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    CreateMemoryBlockRequest,
    MemoryBlockResponse,
    MessageRequest,
    MessagesResponse,
    SystemInstructionsResponse,
    UpdateBlockContentRequest,
)
from memory.block_crud import (
    ContentExceedsLimitError,
    DuplicateBlockError,
    InvalidBlockOrderListError,
    create_block,
    get_block,
    get_blocks,
    reorder_blocks,
    update_block,
    update_block_settings,
)
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import load_messages


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents")

# --- Helpers ---

def map_to_sse(event: Any) -> ServerSentEvent:
    """Convert a Pydantic AI streaming event to a ServerSentEvent.

    The event type name goes in the SSE 'event' field, allowing clients to filter
    with addEventListener(). The event object is passed directly to 'data' and
    serialized by FastAPI's jsonable_encoder.

    TODO: Document the SSE event types in the API readme.
    """
    if isinstance(event, AgentRunResultEvent):
        # Stream-end signal only — don't expose the result object
        return ServerSentEvent(data={}, event="AgentRunResultEvent")
    return ServerSentEvent(data=event, event=type(event).__name__)



async def _get_agent_record_or_404(session: AsyncSession, agent_id: str) -> Any:
    """Load agent record, raising 404 if not found."""
    record = await get_agent_record(session, agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return record


# --- Routes ---

@router.post("/{agent_id}/messages", response_class=EventSourceResponse)
async def handle_message(
    agent_id: str,
    body: MessageRequest,
    agent_and_deps: tuple[Agent, AgentDeps] = Depends(get_agent_and_deps),
    agent_app_state_reg: dict[str, AgentAppState] = Depends(get_agent_app_state_reg),
) -> AsyncGenerator[ServerSentEvent, None]:
    """TODO: Agent run should still be able to complete and persist in the event that client disconnects"""
    # AgentNotFoundError / AgentLockedError are translated to HTTP 404/503 by get_agent_and_deps
    agent, deps = agent_and_deps # This would be inside the try/except but cleanup assumes we have deps

    try:
        agent_app_state = agent_app_state_reg[agent_id]
        async for event in run_stateful_agent(agent=agent,
                                              deps=deps, 
                                              agent_app_state=agent_app_state,
                                              user_prompt=body.message):
            yield map_to_sse(event)
    except Exception as e:
        # TODO (low priority): put more thought into logging strategy (log levels, handler chain, structured logging)
        logger.exception("Unexpected error in handle_message for agent %s", agent_id)
        await deps.session.rollback()
        yield ServerSentEvent(
            data={"message": f"\n\nUnexpected internal server error: '{type(e).__name__}: {str(e)}'"},
            event="Error",
        )


@router.post("/{agent_id}/recompile_system_prompt")
async def recompile_system_prompt_route_handler(
    agent_id: str,
    deps: AgentDeps = Depends(get_agent_deps),
) -> bool: # We may or may not want this to return a bool
    """
    TODO: This is temp just to be able to test out memory system functionality, we may not actually want this
    if we do, need to design proper and unit test
    """

    await compile_system_prompt(deps)
    return True


@router.get("")
async def list_agents(
    session: AsyncSession = Depends(get_session_dep),
) -> list[AgentMetadataResponse]:
    """Return all agents on the server."""
    records = await get_all_agents(session)
    return [AgentMetadataResponse.from_record(r) for r in records]


@router.post("", status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Create a new agent and return its metadata.

    TODO: Compile the system prompt immediately after creation so the agent's memory blocks
    are visible from the first turn. Currently the agent starts without a compiled system
    prompt and won't see its blocks until an explicit recompile or first compaction.
    This should call compile_system_prompt (or equivalent) before returning, and the
    behaviour should be covered by tests in test_routes.py::TestCreateAgent.
    
    TODO: I was able to get an invalid model name through to the DB. Errored out when trying to do a model request but did persist
    (update: pretty sure this is fixed now)
    TODO: this route and the associated crud function do not follow our read only scheme. granted, the read only scheme is intended to
    block concurrent access to an existing agent, which doesn't apply here. we can't use our normal deps scheme to lock because there's
    no agent to construct deps for. will need to figure out, perhaps this route manually acquires the lock.
    """
    try:
        record = await create_agent_record(session, body.name, body.system_instructions, body.config)
    except IntegrityError as e:
        if "UNIQUE constraint failed: agent.name" in str(e.orig):
            # SQLite-specific string check. This is brittle but worst case user just gets a less helpful but still helpful error msg
            raise HTTPException(status_code=409, detail=f"Agent name already in use: {body.name!r}")
        raise
    return AgentMetadataResponse.from_record(record)


@router.get("/{agent_id}/system-instructions")
async def get_system_instructions(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> SystemInstructionsResponse:
    """Return the system instructions for an existing agent."""
    record = await _get_agent_record_or_404(session, agent_id)
    return SystemInstructionsResponse(system_instructions=record.system_instructions)


@router.put("/{agent_id}/config")
async def put_config(
    config: AgentConfig,
    deps: AgentDeps = Depends(get_agent_deps),
) -> AgentConfig:
    """Replace the config for an existing agent."""
    return await replace_agent_config(deps, config)


@router.put("/{agent_id}/system-instructions")
async def put_system_instructions(
    body: SystemInstructionsResponse,
    deps: AgentDeps = Depends(get_agent_deps),
) -> SystemInstructionsResponse:
    """Replace system instructions for an existing agent and recompile."""
    result = await replace_system_instructions(deps, body.system_instructions)
    return SystemInstructionsResponse(system_instructions=result)


@router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Return metadata for an existing agent."""
    record = await _get_agent_record_or_404(session, agent_id)
    return AgentMetadataResponse.from_record(record)


@router.get("/{agent_id}/config")
async def get_config(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentConfig:
    """Return the config for an existing agent."""
    record = await _get_agent_record_or_404(session, agent_id)
    return record.agent_config


@router.get("/{agent_id}/memory/blocks")
async def get_memory_blocks(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> CoreMemoryResponse:
    """Return core memory blocks for an agent."""
    blocks = await get_blocks(session, agent_id)
    # get_blocks returns empty list if agent DNE OR if agent has no blocks
    if not blocks and not (await agent_exists(session, agent_id)):
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return CoreMemoryResponse(blocks=[MemoryBlockResponse.from_record(b) for b in blocks])


@router.get("/{agent_id}/memory/blocks/{label}")
async def get_memory_block(
    agent_id: str,
    label: str,
    session: AsyncSession = Depends(get_session_dep),
) -> MemoryBlockResponse:
    """Return a single memory block by label."""
    block = await get_block(session, agent_id, label)
    if block is None:
        raise HTTPException(status_code=404, detail=f"Block {label!r} not found")
    return MemoryBlockResponse.from_record(block)


@router.post("/{agent_id}/memory/blocks", status_code=201)
async def create_memory_block(
    agent_id: str,
    body: CreateMemoryBlockRequest,
    deps: AgentDeps = Depends(get_agent_deps),
) -> MemoryBlockResponse:
    """Create a new memory block for an agent."""
    try:
        settings = BlockSettings(
            label=body.label,
            description=body.description,
            char_limit=body.char_limit,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid block settings: {e}") from e
    try:
        block = await create_block(deps, settings, body.content)
    except DuplicateBlockError as e:
        raise HTTPException(status_code=422, detail=f"Duplicate block: {e}") from e
    return MemoryBlockResponse.from_record(block)


@router.put("/{agent_id}/memory/blocks/{label}/content")
async def update_block_content(
    agent_id: str,
    label: str,
    body: UpdateBlockContentRequest,
    deps: AgentDeps = Depends(get_agent_deps),
) -> MemoryBlockResponse:
    """Update the content of a memory block."""
    try:
        block = await update_block(deps, label, body.content)
    except ContentExceedsLimitError as e:
        raise HTTPException(status_code=422, detail=f"Content exceeds char limit") from e
    return MemoryBlockResponse.from_record(block)


@router.get("/{agent_id}/memory/blocks/{label}/settings")
async def get_block_settings(
    agent_id: str,
    label: str,
    session: AsyncSession = Depends(get_session_dep),
) -> BlockSettings:
    """Get settings/metadata for a memory block."""
    block = await get_block(session, agent_id, label)
    if block is None:
        raise HTTPException(status_code=404, detail=f"Block {label!r} not found")
    return BlockSettings.from_record(block)


@router.put("/{agent_id}/memory/blocks/{label}/settings")
async def put_block_settings(
    agent_id: str,
    label: str,
    settings: BlockSettings,
    deps: AgentDeps = Depends(get_agent_deps),
) -> BlockSettings:
    """Update settings/metadata for a memory block."""
    try:
        block = await update_block_settings(deps, label, settings)
    except DuplicateBlockError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return BlockSettings.from_record(block)


@router.put("/{agent_id}/memory/blocks/order", status_code=204)
async def put_block_order(
    agent_id: str,
    labels_in_order: list[str],
    deps: AgentDeps = Depends(get_agent_deps),
) -> None:
    """Reorder memory blocks by specifying labels in desired order."""
    try:
        await reorder_blocks(deps, labels_in_order)
    except InvalidBlockOrderListError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/{agent_id}/cancel", status_code=202)
async def cancel_agent_run(
    agent_id: str,
    agent_app_state_reg: dict[str, AgentAppState] = Depends(get_agent_app_state_reg),
) -> None:
    """
    Cancel an active agent run.

    Sets the cancel_requested for the given agent if a run is currently active.
    Returns 202 if the cancel signal was sent, 409 if no run is active.

    Redundant cancels (event already set) succeed and return 202.
    """
    agent_app_state = agent_app_state_reg.get(agent_id)
    if agent_app_state is None or not agent_app_state.lock.locked():
        raise HTTPException(status_code=409, detail=f"Agent {agent_id!r} has no active run")
    agent_app_state.cancel_requested.set() # If there was a previous unserviced cancellation request, no harm in setting again


@router.get("/{agent_id}/messages")
async def get_messages(
    agent_id: str,
    full: bool = False,
    session: AsyncSession = Depends(get_session_dep),
) -> MessagesResponse:
    """
    Return conversation history. Use ?full=true for complete history.
    TODO: Another instance of bad read-only control
    """
    record = await _get_agent_record_or_404(session, agent_id)
    # TODO: Don't need agent record if requesting full, but we're likely gonna rework this anyway
    start_seq_id = 0 if full else record.context_window_start
    messages = await load_messages(session, agent_id, start_seq_id=start_seq_id)
    # Parse stored JSON and return — format TBD, this is throwaway (TODO)
    import json
    return MessagesResponse(messages=[json.loads(m.content) for m in messages])
