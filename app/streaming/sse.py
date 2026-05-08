"""
Streaming module
Handles Server-Sent Events (SSE) for real-time agent output streaming.
Clients see: which agent is writing, tool calls in flight, budget remaining.
"""
import json
import asyncio
from typing import AsyncIterator


def sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def sse_token(agent_id: str, token: str) -> str:
    return sse_event({"event": "token", "agent": agent_id, "token": token})


def sse_tool_call(agent_id: str, tool_name: str, budget_remaining) -> str:
    return sse_event({
        "event": "tool_call",
        "agent": agent_id,
        "tool": tool_name,
        "budget_remaining": budget_remaining,
    })


def sse_tool_result(agent_id: str, tool_name: str, success: bool, accepted: bool, budget_remaining) -> str:
    return sse_event({
        "event": "tool_result",
        "agent": agent_id,
        "tool": tool_name,
        "success": success,
        "accepted": accepted,
        "budget_remaining": budget_remaining,
    })


def sse_job_start(job_id: str, query: str) -> str:
    return sse_event({"event": "job_start", "job_id": job_id, "query": query[:100]})


def sse_job_end(job_id: str, status: str, final_answer: str | None) -> str:
    return sse_event({
        "event": "job_end",
        "job_id": job_id,
        "status": status,
        "final_answer": final_answer,
    })


def sse_heartbeat() -> str:
    return sse_event({"event": "heartbeat"})


def sse_error(message: str, job_id: str | None = None) -> str:
    return sse_event({"event": "error", "message": message, "job_id": job_id})


async def stream_pipeline(job_id: str, query: str, queue: asyncio.Queue, done: asyncio.Event) -> AsyncIterator[str]:
    """
    Consume events from queue and yield SSE-formatted strings.
    Emits heartbeats to keep connection alive during long agent runs.
    """
    yield sse_job_start(job_id, query)

    while not (done.is_set() and queue.empty()):
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
            token = event.get("token", "")

            # Detect inline tool markers emitted by agents
            if token.startswith("[TOOL_CALL:") or token.startswith("\n[TOOL_CALL:"):
                yield sse_event({"event": "tool_call", "agent": event.get("agent"), "detail": token.strip()})
            elif token.startswith("[TOOL_RESULT:") or token.startswith("\n[TOOL_RESULT:"):
                yield sse_event({"event": "tool_result", "agent": event.get("agent"), "detail": token.strip()})
            else:
                yield sse_token(event.get("agent", "unknown"), token)

        except asyncio.TimeoutError:
            yield sse_heartbeat()
