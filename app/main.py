"""
FastAPI application exposing exactly 5 endpoints:
1. POST /query - submit query, receive SSE stream
2. GET /jobs/{job_id}/trace - full execution trace
3. GET /eval/latest - latest eval run summary
4. POST /eval/rewrites/{rewrite_id}/review - approve/reject rewrite
5. POST /eval/rerun - trigger re-eval on failed cases
"""
import asyncio
import json
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select, desc

from app.config import settings
from app.database import (
    init_db, get_db, AsyncSession, Job, JobStatus, ExecutionLog,
    EvalRun, EvalResult, PromptRewrite, PromptRewriteStatus, ToolCallLog
)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mega_ai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized")
    yield


app = FastAPI(
    title="Mega AI - Multi-Agent LLM Orchestration System",
    description=(
        "Production-grade multi-agent system with self-improving evaluation loop, "
        "dynamic tool orchestration, and adversarial robustness testing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class QueryRequest(BaseModel):
    query: str
    stream: bool = True


class ReviewRequest(BaseModel):
    approved: bool
    reviewer_note: Optional[str] = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: Optional[str] = None


def error_response(code: str, msg: str, job_id: Optional[str] = None, status_code: int = 400):
    raise HTTPException(
        status_code=status_code,
        detail={"error_code": code, "message": msg, "job_id": job_id},
    )


# ============================================================
# ENDPOINT 1: Submit query with SSE streaming
# ============================================================

async def _sse_job_stream(job_id: str, query: str) -> AsyncIterator[str]:
    """Stream agent activity token by token via SSE.
    
    Events emitted:
    - job_start: initial event with job_id
    - token: each streamed LLM token with agent_id
    - tool_call: tool invocation with name, budget_remaining
    - tool_result: tool outcome with success, latency, accepted
    - budget_update: context budget remaining per agent
    - heartbeat: keep-alive
    - job_end: final answer + status
    """
    from app.agents.orchestrator import run_job
    from app.database import AsyncSessionLocal

    queue: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()

    async def stream_callback(agent_id: str, token: str):
        """Called by agents on every streamed token and tool event."""
        # Detect tool event markers embedded in the stream
        if token.startswith("\n[TOOL_CALL:") or token.startswith("[TOOL_CALL:"):
            await queue.put({"event": "tool_call", "agent": agent_id, "detail": token.strip()})
        elif token.startswith("\n[TOOL_RESULT:") or token.startswith("[TOOL_RESULT:"):
            await queue.put({"event": "tool_result", "agent": agent_id, "detail": token.strip()})
        else:
            await queue.put({
                "event": "token",
                "agent": agent_id,
                "token": token,
            })

    # Yield initial job_id event
    yield f"data: {json.dumps({'event': 'job_start', 'job_id': job_id, 'query': query[:100]})}\n\n"

    # Run job in background task
    async def run():
        try:
            await run_job(job_id, query, stream_callback=stream_callback)
        except Exception as e:
            await queue.put({"event": "error", "message": str(e)})
        finally:
            done.set()

    task = asyncio.create_task(run())

    # Stream events as they arrive
    while not (done.is_set() and queue.empty()):
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
            yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"

    # Final state
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if job:
            yield f"data: {json.dumps({'event': 'job_end', 'job_id': job_id, 'status': job.status.value, 'final_answer': job.final_answer})}\n\n"

    yield "data: [DONE]\n\n"


@app.post(
    "/query",
    summary="Submit a query and receive streaming SSE response",
    description=(
        "Submits a query to the multi-agent pipeline. "
        "Returns Server-Sent Events stream with real-time agent activity, "
        "tool calls in flight, and context budget remaining."
    ),
    tags=["Query"],
)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if not request.query.strip():
        error_response("EMPTY_QUERY", "Query must not be empty")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, query=request.query)
    db.add(job)
    await db.commit()

    logger.info(f"Job created: {job_id} query='{request.query[:80]}'")

    if request.stream:
        return StreamingResponse(
            _sse_job_stream(job_id, request.query),
            media_type="text/event-stream",
            headers={
                "X-Job-ID": job_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming: run in background, return job_id
        from app.agents.orchestrator import run_job

        async def run_bg():
            try:
                await run_job(job_id, request.query)
            except Exception as e:
                logger.error(f"Background job {job_id} failed: {e}")

        background_tasks.add_task(run_bg)
        return {"job_id": job_id, "status": "pending", "message": "Job submitted. Poll /jobs/{job_id}/trace for results."}


# ============================================================
# ENDPOINT 2: Full execution trace for a job
# ============================================================

@app.get(
    "/jobs/{job_id}/trace",
    summary="Retrieve full execution trace for a completed job",
    description=(
        "Returns the complete execution trace: exact sequence of agent decisions, "
        "tool calls, handoffs, token usage, policy violations, and final answer."
    ),
    tags=["Trace"],
)
async def get_execution_trace(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        error_response("JOB_NOT_FOUND", f"Job '{job_id}' not found", job_id=job_id, status_code=404)

    logs_result = await db.execute(
        select(ExecutionLog)
        .where(ExecutionLog.job_id == job_id)
        .order_by(ExecutionLog.sequence_num)
    )
    logs = logs_result.scalars().all()

    tool_calls_result = await db.execute(
        select(ToolCallLog)
        .where(ToolCallLog.job_id == job_id)
        .order_by(ToolCallLog.timestamp)
    )
    tool_calls = tool_calls_result.scalars().all()

    return {
        "job_id": job_id,
        "query": job.query,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "total_tokens": job.total_tokens,
        "final_answer": job.final_answer,
        "provenance_map": job.provenance_map or [],
        "routing_plan": job.routing_plan or [],
        "error_message": job.error_message,
        "execution_trace": [
            {
                "seq": log.sequence_num,
                "timestamp": log.timestamp.isoformat(),
                "agent_id": log.agent_id,
                "event_type": log.event_type.value,
                "input_hash": log.input_hash,
                "output_hash": log.output_hash,
                "latency_ms": log.latency_ms,
                "token_count": log.token_count,
                "policy_violation": log.policy_violation,
                "data": log.data,
            }
            for log in logs
        ],
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "agent_id": tc.agent_id,
                "attempt": tc.attempt_num,
                "input": tc.input_data,
                "output": tc.output_data,
                "latency_ms": tc.latency_ms,
                "success": tc.success,
                "failure_mode": tc.failure_mode,
                "agent_accepted": tc.agent_accepted,
                "timestamp": tc.timestamp.isoformat(),
            }
            for tc in tool_calls
        ],
    }


# ============================================================
# ENDPOINT 3: Latest eval run summary
# ============================================================

@app.get(
    "/eval/latest",
    summary="Retrieve latest eval run summary by test category and scoring dimension",
    description=(
        "Returns the most recent evaluation run results broken down by test category "
        "(baseline, ambiguous, adversarial) and all 6 scoring dimensions."
    ),
    tags=["Evaluation"],
)
async def get_latest_eval(db: AsyncSession = Depends(get_db)):
    run_result = await db.execute(
        select(EvalRun).order_by(desc(EvalRun.triggered_at)).limit(1)
    )
    run = run_result.scalar_one_or_none()

    if not run:
        error_response("NO_EVAL_RUN", "No evaluation runs found. POST /eval/run to start one.", status_code=404)

    results_result = await db.execute(
        select(EvalResult).where(EvalResult.run_id == run.id)
    )
    results = results_result.scalars().all()

    return {
        "eval_run_id": run.id,
        "triggered_at": run.triggered_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "is_reeval": run.is_reeval,
        "reeval_of": run.reeval_of,
        "aggregate_scores": run.aggregate_scores or {},
        "test_results": [
            {
                "test_case_id": r.test_case_id,
                "category": r.test_category,
                "query": r.query[:100],
                "passed": r.passed,
                "overall_score": r.overall_score,
                "scores": {
                    "answer_correctness": {
                        "score": r.answer_correctness,
                        "justification": r.answer_correctness_justification,
                    },
                    "citation_accuracy": {
                        "score": r.citation_accuracy,
                        "justification": r.citation_accuracy_justification,
                    },
                    "contradiction_resolution": {
                        "score": r.contradiction_resolution,
                        "justification": r.contradiction_resolution_justification,
                    },
                    "tool_efficiency": {
                        "score": r.tool_efficiency,
                        "justification": r.tool_efficiency_justification,
                    },
                    "budget_compliance": {
                        "score": r.budget_compliance,
                        "justification": r.budget_compliance_justification,
                    },
                    "critique_agreement": {
                        "score": r.critique_agreement,
                        "justification": r.critique_agreement_justification,
                    },
                },
            }
            for r in results
        ],
    }


# ============================================================
# ENDPOINT 4: Submit approval/rejection for prompt rewrite
# ============================================================

@app.post(
    "/eval/rewrites/{rewrite_id}/review",
    summary="Submit human approval or rejection for a pending prompt rewrite",
    description=(
        "Allows a human to approve or reject a meta-agent proposed prompt rewrite. "
        "If approved, triggers re-evaluation on previously failed test cases. "
        "All decisions are stored with timestamps and queryable."
    ),
    tags=["Self-Improvement"],
)
async def review_rewrite(
    rewrite_id: str,
    request: ReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    rewrite = await db.get(PromptRewrite, rewrite_id)
    if not rewrite:
        error_response("REWRITE_NOT_FOUND", f"Rewrite '{rewrite_id}' not found", status_code=404)

    if rewrite.status != PromptRewriteStatus.PENDING:
        error_response(
            "REWRITE_NOT_PENDING",
            f"Rewrite is already {rewrite.status.value}",
            status_code=409,
        )

    from app.evaluation.meta_agent import approve_rewrite
    result = await approve_rewrite(rewrite_id, request.approved, request.reviewer_note)

    return {
        "rewrite_id": rewrite_id,
        "action": "approved" if request.approved else "rejected",
        "reviewer_note": request.reviewer_note,
        **result,
    }


# ============================================================
# ENDPOINT 5: Trigger re-eval on failed cases
# ============================================================

@app.post(
    "/eval/rerun",
    summary="Trigger targeted re-eval on previously failed cases",
    description=(
        "Runs evaluation only on the test cases that failed in the most recent eval run, "
        "using the latest approved prompts."
    ),
    tags=["Evaluation"],
)
async def trigger_reeval(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # Find latest eval run
    run_result = await db.execute(
        select(EvalRun).order_by(desc(EvalRun.triggered_at)).limit(1)
    )
    run = run_result.scalar_one_or_none()

    if not run:
        error_response("NO_EVAL_RUN", "No eval runs found to rerun", status_code=404)

    # Get failed case IDs
    failed_result = await db.execute(
        select(EvalResult.test_case_id)
        .where(EvalResult.run_id == run.id)
        .where(EvalResult.passed == False)
    )
    failed_ids = [r[0] for r in failed_result.all()]

    if not failed_ids:
        return {
            "message": "No failed cases in latest eval run - all tests passed!",
            "eval_run_id": run.id,
        }

    reeval_run_id = str(uuid.uuid4())

    async def run_reeval():
        from app.evaluation.harness import run_eval
        await run_eval(
            test_case_ids=failed_ids,
            is_reeval=True,
            reeval_of=run.id,
        )

    background_tasks.add_task(run_reeval)

    return {
        "message": f"Re-evaluation triggered for {len(failed_ids)} failed cases",
        "failed_case_ids": failed_ids,
        "reeval_run_id": reeval_run_id,
        "based_on_run": run.id,
    }


# ============================================================
# Helper endpoints (not in the required 5, but useful)
# ============================================================

@app.post("/eval/run", tags=["Evaluation"], summary="Trigger a full eval run (all 15 test cases)")
async def run_full_eval(background_tasks: BackgroundTasks):
    """Trigger a full evaluation run in the background."""
    run_id_holder = {}

    async def run():
        from app.evaluation.harness import run_eval
        run_id = await run_eval()
        run_id_holder["run_id"] = run_id

        # After eval, run meta-agent analysis
        from app.evaluation.meta_agent import MetaAgent
        from app.core.logger import StructuredLogger
        import uuid
        meta_logger = StructuredLogger(str(uuid.uuid4()))
        meta = MetaAgent(meta_logger)
        await meta.analyze_failures(run_id)

    background_tasks.add_task(run)
    return {"message": "Full evaluation started in background. GET /eval/latest when complete."}


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/eval/rewrites", tags=["Self-Improvement"], summary="List all prompt rewrites")
async def list_rewrites(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PromptRewrite).order_by(desc(PromptRewrite.proposed_at))
    )
    rewrites = result.scalars().all()
    return {
        "rewrites": [
            {
                "id": r.id,
                "agent_id": r.agent_id,
                "worst_dimension": r.worst_dimension,
                "status": r.status.value,
                "proposed_at": r.proposed_at.isoformat(),
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "reviewer_note": r.reviewer_note,
                "diff_preview": r.diff[:200],
                "justification_preview": r.justification[:200],
                "performance_delta": r.performance_delta,
            }
            for r in rewrites
        ]
    }
