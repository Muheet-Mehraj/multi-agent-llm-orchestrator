"""
API route handlers - separated from main.py for clean structure.
All 5 required endpoints are registered here.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import uuid
from datetime import datetime

from app.database import (
    get_db, Job, JobStatus, ExecutionLog, EvalRun, EvalResult,
    PromptRewrite, PromptRewriteStatus, ToolCallLog, AsyncSessionLocal
)
from app.streaming.sse import stream_pipeline, sse_job_end
from app.agents.orchestrator import run_job


router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    stream: bool = True


class ReviewRequest(BaseModel):
    approved: bool
    reviewer_note: Optional[str] = None


def _err(code: str, msg: str, job_id=None, status=400):
    raise HTTPException(status_code=status, detail={
        "error_code": code, "message": msg, "job_id": job_id
    })


# ── Endpoint 1: POST /query ───────────────────────────────────────────────────

@router.post("/query", tags=["Query"],
             summary="Submit query — receive SSE stream with real-time agent activity")
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks,
                       db: AsyncSession = Depends(get_db)):
    if not request.query.strip():
        _err("EMPTY_QUERY", "Query must not be empty")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, query=request.query)
    db.add(job)
    await db.commit()

    if request.stream:
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        async def stream_callback(agent_id: str, token: str):
            await queue.put({"agent": agent_id, "token": token})

        async def run():
            try:
                await run_job(job_id, request.query, stream_callback=stream_callback)
            except Exception as e:
                await queue.put({"agent": "system", "token": f"[ERROR:{e}]"})
            finally:
                done.set()

        background_tasks.add_task(run)

        async def full_stream():
            async for chunk in stream_pipeline(job_id, request.query, queue, done):
                yield chunk
            # Final state
            async with AsyncSessionLocal() as session:
                job_rec = await session.get(Job, job_id)
                if job_rec:
                    yield sse_job_end(job_id, job_rec.status.value, job_rec.final_answer)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            full_stream(),
            media_type="text/event-stream",
            headers={"X-Job-ID": job_id, "Cache-Control": "no-cache"},
        )

    # Non-streaming: launch async job directly on the event loop,
    # independent of request lifecycle.
    async def debug_run():
        try:
            await run_job(job_id, request.query)
        except Exception as e:
            import traceback
            print("\n===== RUN_JOB ERROR =====")
            traceback.print_exc()
            print("=========================\n")

    asyncio.create_task(debug_run())
    return {"job_id": job_id, "status": "pending"}


# ── Endpoint 2: GET /jobs/{job_id}/trace ──────────────────────────────────────

@router.get("/jobs/{job_id}/trace", tags=["Trace"],
            summary="Full execution trace for a completed job")
async def get_trace(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        _err("JOB_NOT_FOUND", f"Job '{job_id}' not found", job_id=job_id, status=404)

    logs = (await db.execute(
        select(ExecutionLog).where(ExecutionLog.job_id == job_id)
        .order_by(ExecutionLog.sequence_num)
    )).scalars().all()

    tools = (await db.execute(
        select(ToolCallLog).where(ToolCallLog.job_id == job_id)
        .order_by(ToolCallLog.timestamp)
    )).scalars().all()

    return {
        "job_id": job_id, "query": job.query, "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "total_tokens": job.total_tokens, "final_answer": job.final_answer,
        "provenance_map": job.provenance_map or [],
        "routing_plan": job.routing_plan or [],
        "execution_trace": [
            {"seq": l.sequence_num, "timestamp": l.timestamp.isoformat(),
             "agent_id": l.agent_id, "event_type": l.event_type.value,
             "input_hash": l.input_hash, "output_hash": l.output_hash,
             "latency_ms": l.latency_ms, "token_count": l.token_count,
             "policy_violation": l.policy_violation, "data": l.data}
            for l in logs
        ],
        "tool_calls": [
            {"tool_name": t.tool_name, "agent_id": t.agent_id, "attempt": t.attempt_num,
             "input": t.input_data, "output": t.output_data, "latency_ms": t.latency_ms,
             "success": t.success, "failure_mode": t.failure_mode,
             "agent_accepted": t.agent_accepted, "timestamp": t.timestamp.isoformat()}
            for t in tools
        ],
    }


# ── Endpoint 3: GET /eval/latest ─────────────────────────────────────────────

@router.get("/eval/latest", tags=["Evaluation"],
            summary="Latest eval run summary by category and scoring dimension")
async def get_latest_eval(db: AsyncSession = Depends(get_db)):
    run = (await db.execute(
        select(EvalRun).order_by(desc(EvalRun.triggered_at)).limit(1)
    )).scalar_one_or_none()

    if not run:
        _err("NO_EVAL_RUN", "No evaluation runs found. POST /eval/run to start.", status=404)

    results = (await db.execute(
        select(EvalResult).where(EvalResult.run_id == run.id)
    )).scalars().all()

    return {
        "eval_run_id": run.id,
        "triggered_at": run.triggered_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "is_reeval": run.is_reeval, "reeval_of": run.reeval_of,
        "aggregate_scores": run.aggregate_scores or {},
        "test_results": [
            {
                "test_case_id": r.test_case_id, "category": r.test_category,
                "query": r.query[:100], "passed": r.passed, "overall_score": r.overall_score,
                "scores": {
                    "answer_correctness": {"score": r.answer_correctness, "justification": r.answer_correctness_justification},
                    "citation_accuracy": {"score": r.citation_accuracy, "justification": r.citation_accuracy_justification},
                    "contradiction_resolution": {"score": r.contradiction_resolution, "justification": r.contradiction_resolution_justification},
                    "tool_efficiency": {"score": r.tool_efficiency, "justification": r.tool_efficiency_justification},
                    "budget_compliance": {"score": r.budget_compliance, "justification": r.budget_compliance_justification},
                    "critique_agreement": {"score": r.critique_agreement, "justification": r.critique_agreement_justification},
                },
            }
            for r in results
        ],
    }


# ── Endpoint 4: POST /eval/rewrites/{id}/review ───────────────────────────────

@router.post("/eval/rewrites/{rewrite_id}/review", tags=["Self-Improvement"],
             summary="Approve or reject a pending prompt rewrite")
async def review_rewrite(rewrite_id: str, request: ReviewRequest,
                         db: AsyncSession = Depends(get_db)):
    rewrite = await db.get(PromptRewrite, rewrite_id)
    if not rewrite:
        _err("REWRITE_NOT_FOUND", f"Rewrite '{rewrite_id}' not found", status=404)
    if rewrite.status != PromptRewriteStatus.PENDING:
        _err("REWRITE_NOT_PENDING", f"Rewrite already {rewrite.status.value}", status=409)

    from app.evaluation.meta_agent import approve_rewrite
    result = await approve_rewrite(rewrite_id, request.approved, request.reviewer_note)
    return {"rewrite_id": rewrite_id,
            "action": "approved" if request.approved else "rejected",
            "reviewer_note": request.reviewer_note, **result}


# ── Endpoint 5: POST /eval/rerun ─────────────────────────────────────────────

@router.post("/eval/rerun", tags=["Evaluation"],
             summary="Re-eval on previously failed cases with latest approved prompts")
async def trigger_reeval(background_tasks: BackgroundTasks,
                         db: AsyncSession = Depends(get_db)):
    run = (await db.execute(
        select(EvalRun).order_by(desc(EvalRun.triggered_at)).limit(1)
    )).scalar_one_or_none()

    if not run:
        _err("NO_EVAL_RUN", "No eval runs found", status=404)

    failed_ids = [r[0] for r in (await db.execute(
        select(EvalResult.test_case_id)
        .where(EvalResult.run_id == run.id)
        .where(EvalResult.passed == False)
    )).all()]

    if not failed_ids:
        return {"message": "No failed cases — all passed!", "eval_run_id": run.id}

    async def run_reeval():
        from app.evaluation.harness import run_eval
        await run_eval(test_case_ids=failed_ids, is_reeval=True, reeval_of=run.id)

    background_tasks.add_task(run_reeval)
    return {"message": f"Re-eval triggered for {len(failed_ids)} failed cases",
            "failed_case_ids": failed_ids, "based_on_run": run.id}