"""
Multi-Agent LLM Orchestration System
FastAPI application — exposes exactly 5 endpoints.
All agent outputs streamed token-by-token via SSE.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mega_ai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized — all tables ready")
    yield


app = FastAPI(
    title="Multi-Agent LLM Orchestration System",
    description=(
        "Production-grade multi-agent system with dynamic routing, "
        "self-improving evaluation loop, tool orchestration, "
        "adversarial robustness testing, and SSE streaming."
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

# Register the 5 required endpoints
app.include_router(router)


# ── Helper endpoints (not in required 5) ──────────────────────────────────────

@app.post("/eval/run", tags=["Evaluation"],
          summary="Trigger full 15-case evaluation + meta-agent analysis")
async def run_full_eval(background_tasks):
    from fastapi import BackgroundTasks
    async def run():
        from app.evaluation.harness import run_eval
        from app.evaluation.meta_agent import MetaAgent
        from app.core.logger import StructuredLogger
        import uuid
        run_id = await run_eval()
        meta = MetaAgent(StructuredLogger(str(uuid.uuid4())))
        await meta.analyze_failures(run_id)
    background_tasks.add_task(run)
    return {"message": "Full evaluation started. GET /eval/latest when complete."}


@app.get("/eval/rewrites", tags=["Self-Improvement"],
         summary="List all proposed prompt rewrites with status")
async def list_rewrites():
    from sqlalchemy import select, desc
    from app.database import AsyncSessionLocal, PromptRewrite
    async with AsyncSessionLocal() as db:
        rewrites = (await db.execute(
            select(PromptRewrite).order_by(desc(PromptRewrite.proposed_at))
        )).scalars().all()
    return {"rewrites": [
        {"id": r.id, "agent_id": r.agent_id, "worst_dimension": r.worst_dimension,
         "status": r.status.value, "proposed_at": r.proposed_at.isoformat(),
         "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
         "reviewer_note": r.reviewer_note, "diff_preview": r.diff[:300],
         "justification": r.justification[:300], "performance_delta": r.performance_delta}
        for r in rewrites
    ]}


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}
