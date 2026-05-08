from __future__ import annotations
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from sqlalchemy import (
    String, Text, Float, Integer, Boolean, DateTime, JSON, ForeignKey,
    Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from app.config import settings


engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(str, enum.Enum):
    JOB_START = "job_start"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_RETRY = "tool_retry"
    HANDOFF = "handoff"
    CONTEXT_COMPRESS = "context_compress"
    BUDGET_VIOLATION = "budget_violation"
    TOKEN_STREAM = "token_stream"
    JOB_END = "job_end"
    ERROR = "error"


class PromptRewriteStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provenance_map: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    routing_plan: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    logs: Mapped[list["ExecutionLog"]] = relationship("ExecutionLog", back_populates="job")


class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    sequence_num: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    agent_id: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[EventType] = mapped_column(SAEnum(EventType))
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    policy_violation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped["Job"] = relationship("Job", back_populates="logs")


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36))
    agent_id: Mapped[str] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100))
    attempt_num: Mapped[int] = mapped_column(Integer, default=1)
    input_data: Mapped[dict] = mapped_column(JSON)
    output_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    success: Mapped[bool] = mapped_column(Boolean)
    failure_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    agent_accepted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    triggered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    prompt_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    aggregate_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    is_reeval: Mapped[bool] = mapped_column(Boolean, default=False)
    reeval_of: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    results: Mapped[list["EvalResult"]] = relationship("EvalResult", back_populates="run")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id"))
    test_case_id: Mapped[str] = mapped_column(String(100))
    test_category: Mapped[str] = mapped_column(String(50))
    job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    query: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actual_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scores
    answer_correctness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_correctness_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    citation_accuracy_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    contradiction_resolution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contradiction_resolution_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_efficiency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tool_efficiency_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    budget_compliance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_compliance_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    critique_agreement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    critique_agreement_justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    exact_prompts: Mapped[dict] = mapped_column(JSON, default=dict)
    exact_tool_calls: Mapped[dict] = mapped_column(JSON, default=dict)
    exact_outputs: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="results")


class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    eval_run_id: Mapped[str] = mapped_column(String(36))
    agent_id: Mapped[str] = mapped_column(String(100))
    worst_dimension: Mapped[str] = mapped_column(String(100))
    original_prompt: Mapped[str] = mapped_column(Text)
    proposed_prompt: Mapped[str] = mapped_column(Text)
    diff: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    status: Mapped[PromptRewriteStatus] = mapped_column(SAEnum(PromptRewriteStatus), default=PromptRewriteStatus.PENDING)
    proposed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    performance_delta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reeval_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
