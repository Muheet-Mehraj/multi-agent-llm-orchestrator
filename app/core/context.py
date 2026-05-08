"""
Shared context object schema. ALL inter-agent communication passes through this.
Agents NEVER call each other directly. The orchestrator mediates all handoffs.
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
import uuid


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str  # "research", "compute", "synthesize", "validate"
    description: str
    dependencies: list[str] = Field(default_factory=list)  # list of subtask IDs
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    assigned_agent: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float
    retrieval_hop: int  # 1 or 2+ for multi-hop


class Claim(BaseModel):
    text: str
    confidence: float  # 0.0-1.0
    source_agent: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    flagged: bool = False
    flag_reason: Optional[str] = None


class ToolCall(BaseModel):
    tool_name: str
    input: dict
    output: Optional[dict] = None
    latency_ms: float = 0.0
    success: bool = False
    failure_mode: Optional[str] = None  # "timeout", "empty", "malformed"
    attempt_num: int = 1
    agent_accepted: Optional[bool] = None


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    agent_id: str
    output_type: str
    content: Any
    token_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ContextBudget(BaseModel):
    agent_id: str
    max_tokens: int
    used_tokens: int = 0
    violations: list[str] = Field(default_factory=list)

    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

    def check_and_consume(self, tokens: int, label: str = "") -> bool:
        """Returns True if within budget and consumes. False if violation."""
        if self.used_tokens + tokens > self.max_tokens:
            self.violations.append(
                f"Budget violation: tried to add {tokens} tokens (label='{label}') "
                f"but only {self.remaining()} remaining out of {self.max_tokens}"
            )
            return False
        self.used_tokens += tokens
        return True


class SharedContext(BaseModel):
    """The single shared context object mediating all agent communication."""
    job_id: str
    original_query: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Orchestrator routing plan
    routing_plan: Optional[dict] = None
    routing_justification: Optional[str] = None

    # Decomposition agent outputs
    sub_tasks: list[SubTask] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)

    # Retrieval agent outputs
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_reasoning: Optional[str] = None

    # All agent outputs keyed by agent_id
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)

    # Critique agent outputs
    claims: list[Claim] = Field(default_factory=list)
    flagged_spans: list[dict] = Field(default_factory=list)
    critique_summary: Optional[str] = None

    # Synthesis agent outputs
    final_answer: Optional[str] = None
    provenance_map: list[ProvenanceEntry] = Field(default_factory=list)
    contradictions_resolved: list[dict] = Field(default_factory=list)

    # Tool call history
    tool_call_history: list[ToolCall] = Field(default_factory=list)

    # Context budgets per agent
    budgets: dict[str, ContextBudget] = Field(default_factory=dict)

    # Compression log
    compression_events: list[dict] = Field(default_factory=list)

    # Errors and policy violations
    policy_violations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def get_budget(self, agent_id: str) -> Optional[ContextBudget]:
        return self.budgets.get(agent_id)

    def register_budget(self, agent_id: str, max_tokens: int):
        self.budgets[agent_id] = ContextBudget(agent_id=agent_id, max_tokens=max_tokens)

    def add_tool_call(self, tool_call: ToolCall):
        self.tool_call_history.append(tool_call)

    def get_completed_subtasks(self) -> list[SubTask]:
        return [t for t in self.sub_tasks if t.status == "completed"]

    def can_execute_subtask(self, task: SubTask) -> bool:
        """Check all dependencies are completed before executing."""
        completed_ids = {t.id for t in self.get_completed_subtasks()}
        return all(dep in completed_ids for dep in task.dependencies)
