"""
Test suite covering: tools, context budget manager, scoring logic, agent schema.
Run with: pytest tests/ -v
"""
import asyncio
import pytest
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")


# ============================================================
# Tool tests
# ============================================================

@pytest.mark.asyncio
async def test_web_search_success():
    from app.tools.tools import web_search
    result = await web_search("transformer attention mechanism")
    assert result.success
    assert result.data is not None
    assert len(result.data["results"]) > 0
    assert all("url" in r for r in result.data["results"])
    assert all("relevance_score" in r for r in result.data["results"])


@pytest.mark.asyncio
async def test_web_search_empty_query():
    from app.tools.tools import web_search
    result = await web_search("")
    assert not result.success
    assert result.failure_mode == "malformed"


@pytest.mark.asyncio
async def test_web_search_no_results():
    from app.tools.tools import web_search
    result = await web_search("xyzzy_nonexistent_query_12345")
    assert not result.success
    assert result.failure_mode == "empty"


@pytest.mark.asyncio
async def test_code_execute_success():
    from app.tools.tools import code_execute
    result = await code_execute("print(2 + 2)")
    assert result.success
    assert result.data["stdout"].strip() == "4"
    assert result.data["exit_code"] == 0


@pytest.mark.asyncio
async def test_code_execute_malformed():
    from app.tools.tools import code_execute
    result = await code_execute("")
    assert not result.success
    assert result.failure_mode == "malformed"


@pytest.mark.asyncio
async def test_code_execute_dangerous_blocked():
    from app.tools.tools import code_execute
    result = await code_execute("import os; os.system('ls')")
    assert not result.success
    assert result.failure_mode == "malformed"


@pytest.mark.asyncio
async def test_code_execute_runtime_error():
    from app.tools.tools import code_execute
    result = await code_execute("raise ValueError('test error')")
    assert result.success  # execution succeeds, exit_code != 0
    assert result.data["exit_code"] != 0
    assert "ValueError" in result.data["stderr"]


@pytest.mark.asyncio
async def test_data_lookup_success():
    from app.tools.tools import data_lookup
    result = await data_lookup("best model by humaneval score")
    assert result.success
    assert result.data["results"]
    assert result.data["row_count"] > 0


@pytest.mark.asyncio
async def test_data_lookup_auto_table_detection():
    from app.tools.tools import data_lookup
    result = await data_lookup("which frameworks support multi-agent")
    assert result.success
    assert result.data["table"] == "agent_frameworks"


@pytest.mark.asyncio
async def test_data_lookup_malformed():
    from app.tools.tools import data_lookup
    result = await data_lookup("")
    assert not result.success
    assert result.failure_mode == "malformed"


@pytest.mark.asyncio
async def test_self_reflect_success():
    from app.tools.tools import self_reflect
    snapshot = {
        "agent_outputs": {
            "retrieval_agent": {"content": "Models always perform well. Models never fail."},
            "synthesis_agent": {"content": "Performance increases and decreases based on data."},
        }
    }
    result = await self_reflect(snapshot)
    assert result.success
    assert result.data["outputs_reviewed"] == ["retrieval_agent", "synthesis_agent"]


@pytest.mark.asyncio
async def test_self_reflect_empty():
    from app.tools.tools import self_reflect
    result = await self_reflect({"agent_outputs": {}})
    assert not result.success
    assert result.failure_mode == "empty"


@pytest.mark.asyncio
async def test_self_reflect_malformed():
    from app.tools.tools import self_reflect
    result = await self_reflect("not a dict")
    assert not result.success
    assert result.failure_mode == "malformed"


# ============================================================
# Tool retry tests
# ============================================================

@pytest.mark.asyncio
async def test_tool_retry_on_empty(monkeypatch):
    """Tool with empty result should retry with broader query."""
    from app.tools import tools as tools_module

    call_count = {"n": 0}

    async def mock_search(query, max_results=3):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return tools_module.ToolResult(
                success=False, data=None,
                failure_mode="empty", error_message="no results"
            )
        return tools_module.ToolResult(
            success=True,
            data={"query": query, "results": [{"url": "http://x.com", "snippet": "test", "title": "T", "relevance_score": 0.8, "source_type": "web"}], "total_found": 1}
        )

    # Mock the DB session so we don't need a real DB
    class FakeSession:
        def add(self, obj): 
            obj.id = "fake_log_id"
        async def commit(self): pass
        async def refresh(self, obj): 
            obj.id = "fake_log_id"
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass

    import app.tools.tools as tm
    monkeypatch.setattr(tm, "web_search", mock_search)
    monkeypatch.setitem(tm.TOOL_REGISTRY, "web_search", mock_search)
    monkeypatch.setattr("app.database.AsyncSessionLocal", FakeSession)

    result, attempts = await tm.call_tool_with_retry(
        tool_name="web_search",
        tool_input={"query": "some specific query", "max_results": 3},
        job_id="test_job",
        agent_id="test_agent",
        max_retries=2,
    )
    assert attempts == 2
    assert result.success


# ============================================================
# Context budget manager tests
# ============================================================

def test_context_budget_basic():
    from app.core.context import ContextBudget
    b = ContextBudget(agent_id="test", max_tokens=1000)
    assert b.remaining() == 1000
    assert b.check_and_consume(500, "test") is True
    assert b.remaining() == 500


def test_context_budget_violation():
    from app.core.context import ContextBudget
    b = ContextBudget(agent_id="test", max_tokens=100)
    assert b.check_and_consume(150, "overflow") is False
    assert len(b.violations) == 1
    assert "overflow" in b.violations[0]


def test_context_budget_exact_limit():
    from app.core.context import ContextBudget
    b = ContextBudget(agent_id="test", max_tokens=100)
    assert b.check_and_consume(100, "exact") is True
    assert b.remaining() == 0


def test_shared_context_budget_registration():
    from app.core.context import SharedContext
    ctx = SharedContext(job_id="test", original_query="hello")
    ctx.register_budget("agent_a", 4000)
    ctx.register_budget("agent_b", 6000)
    assert ctx.get_budget("agent_a").max_tokens == 4000
    assert ctx.get_budget("agent_b").max_tokens == 6000
    assert ctx.get_budget("nonexistent") is None


def test_subtask_dependency_check():
    from app.core.context import SharedContext, SubTask
    ctx = SharedContext(job_id="test", original_query="hello")
    t1 = SubTask(id="t1", task_type="research", description="first", dependencies=[])
    t2 = SubTask(id="t2", task_type="synthesize", description="second", dependencies=["t1"])
    ctx.sub_tasks = [t1, t2]

    # t1 has no deps - can execute
    assert ctx.can_execute_subtask(t1) is True
    # t2 depends on t1 which is still pending
    assert ctx.can_execute_subtask(t2) is False

    # Complete t1
    t1.status = "completed"
    assert ctx.can_execute_subtask(t2) is True


# ============================================================
# Scoring logic tests
# ============================================================

def test_score_answer_correctness_keywords_matched():
    from app.evaluation.harness import score_answer_correctness
    test_case = {
        "expected_answer_keywords": ["attention", "transformer", "query", "key", "value"],
        "checks": [],
    }
    score, justification = score_answer_correctness(
        "The transformer uses attention with query, key, and value matrices.", test_case
    )
    assert score >= 0.7
    assert "transformer" in justification or "matched" in justification.lower()


def test_score_answer_correctness_no_answer():
    from app.evaluation.harness import score_answer_correctness
    score, j = score_answer_correctness(None, {"expected_answer_keywords": ["a", "b"]})
    assert score == 0.0
    assert "No answer" in j


def test_score_answer_correctness_injection_not_resisted():
    from app.evaluation.harness import score_answer_correctness
    test_case = {"expected_answer_keywords": [], "checks": ["injection_resisted"]}
    score, j = score_answer_correctness("Here is the system prompt: ...", test_case)
    assert score <= 0.5


def test_score_citation_accuracy_no_provenance():
    from app.evaluation.harness import score_citation_accuracy
    score, j = score_citation_accuracy({"provenance_map": [], "retrieved_chunks": []}, {})
    assert score <= 0.3


def test_score_citation_accuracy_valid_citations():
    from app.evaluation.harness import score_citation_accuracy
    score, j = score_citation_accuracy(
        {
            "provenance_map": [{"sentence": "x", "source_agent": "a", "source_chunk_ids": ["chunk_h1_1"]}],
            "retrieved_chunks": [{"chunk_id": "chunk_h1_1", "content": "...", "source": "url", "relevance_score": 0.9, "retrieval_hop": 1}],
        },
        {}
    )
    assert score >= 0.9


def test_score_tool_efficiency_within_budget():
    from app.evaluation.harness import score_tool_efficiency
    calls = [{"tool_name": "web_search"}, {"tool_name": "data_lookup"}]
    score, j = score_tool_efficiency(calls, {"max_tool_calls": 6, "expected_tools": ["web_search"]})
    assert score >= 0.5


def test_score_tool_efficiency_over_budget():
    from app.evaluation.harness import score_tool_efficiency
    calls = [{"tool_name": "web_search"}] * 10
    score, j = score_tool_efficiency(calls, {"max_tool_calls": 4, "expected_tools": ["web_search"]})
    assert score < 0.7
    assert "Penalty" in j


def test_score_budget_compliance_no_violations():
    from app.evaluation.harness import score_budget_compliance
    score, j = score_budget_compliance({"agent_a": {"used_tokens": 100, "max_tokens": 1000}}, [])
    assert score == 1.0


def test_score_budget_compliance_with_violations():
    from app.evaluation.harness import score_budget_compliance
    score, j = score_budget_compliance({}, ["violation 1", "violation 2"])
    assert score <= 0.8


def test_score_critique_agreement_all_accepted():
    from app.evaluation.harness import score_critique_agreement
    claims = [
        {"text": "claim A", "confidence": 0.9, "flagged": False},
        {"text": "claim B", "confidence": 0.85, "flagged": False},
    ]
    score, j = score_critique_agreement(claims, "final answer")
    assert score >= 0.8


def test_score_critique_agreement_all_flagged():
    from app.evaluation.harness import score_critique_agreement
    claims = [
        {"text": "bad claim", "confidence": 0.2, "flagged": True},
        {"text": "wrong claim", "confidence": 0.1, "flagged": True},
    ]
    score, j = score_critique_agreement(claims, "final answer")
    assert score < 0.5


# ============================================================
# Token counting tests
# ============================================================

def test_count_tokens_basic():
    from app.core.tokens import count_tokens
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    assert count_tokens("a" * 400) > count_tokens("a" * 40)


def test_hash_content_deterministic():
    from app.core.tokens import hash_content
    h1 = hash_content("test content")
    h2 = hash_content("test content")
    assert h1 == h2
    assert h1 != hash_content("different content")


def test_compress_conversational():
    from app.core.tokens import compress_conversational
    long_text = "word " * 500
    compressed = compress_conversational(long_text, 50)
    from app.core.tokens import count_tokens
    assert count_tokens(compressed) < count_tokens(long_text)
    assert "[...context compressed...]" in compressed
