"""
Four tools with defined failure contracts.
Each tool returns structured error info on: timeout, empty results, malformed input.
Fallback logic is in CODE, not in prompts.
"""
import asyncio
import time
import subprocess
import json
import re
import tempfile
import os
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

from app.config import settings


@dataclass
class ToolResult:
    success: bool
    data: Any
    failure_mode: Optional[str] = None  # "timeout" | "empty" | "malformed" | "error"
    error_message: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# TOOL 1: Web Search Stub
# ============================================================

WEB_SEARCH_CORPUS = [
    {
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
        "title": "Large language model - Wikipedia",
        "snippet": "A large language model (LLM) is a type of machine learning model designed to understand and generate human language. LLMs are trained on vast amounts of text data using self-supervised learning.",
        "relevance_score": 0.95,
        "source_type": "encyclopedia",
    },
    {
        "url": "https://arxiv.org/abs/2005.14165",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "snippet": "We demonstrate that scaling language models greatly improves task-agnostic, few-shot performance. GPT-3 achieves strong performance on many NLP tasks.",
        "relevance_score": 0.91,
        "source_type": "academic",
    },
    {
        "url": "https://www.anthropic.com/research/constitutional-ai",
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "snippet": "Constitutional AI is a method for training AI systems to be helpful, harmless, and honest using a set of principles and self-critique.",
        "relevance_score": 0.88,
        "source_type": "research",
    },
    {
        "url": "https://openai.com/research/gpt-4",
        "title": "GPT-4 Technical Report",
        "snippet": "GPT-4 is a large multimodal model that accepts image and text inputs, emitting text outputs. It exhibits human-level performance on various professional benchmarks.",
        "relevance_score": 0.87,
        "source_type": "research",
    },
    {
        "url": "https://huggingface.co/docs/transformers",
        "title": "Transformers Documentation - HuggingFace",
        "snippet": "The Transformer architecture uses self-attention mechanisms to process sequences. BERT uses bidirectional training while GPT uses unidirectional (autoregressive) training.",
        "relevance_score": 0.84,
        "source_type": "documentation",
    },
    {
        "url": "https://arxiv.org/abs/1706.03762",
        "title": "Attention Is All You Need",
        "snippet": "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",
        "relevance_score": 0.92,
        "source_type": "academic",
    },
    {
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "title": "Python programming language - Wikipedia",
        "snippet": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. Python is dynamically typed and garbage-collected.",
        "relevance_score": 0.90,
        "source_type": "encyclopedia",
    },
    {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "title": "asyncio — Asynchronous I/O — Python 3 docs",
        "snippet": "asyncio is used as a foundation for multiple Python asynchronous frameworks that provide high-performance network and web-servers, database connection libraries, distributed task queues, etc.",
        "relevance_score": 0.86,
        "source_type": "documentation",
    },
]


async def web_search(query: str, max_results: int = 3) -> ToolResult:
    """
    Web search stub returning structured results with URLs and relevance scores.
    Failure contract:
    - timeout: returns failure_mode="timeout" after TOOL_TIMEOUT_SECONDS
    - empty: returns failure_mode="empty" when no results match
    - malformed: returns failure_mode="malformed" for invalid input types
    """
    start = time.time()

    # Malformed input check
    if not isinstance(query, str) or not query.strip():
        return ToolResult(
            success=False,
            data=None,
            failure_mode="malformed",
            error_message="Query must be a non-empty string",
            latency_ms=(time.time() - start) * 1000,
        )

    # Simulate timeout for very long queries
    if len(query) > 1000:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="timeout",
            error_message="Query too long, simulated timeout",
            latency_ms=settings.tool_timeout_seconds * 1000,
        )

    await asyncio.sleep(0.1)  # Simulate network latency

    # Simple keyword matching against corpus
    query_lower = query.lower()
    results = []
    for item in WEB_SEARCH_CORPUS:
        score = 0.0
        text = (item["title"] + " " + item["snippet"]).lower()
        words = query_lower.split()
        matched = sum(1 for w in words if w in text)
        if matched > 0:
            score = item["relevance_score"] * (matched / len(words))
            results.append({**item, "relevance_score": round(score, 3)})

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = results[:max_results]

    if not results:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="empty",
            error_message=f"No results found for query: '{query}'",
            latency_ms=(time.time() - start) * 1000,
        )

    return ToolResult(
        success=True,
        data={"query": query, "results": results, "total_found": len(results)},
        latency_ms=(time.time() - start) * 1000,
    )


# ============================================================
# TOOL 2: Code Execution Sandbox
# ============================================================

async def code_execute(code: str, timeout: int = 10) -> ToolResult:
    """
    Execute Python code in a sandbox subprocess.
    Returns stdout, stderr, exit_code.
    Failure contract:
    - timeout: subprocess killed after `timeout` seconds
    - malformed: non-string or empty code
    - error: execution error (exit_code != 0 is still success=True with error info)
    """
    start = time.time()

    if not isinstance(code, str) or not code.strip():
        return ToolResult(
            success=False,
            data=None,
            failure_mode="malformed",
            error_message="Code must be a non-empty string",
            latency_ms=(time.time() - start) * 1000,
        )

    # Security: block dangerous imports
    dangerous = ["os.system", "subprocess", "shutil.rmtree", "__import__('os')", "eval(", "exec("]
    for d in dangerous:
        if d in code:
            return ToolResult(
                success=False,
                data=None,
                failure_mode="malformed",
                error_message=f"Dangerous pattern detected: {d}",
                latency_ms=(time.time() - start) * 1000,
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name

    try:
        loop = asyncio.get_event_loop()
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "python3", fname,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        exit_code = proc.returncode

        return ToolResult(
            success=True,
            data={
                "stdout": stdout.decode()[:2000],
                "stderr": stderr.decode()[:500],
                "exit_code": exit_code,
            },
            latency_ms=(time.time() - start) * 1000,
        )
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="timeout",
            error_message=f"Code execution timed out after {timeout}s",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="error",
            error_message=str(e),
            latency_ms=(time.time() - start) * 1000,
        )
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


# ============================================================
# TOOL 3: Structured Data Lookup (NL→SQL)
# ============================================================

# Simulated in-memory database for demo
KNOWLEDGE_BASE = {
    "llm_benchmarks": [
        {"model": "GPT-4", "mmlu_score": 86.4, "humaneval": 67.0, "params_b": 1000, "year": 2023},
        {"model": "Claude-3-Opus", "mmlu_score": 86.8, "humaneval": 84.9, "params_b": 200, "year": 2024},
        {"model": "Claude-3-Sonnet", "mmlu_score": 79.0, "humaneval": 73.0, "params_b": 70, "year": 2024},
        {"model": "Llama-3-70B", "mmlu_score": 82.0, "humaneval": 81.7, "params_b": 70, "year": 2024},
        {"model": "Gemini-1.5-Pro", "mmlu_score": 81.9, "humaneval": 71.9, "params_b": 340, "year": 2024},
        {"model": "Mistral-7B", "mmlu_score": 64.2, "humaneval": 30.5, "params_b": 7, "year": 2023},
    ],
    "agent_frameworks": [
        {"name": "LangChain", "language": "Python", "stars_k": 88, "multi_agent": True},
        {"name": "AutoGen", "language": "Python", "stars_k": 30, "multi_agent": True},
        {"name": "CrewAI", "language": "Python", "stars_k": 20, "multi_agent": True},
        {"name": "LlamaIndex", "language": "Python", "stars_k": 35, "multi_agent": False},
        {"name": "Semantic Kernel", "language": "C#", "stars_k": 21, "multi_agent": True},
    ],
}


def _simple_nl_to_filter(nl_query: str, table: str) -> list[dict]:
    """Extremely simplified NL-to-filter for demo. In production, use LLM."""
    data = KNOWLEDGE_BASE.get(table, [])
    q = nl_query.lower()

    if "best" in q or "top" in q or "highest" in q:
        if "mmlu" in q:
            return sorted(data, key=lambda x: x.get("mmlu_score", 0), reverse=True)[:3]
        if "humaneval" in q or "code" in q:
            return sorted(data, key=lambda x: x.get("humaneval", 0), reverse=True)[:3]
        return data[:3]

    if "small" in q or "7b" in q or "tiny" in q:
        return [x for x in data if x.get("params_b", 999) <= 10]

    if "multi" in q and "agent" in q:
        return [x for x in data if x.get("multi_agent") is True]

    return data[:5]


async def data_lookup(nl_query: str, table: Optional[str] = None) -> ToolResult:
    """
    Structured data lookup via NL query.
    Failure contract:
    - malformed: empty or non-string query
    - empty: table not found or no results
    - timeout: simulated for very complex queries
    """
    start = time.time()

    if not isinstance(nl_query, str) or not nl_query.strip():
        return ToolResult(
            success=False,
            data=None,
            failure_mode="malformed",
            error_message="Query must be a non-empty string",
            latency_ms=(time.time() - start) * 1000,
        )

    # Auto-detect table from query
    if not table:
        q = nl_query.lower()
        if "benchmark" in q or "mmlu" in q or "humaneval" in q or "model" in q:
            table = "llm_benchmarks"
        elif "framework" in q or "langchain" in q or "agent" in q:
            table = "agent_frameworks"
        else:
            table = "llm_benchmarks"  # default

    if table not in KNOWLEDGE_BASE:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="empty",
            error_message=f"Table '{table}' not found. Available: {list(KNOWLEDGE_BASE.keys())}",
            latency_ms=(time.time() - start) * 1000,
        )

    await asyncio.sleep(0.05)
    results = _simple_nl_to_filter(nl_query, table)

    if not results:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="empty",
            error_message=f"No records matched query in table '{table}'",
            latency_ms=(time.time() - start) * 1000,
        )

    generated_sql = f"SELECT * FROM {table} WHERE /* nl_query: {nl_query[:50]} */ LIMIT 5"
    return ToolResult(
        success=True,
        data={
            "table": table,
            "nl_query": nl_query,
            "generated_sql": generated_sql,
            "results": results,
            "row_count": len(results),
        },
        latency_ms=(time.time() - start) * 1000,
    )


# ============================================================
# TOOL 4: Self-Reflection
# ============================================================

async def self_reflect(context_snapshot: dict, focus: str = "contradictions") -> ToolResult:
    """
    Agent calls this to re-read its own previous outputs and identify contradictions.
    Failure contract:
    - malformed: missing context snapshot
    - empty: no previous outputs to reflect on
    """
    start = time.time()

    if not isinstance(context_snapshot, dict):
        return ToolResult(
            success=False,
            data=None,
            failure_mode="malformed",
            error_message="context_snapshot must be a dict",
            latency_ms=(time.time() - start) * 1000,
        )

    outputs = context_snapshot.get("agent_outputs", {})
    if not outputs:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="empty",
            error_message="No previous agent outputs to reflect on",
            latency_ms=(time.time() - start) * 1000,
        )

    # Find potential contradictions by looking for opposing keywords
    contradiction_pairs = [
        ("always", "never"), ("increase", "decrease"), ("true", "false"),
        ("yes", "no"), ("higher", "lower"), ("more", "less"),
    ]

    all_text = " ".join(
        str(o.get("content", "")) for o in outputs.values()
        if isinstance(o, dict)
    ).lower()

    found_contradictions = []
    for pos, neg in contradiction_pairs:
        if pos in all_text and neg in all_text:
            found_contradictions.append({
                "type": "potential_contradiction",
                "terms": [pos, neg],
                "note": f"Both '{pos}' and '{neg}' appear in outputs - verify consistency",
            })

    return ToolResult(
        success=True,
        data={
            "focus": focus,
            "outputs_reviewed": list(outputs.keys()),
            "contradictions_found": found_contradictions,
            "total_contradictions": len(found_contradictions),
            "reflection_summary": (
                f"Reviewed {len(outputs)} agent outputs. "
                f"Found {len(found_contradictions)} potential contradictions."
                if found_contradictions else
                f"Reviewed {len(outputs)} agent outputs. No obvious contradictions detected."
            ),
        },
        latency_ms=(time.time() - start) * 1000,
    )


# ============================================================
# Tool registry
# ============================================================

TOOL_REGISTRY = {
    "web_search": web_search,
    "code_execute": code_execute,
    "data_lookup": data_lookup,
    "self_reflect": self_reflect,
}


async def call_tool_with_retry(
    tool_name: str,
    tool_input: dict,
    job_id: str,
    agent_id: str,
    logger=None,
    max_retries: int = 2,
) -> tuple[ToolResult, int]:
    """
    Call a tool with up to max_retries retries.
    Each retry is logged separately.
    Returns (final_result, attempts_made).
    """
    from app.database import AsyncSessionLocal, ToolCallLog

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return ToolResult(
            success=False,
            data=None,
            failure_mode="malformed",
            error_message=f"Unknown tool: {tool_name}",
        ), 0

    for attempt in range(1, max_retries + 2):  # +2 for initial + retries
        result = await tool_fn(**tool_input)

        # Log this attempt
        async with AsyncSessionLocal() as session:
            log = ToolCallLog(
                job_id=job_id,
                agent_id=agent_id,
                tool_name=tool_name,
                attempt_num=attempt,
                input_data=tool_input,
                output_data=result.to_dict(),
                latency_ms=result.latency_ms,
                success=result.success,
                failure_mode=result.failure_mode,
            )
            session.add(log)
            await session.commit()

        if result.success:
            return result, attempt

        if attempt <= max_retries:
            # Modify input on retry based on failure mode
            if result.failure_mode == "empty":
                # Broaden the query
                if "query" in tool_input:
                    words = tool_input["query"].split()
                    tool_input = {**tool_input, "query": " ".join(words[:max(1, len(words)//2)])}
            elif result.failure_mode == "timeout":
                # Simplify input
                if "query" in tool_input:
                    tool_input = {**tool_input, "query": tool_input["query"][:100]}
            # malformed - don't retry
            else:
                return result, attempt

    return result, max_retries + 1
