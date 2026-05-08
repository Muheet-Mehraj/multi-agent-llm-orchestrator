"""
Meta-Agent: Self-Improving Prompt Loop
Reads failure cases, identifies worst-performing prompt dimension,
proposes rewrite with structured diff and justification.
Stores proposed rewrite - does NOT auto-apply.
"""
import json
import re
import uuid
import difflib
from datetime import datetime
from typing import Optional

from app.agents.base import BaseAgent
from app.core.context import SharedContext
from app.core.logger import StructuredLogger
from app.database import (
    AsyncSessionLocal, EvalRun, EvalResult, PromptRewrite,
    PromptRewriteStatus, EventType
)
from sqlalchemy import select

# Current prompt versions (source of truth for what's active)
CURRENT_PROMPTS = {
    "decomposition_agent": """You are a Decomposition Agent. Your role is to break down complex queries into structured sub-tasks with explicit dependencies.

For each query, output a JSON object with this exact schema:
{
  "sub_tasks": [...],
  "dependency_graph": {...},
  "reasoning": "Why you decomposed this way"
}

Rules:
- task_type must be one of: research, compute, synthesize, validate
- dependencies list IDs of tasks that must complete first
- Tasks with empty dependencies run first (in parallel if possible)
- Output ONLY valid JSON, no markdown fences
""",
    "retrieval_agent": """You are a Retrieval-Augmented Agent. You perform multi-hop reasoning across retrieved chunks.

You MUST:
1. Perform at least 2 retrieval hops (first find broad context, then specific details)
2. Cite EXACTLY which chunk (by chunk_id) contributed to EACH part of your answer
3. Show your multi-hop reasoning chain explicitly
4. Never answer from just one chunk - always synthesize across multiple sources

Output format (JSON): {...}
""",
    "critique_agent": """You are a Critique Agent. Your job is to critically review other agents' outputs.

You MUST:
1. Assign confidence scores PER CLAIM (not overall)
2. Flag SPECIFIC SPANS of text you disagree with, not whole outputs
3. Use the self_reflect tool to check for internal contradictions
4. Be precise about what exactly you disagree with and why
""",
    "synthesis_agent": """You are a Synthesis Agent. You merge all agent outputs into a final, coherent answer.

You MUST:
1. Resolve ALL contradictions flagged by the critique agent
2. Create a provenance map linking EACH sentence to its source agent and source chunks
3. Produce a clean, definitive final answer
4. Note where you resolved contradictions and how
""",
    "orchestrator": """You are a Master Orchestrator Agent. You dynamically route queries to sub-agents.

Given a query, decide which agents to invoke, in what order, and why.
Output routing plan as structured JSON.
""",
}

META_AGENT_SYSTEM = """You are a Meta-Agent responsible for improving other agents' prompts.

You will receive:
1. A failing test case with its scores per dimension
2. The current prompt for the worst-performing agent
3. The actual output that agent produced

Your job:
1. Identify the ROOT CAUSE of the failure
2. Propose a CONCRETE rewrite of the agent's prompt that would fix it
3. Provide a structured diff explanation
4. Justify why this change would improve performance

Output (JSON):
{
  "root_cause": "Specific reason why the agent failed",
  "proposed_prompt": "The full rewritten prompt text",
  "diff_description": "Line-by-line description of what changed and why",
  "expected_improvement": "Which dimension(s) should improve and by how much (e.g. citation_accuracy +0.2)",
  "confidence": 0.0-1.0,
  "justification": "Detailed reasoning for the proposed changes"
}

Be specific. Vague changes don't help. Output ONLY valid JSON.
"""


class MetaAgent(BaseAgent):
    agent_id = "meta_agent"
    max_context_budget = 6000

    async def analyze_failures(self, eval_run_id: str) -> Optional[str]:
        """
        Analyze failures from an eval run.
        Find worst-performing dimension.
        Propose prompt rewrite.
        Returns prompt_rewrite_id if created, None otherwise.
        """
        async with AsyncSessionLocal() as session:
            results = (
                await session.execute(
                    select(EvalResult)
                    .where(EvalResult.run_id == eval_run_id)
                    .where(EvalResult.passed == False)
                )
            ).scalars().all()

        if not results:
            return None  # All passed!

        # Find worst dimension across failures
        dim_scores = {
            "answer_correctness": [],
            "citation_accuracy": [],
            "contradiction_resolution": [],
            "tool_efficiency": [],
            "budget_compliance": [],
            "critique_agreement": [],
        }

        worst_result = min(results, key=lambda r: r.overall_score or 0)

        for r in results:
            dim_scores["answer_correctness"].append(r.answer_correctness or 0)
            dim_scores["citation_accuracy"].append(r.citation_accuracy or 0)
            dim_scores["contradiction_resolution"].append(r.contradiction_resolution or 0)
            dim_scores["tool_efficiency"].append(r.tool_efficiency or 0)
            dim_scores["budget_compliance"].append(r.budget_compliance or 0)
            dim_scores["critique_agreement"].append(r.critique_agreement or 0)

        dim_avgs = {
            k: sum(v) / len(v) if v else 1.0
            for k, v in dim_scores.items()
        }

        worst_dimension = min(dim_avgs, key=dim_avgs.get)

        # Map dimension to agent
        dim_to_agent = {
            "answer_correctness": "synthesis_agent",
            "citation_accuracy": "retrieval_agent",
            "contradiction_resolution": "synthesis_agent",
            "tool_efficiency": "orchestrator",
            "budget_compliance": "orchestrator",
            "critique_agreement": "critique_agent",
        }

        target_agent = dim_to_agent.get(worst_dimension, "synthesis_agent")
        current_prompt = CURRENT_PROMPTS.get(target_agent, "")

        # Build meta-agent context
        failure_summary = f"""
Worst dimension: {worst_dimension} (avg score: {dim_avgs[worst_dimension]:.3f})
Target agent to improve: {target_agent}

Worst failing test case:
- Query: {worst_result.query}
- Category: {worst_result.test_category}
- Overall score: {worst_result.overall_score:.3f}
- {worst_dimension} score: {getattr(worst_result, worst_dimension, 0):.3f}
- Justification: {getattr(worst_result, worst_dimension + '_justification', 'N/A')}
- Actual answer: {(worst_result.actual_answer or 'None')[:300]}

All dimension averages:
{json.dumps(dim_avgs, indent=2)}
"""

        # Create minimal context for meta agent call
        from app.core.context import SharedContext as SC
        dummy_context = SC(job_id=str(uuid.uuid4()), original_query="meta_analysis")
        dummy_context.register_budget(self.agent_id, self.max_context_budget)

        logger = StructuredLogger(job_id=dummy_context.job_id)
        self.logger = logger

        messages = [
            {
                "role": "user",
                "content": (
                    f"Failure analysis:\n{failure_summary}\n\n"
                    f"Current prompt for {target_agent}:\n{current_prompt}\n\n"
                    f"Propose an improved prompt."
                ),
            }
        ]

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=META_AGENT_SYSTEM,
            context=dummy_context,
            max_tokens=1500,
        )

        # Parse response
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            meta_data = json.loads(clean)
        except json.JSONDecodeError:
            meta_data = {
                "proposed_prompt": response_text[:2000],
                "diff_description": "Failed to parse structured diff",
                "justification": "Meta-agent produced unstructured output",
                "root_cause": "Unknown",
                "confidence": 0.3,
            }

        proposed_prompt = meta_data.get("proposed_prompt", "")

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            current_prompt.splitlines(keepends=True),
            proposed_prompt.splitlines(keepends=True),
            fromfile=f"{target_agent}/current",
            tofile=f"{target_agent}/proposed",
            lineterm="",
        ))
        diff_text = "".join(diff_lines)[:3000]

        # Store rewrite (NOT applying it)
        async with AsyncSessionLocal() as session:
            rewrite = PromptRewrite(
                id=str(uuid.uuid4()),
                eval_run_id=eval_run_id,
                agent_id=target_agent,
                worst_dimension=worst_dimension,
                original_prompt=current_prompt,
                proposed_prompt=proposed_prompt,
                diff=diff_text or meta_data.get("diff_description", "No diff generated"),
                justification=meta_data.get("justification", ""),
                status=PromptRewriteStatus.PENDING,
            )
            session.add(rewrite)
            await session.commit()
            rewrite_id = rewrite.id

        return rewrite_id


async def approve_rewrite(rewrite_id: str, approved: bool, reviewer_note: Optional[str] = None) -> dict:
    """
    Approve or reject a pending prompt rewrite.
    If approved, triggers re-eval on previously failed cases.
    """
    async with AsyncSessionLocal() as session:
        rewrite = await session.get(PromptRewrite, rewrite_id)
        if not rewrite:
            return {"error": "Rewrite not found", "rewrite_id": rewrite_id}

        if rewrite.status != PromptRewriteStatus.PENDING:
            return {
                "error": f"Rewrite already {rewrite.status.value}",
                "rewrite_id": rewrite_id
            }

        rewrite.status = PromptRewriteStatus.APPROVED if approved else PromptRewriteStatus.REJECTED
        rewrite.reviewed_at = datetime.utcnow()
        rewrite.reviewer_note = reviewer_note
        await session.commit()
        run_id = rewrite.eval_run_id
        agent_id = rewrite.agent_id
        proposed = rewrite.proposed_prompt

    if not approved:
        return {"status": "rejected", "rewrite_id": rewrite_id}

    # Apply prompt: update in-memory dict AND write to prompts/active/ file
    CURRENT_PROMPTS[agent_id] = proposed
    try:
        from app.core.prompts import promote_prompt, save_proposed_prompt
        # First save to proposed/ for audit trail
        save_proposed_prompt(agent_id, proposed, rewrite_id)
        # Then promote to active/ (archives old version to history/)
        new_path = promote_prompt(agent_id, proposed)
    except Exception as e:
        # File I/O failure doesn't block the approval flow
        pass

    # Get failed test case IDs from original run
    async with AsyncSessionLocal() as session:
        failed_results = (
            await session.execute(
                select(EvalResult.test_case_id)
                .where(EvalResult.run_id == run_id)
                .where(EvalResult.passed == False)
            )
        ).scalars().all()

    failed_ids = list(failed_results)

    if not failed_ids:
        return {
            "status": "approved",
            "rewrite_id": rewrite_id,
            "reeval_run_id": None,
            "message": "No failed cases to re-evaluate",
        }

    # Trigger targeted re-eval
    from app.evaluation.harness import run_eval
    reeval_run_id = await run_eval(
        test_case_ids=failed_ids,
        is_reeval=True,
        reeval_of=run_id,
        prompt_versions={agent_id: proposed[:200]},
    )

    # Compute performance delta
    async with AsyncSessionLocal() as session:
        orig_results = (
            await session.execute(
                select(EvalResult)
                .where(EvalResult.run_id == run_id)
                .where(EvalResult.test_case_id.in_(failed_ids))
            )
        ).scalars().all()

        new_results = (
            await session.execute(
                select(EvalResult).where(EvalResult.run_id == reeval_run_id)
            )
        ).scalars().all()

    orig_avg = sum(r.overall_score or 0 for r in orig_results) / len(orig_results) if orig_results else 0
    new_avg = sum(r.overall_score or 0 for r in new_results) / len(new_results) if new_results else 0
    delta = new_avg - orig_avg

    async with AsyncSessionLocal() as session:
        rw = await session.get(PromptRewrite, rewrite_id)
        if rw:
            rw.reeval_run_id = reeval_run_id
            rw.performance_delta = {
                "original_avg": round(orig_avg, 3),
                "new_avg": round(new_avg, 3),
                "delta": round(delta, 3),
                "cases_retested": len(failed_ids),
            }
            await session.commit()

    return {
        "status": "approved",
        "rewrite_id": rewrite_id,
        "reeval_run_id": reeval_run_id,
        "performance_delta": {
            "original_avg": round(orig_avg, 3),
            "new_avg": round(new_avg, 3),
            "delta": round(delta, 3),
        },
    }
