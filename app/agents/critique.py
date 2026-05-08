"""
Critique Agent
Reviews output of every other agent.
Assigns structured confidence score PER CLAIM (not overall output).
Flags specific spans of text it disagrees with.
Sets agent_accepted on self_reflect tool output.
Emits tool-in-flight + budget-remaining to SSE stream.
"""
import json
import re
from app.agents.base import BaseAgent
from app.core.context import SharedContext, Claim, AgentOutput
from app.tools.tools import call_tool_with_retry
from app.database import EventType


CRITIQUE_SYSTEM = """You are a Critique Agent. Your job is to critically review other agents' outputs.

You MUST:
1. Assign confidence scores PER CLAIM (not overall)
2. Flag SPECIFIC SPANS of text you disagree with, not whole outputs
3. Be precise about what exactly you disagree with and why

Output format (JSON):
{
  "claims_reviewed": [
    {
      "text": "The exact text span being reviewed",
      "source_agent": "which_agent",
      "confidence": 0.0-1.0,
      "flagged": true/false,
      "flag_reason": "Specific reason if flagged, null if not flagged",
      "verdict": "accept|reject|uncertain"
    }
  ],
  "flagged_spans": [
    {
      "span": "exact text excerpt",
      "source_agent": "agent_id",
      "issue": "What is wrong with this span",
      "severity": "high|medium|low"
    }
  ],
  "overall_assessment": "Summary of critique findings",
  "contradiction_check": "Results from self-reflection"
}

Be rigorous. Flag uncertain items explicitly. Output ONLY valid JSON.
"""


async def _emit(stream_callback, agent_id: str, msg: str):
    if stream_callback:
        await stream_callback(agent_id, msg)


class CritiqueAgent(BaseAgent):
    agent_id = "critique_agent"
    max_context_budget = 4000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        def _remaining():
            b = context.get_budget(self.agent_id)
            return b.remaining() if b else "?"

        # Self-reflection: check for contradictions in previous outputs
        context_snapshot = {
            "agent_outputs": {
                k: {"content": str(v.content)[:500]}
                for k, v in context.agent_outputs.items()
            }
        }

        await _emit(stream_callback, self.agent_id,
                    f"[TOOL_CALL:self_reflect focus=contradictions budget_remaining={_remaining()}]\n")

        reflect_result, _ = await call_tool_with_retry(
            tool_name="self_reflect",
            tool_input={"context_snapshot": context_snapshot, "focus": "contradictions"},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=1,
        )

        # Agent decides whether self-reflect output is sufficient
        reflection_text = ""
        if reflect_result.success and reflect_result.data:
            reflect_result.agent_accepted = True
            reflection_text = reflect_result.data.get("reflection_summary", "")
        else:
            reflect_result.agent_accepted = False  # Rejected: empty or failed

        await _emit(stream_callback, self.agent_id,
                    f"[TOOL_RESULT:self_reflect success={reflect_result.success} "
                    f"accepted={reflect_result.agent_accepted} "
                    f"contradictions={reflect_result.data.get('total_contradictions', 0) if reflect_result.data else 'n/a'} "
                    f"budget_remaining={_remaining()}]\n")

        # Build critique input
        outputs_text = "\n\n---\n\n".join(
            f"[{aid}]:\n{str(out.content)[:800]}"
            for aid, out in context.agent_outputs.items()
        ) or "No agent outputs available yet."

        messages = [{
            "role": "user",
            "content": (
                f"Original Query: {context.original_query}\n\n"
                f"Agent Outputs to Critique:\n{outputs_text}\n\n"
                f"Self-Reflection Results:\n{reflection_text or 'N/A'}\n\n"
                "Critique each claim individually with per-claim confidence scores. "
                "Flag specific text spans you disagree with."
            ),
        }]

        await self.check_budget(context, len(messages[0]["content"]) // 4, label="critique_input")

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=CRITIQUE_SYSTEM,
            context=context,
            max_tokens=1500,
            stream_callback=stream_callback,
        )

        try:
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            critique_data = json.loads(clean)
        except json.JSONDecodeError:
            critique_data = {
                "claims_reviewed": [],
                "flagged_spans": [],
                "overall_assessment": response_text[:500],
                "contradiction_check": reflection_text,
            }

        claims = [
            Claim(
                text=c.get("text", ""),
                confidence=float(c.get("confidence", 0.5)),
                source_agent=c.get("source_agent", "unknown"),
                flagged=c.get("flagged", False),
                flag_reason=c.get("flag_reason"),
            )
            for c in critique_data.get("claims_reviewed", [])
        ]

        context.claims = claims
        context.flagged_spans = critique_data.get("flagged_spans", [])
        context.critique_summary = critique_data.get("overall_assessment", "")

        context.agent_outputs["critique_agent"] = AgentOutput(
            agent_id="critique_agent",
            output_type="critique",
            content=critique_data,
            token_count=tokens,
        )

        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.AGENT_END,
            data={
                "claims_reviewed": len(claims),
                "flagged_count": sum(1 for c in claims if c.flagged),
                "flagged_spans": len(context.flagged_spans),
                "self_reflect_accepted": reflect_result.agent_accepted,
                "budget_remaining": _remaining(),
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
