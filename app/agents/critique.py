"""
Critique Agent
Reviews output of every other agent.
Assigns structured confidence score PER CLAIM (not overall).
Flags specific spans of text it disagrees with.
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
3. Use the self_reflect tool to check for internal contradictions
4. Be precise about what exactly you disagree with and why

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
      "issue": "What's wrong with this span",
      "severity": "high|medium|low"
    }
  ],
  "overall_assessment": "Summary of critique findings",
  "contradiction_check": "Results from self-reflection"
}

Be rigorous. If you're not sure about something, flag it as uncertain.
Output ONLY valid JSON.
"""


class CritiqueAgent(BaseAgent):
    agent_id = "critique_agent"
    max_context_budget = 4000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        # Self-reflection: check for contradictions in previous outputs
        context_snapshot = {
            "agent_outputs": {
                k: {"content": str(v.content)[:500]}
                for k, v in context.agent_outputs.items()
            }
        }

        reflect_result, reflect_attempts = await call_tool_with_retry(
            tool_name="self_reflect",
            tool_input={"context_snapshot": context_snapshot, "focus": "contradictions"},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=1,
        )

        reflection_text = ""
        if reflect_result.success and reflect_result.data:
            reflection_text = reflect_result.data.get("reflection_summary", "")

        # Build critique input from all agent outputs
        outputs_text = []
        for agent_id, output in context.agent_outputs.items():
            content_str = str(output.content)[:800]
            outputs_text.append(f"[{agent_id}]:\n{content_str}")

        all_outputs = "\n\n---\n\n".join(outputs_text)
        if not all_outputs:
            all_outputs = "No agent outputs available yet."

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original Query: {context.original_query}\n\n"
                    f"Agent Outputs to Critique:\n{all_outputs}\n\n"
                    f"Self-Reflection Results:\n{reflection_text}\n\n"
                    f"Critique each claim individually with confidence scores."
                ),
            }
        ]

        within_budget = await self.check_budget(
            context,
            len(messages[0]["content"]) // 4,
            label="critique_input"
        )

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=CRITIQUE_SYSTEM,
            context=context,
            max_tokens=1500,
            stream_callback=stream_callback,
        )

        # Parse critique output
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

        # Populate context claims
        claims = []
        for c in critique_data.get("claims_reviewed", []):
            claim = Claim(
                text=c.get("text", ""),
                confidence=float(c.get("confidence", 0.5)),
                source_agent=c.get("source_agent", "unknown"),
                flagged=c.get("flagged", False),
                flag_reason=c.get("flag_reason"),
            )
            claims.append(claim)

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
                "reflection_used": bool(reflection_text),
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
