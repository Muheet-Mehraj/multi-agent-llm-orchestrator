"""
Synthesis Agent
Merges outputs from all sub-agents.
Resolves contradictions flagged by critique agent.
Produces final answer with provenance map linking each sentence to source agent and chunk.
"""
import json
import re
from app.agents.base import BaseAgent
from app.core.context import SharedContext, ProvenanceEntry, AgentOutput
from app.database import EventType


SYNTHESIS_SYSTEM = """You are a Synthesis Agent. You merge all agent outputs into a final, coherent answer.

You MUST:
1. Resolve ALL contradictions flagged by the critique agent - do not surface them to the user
2. Create a provenance map linking EACH sentence to its source agent and source chunks
3. Produce a clean, definitive final answer
4. Note where you resolved contradictions and how

Output format (JSON):
{
  "final_answer": "The complete, clean answer to the user's query",
  "provenance_map": [
    {
      "sentence": "Exact sentence from final_answer",
      "source_agent": "which_agent",
      "source_chunk_ids": ["chunk_h1_1", "chunk_h2_2"],
      "confidence": 0.9
    }
  ],
  "contradictions_resolved": [
    {
      "original_contradiction": "What the contradiction was",
      "resolution": "How you resolved it",
      "resolution_strategy": "weighted_average|higher_confidence|additional_evidence|rejected_claim"
    }
  ],
  "synthesis_notes": "Any important notes about the synthesis process"
}

Output ONLY valid JSON.
"""


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis_agent"
    max_context_budget = 6000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        # Compile all inputs for synthesis
        retrieval_output = context.agent_outputs.get("retrieval_agent", {})
        retrieval_content = str(getattr(retrieval_output, "content", ""))[:600]

        decomp_output = context.agent_outputs.get("decomposition_agent", {})
        decomp_content = str(getattr(decomp_output, "content", ""))[:400]

        # Format critique findings
        flagged = "\n".join(
            f"- [{s.get('severity','?')}] {s.get('span','')}: {s.get('issue','')}"
            for s in context.flagged_spans
        ) or "No spans flagged."

        low_confidence = "\n".join(
            f"- ({c.confidence:.2f}) {c.text[:100]}: {c.flag_reason}"
            for c in context.claims
            if c.flagged
        ) or "No claims flagged."

        contradictions = context.agent_outputs.get("critique_agent")
        critique_assessment = ""
        if contradictions:
            data = getattr(contradictions, "content", {})
            if isinstance(data, dict):
                critique_assessment = data.get("overall_assessment", "")[:400]

        # Retrieved chunks summary
        chunks_summary = "\n".join(
            f"[{c.chunk_id}] {c.content[:150]}"
            for c in context.retrieved_chunks[:5]
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Original Query: {context.original_query}\n\n"
                    f"Decomposition:\n{decomp_content}\n\n"
                    f"Retrieval (multi-hop):\n{retrieval_content}\n\n"
                    f"Retrieved Chunks:\n{chunks_summary}\n\n"
                    f"Critique - Flagged Spans:\n{flagged}\n\n"
                    f"Critique - Low Confidence Claims:\n{low_confidence}\n\n"
                    f"Critique Summary:\n{critique_assessment}\n\n"
                    f"Synthesize a final answer resolving all contradictions."
                ),
            }
        ]

        within_budget = await self.check_budget(
            context,
            len(messages[0]["content"]) // 4,
            label="synthesis_input"
        )

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=SYNTHESIS_SYSTEM,
            context=context,
            max_tokens=2000,
            stream_callback=stream_callback,
        )

        # Parse synthesis output
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            synthesis_data = json.loads(clean)
        except json.JSONDecodeError:
            synthesis_data = {
                "final_answer": response_text,
                "provenance_map": [],
                "contradictions_resolved": [],
                "synthesis_notes": "JSON parse failed; using raw output",
            }

        # Populate context
        context.final_answer = synthesis_data.get("final_answer", response_text)
        context.contradictions_resolved = synthesis_data.get("contradictions_resolved", [])

        provenance = []
        for entry in synthesis_data.get("provenance_map", []):
            prov = ProvenanceEntry(
                sentence=entry.get("sentence", ""),
                source_agent=entry.get("source_agent", "unknown"),
                source_chunk_ids=entry.get("source_chunk_ids", []),
            )
            provenance.append(prov)
        context.provenance_map = provenance

        context.agent_outputs["synthesis_agent"] = AgentOutput(
            agent_id="synthesis_agent",
            output_type="final_answer",
            content=synthesis_data,
            token_count=tokens,
        )

        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.AGENT_END,
            data={
                "provenance_entries": len(provenance),
                "contradictions_resolved": len(context.contradictions_resolved),
                "final_answer_length": len(context.final_answer or ""),
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
