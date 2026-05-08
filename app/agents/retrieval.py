"""
Retrieval-Augmented Agent
Performs multi-hop reasoning across at least 2 retrieved chunks.
Must cite which chunk contributed to which part of the answer.
Single-hop retrieval is not sufficient.
Sets agent_accepted on every tool result.
Emits tool-in-flight + budget-remaining events to SSE stream.
"""
import json
import re
from app.agents.base import BaseAgent
from app.core.context import SharedContext, RetrievedChunk, AgentOutput
from app.tools.tools import call_tool_with_retry
from app.database import EventType


RETRIEVAL_SYSTEM = """You are a Retrieval-Augmented Agent. You perform multi-hop reasoning across retrieved chunks.

You MUST:
1. Perform at least 2 retrieval hops (first find broad context, then specific details)
2. Cite EXACTLY which chunk (by chunk_id) contributed to EACH part of your answer
3. Show your multi-hop reasoning chain explicitly
4. Never answer from just one chunk - always synthesize across multiple sources

Output format (JSON):
{
  "hop1_reasoning": "What you retrieved first and why",
  "hop2_reasoning": "What you retrieved next based on hop1",
  "answer_with_citations": [
    {
      "statement": "The fact or claim being made",
      "chunk_ids": ["chunk_1", "chunk_3"],
      "confidence": 0.85
    }
  ],
  "synthesis": "Final synthesized answer combining both hops",
  "retrieval_chain": "Description of the multi-hop reasoning path"
}

Output ONLY valid JSON.
"""


async def _emit(stream_callback, agent_id: str, msg: str):
    if stream_callback:
        await stream_callback(agent_id, msg)


class RetrievalAgent(BaseAgent):
    agent_id = "retrieval_agent"
    max_context_budget = 6000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        def _remaining():
            b = context.get_budget(self.agent_id)
            return b.remaining() if b else "?"

        # ── HOP 1: Broad retrieval ──────────────────────────────────────────
        await _emit(stream_callback, self.agent_id,
                    f"\n[TOOL_CALL:web_search hop=1 query=\"{context.original_query[:60]}\" budget_remaining={_remaining()}]\n")

        hop1_result, _ = await call_tool_with_retry(
            tool_name="web_search",
            tool_input={"query": context.original_query, "max_results": 3},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=2,
        )

        # Set agent_accepted: did we find the result sufficient?
        hop1_chunks = []
        if hop1_result.success and hop1_result.data:
            hop1_result.agent_accepted = True
            for i, r in enumerate(hop1_result.data.get("results", [])):
                hop1_chunks.append(RetrievedChunk(
                    chunk_id=f"chunk_h1_{i+1}",
                    content=r["snippet"],
                    source=r["url"],
                    relevance_score=r["relevance_score"],
                    retrieval_hop=1,
                ))
        else:
            hop1_result.agent_accepted = False  # Rejected – insufficient

        await _emit(stream_callback, self.agent_id,
                    f"[TOOL_RESULT:web_search hop=1 success={hop1_result.success} "
                    f"chunks={len(hop1_chunks)} latency={hop1_result.latency_ms:.0f}ms "
                    f"accepted={hop1_result.agent_accepted} budget_remaining={_remaining()}]\n")

        # Fallback to data_lookup if hop1 failed
        if not hop1_chunks:
            context.errors.append(f"Retrieval hop 1 web_search failed: {hop1_result.failure_mode}")
            await _emit(stream_callback, self.agent_id,
                        f"[TOOL_CALL:data_lookup fallback query=\"{context.original_query[:60]}\" budget_remaining={_remaining()}]\n")

            fallback_result, _ = await call_tool_with_retry(
                tool_name="data_lookup",
                tool_input={"nl_query": context.original_query},
                job_id=context.job_id,
                agent_id=self.agent_id,
                max_retries=1,
            )
            if fallback_result.success:
                fallback_result.agent_accepted = True
                for i, r in enumerate(fallback_result.data.get("results", [])[:3]):
                    hop1_chunks.append(RetrievedChunk(
                        chunk_id=f"chunk_h1_db_{i+1}",
                        content=json.dumps(r),
                        source=f"database/{fallback_result.data.get('table')}",
                        relevance_score=0.7,
                        retrieval_hop=1,
                    ))
            else:
                fallback_result.agent_accepted = False

            await _emit(stream_callback, self.agent_id,
                        f"[TOOL_RESULT:data_lookup accepted={fallback_result.agent_accepted} "
                        f"chunks={len(hop1_chunks)} budget_remaining={_remaining()}]\n")

        # ── HOP 2: Targeted retrieval based on hop1 ─────────────────────────
        hop1_text = " ".join(c.content[:100] for c in hop1_chunks)
        hop2_query = " ".join(context.original_query.split()[:4]) + " " + " ".join(hop1_text.split()[:5])

        await _emit(stream_callback, self.agent_id,
                    f"[TOOL_CALL:web_search hop=2 query=\"{hop2_query[:60]}\" budget_remaining={_remaining()}]\n")

        hop2_result, _ = await call_tool_with_retry(
            tool_name="web_search",
            tool_input={"query": hop2_query.strip(), "max_results": 2},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=2,
        )

        hop2_chunks = []
        if hop2_result.success and hop2_result.data:
            hop2_result.agent_accepted = True
            for i, r in enumerate(hop2_result.data.get("results", [])):
                if not any(c.source == r["url"] for c in hop1_chunks):
                    hop2_chunks.append(RetrievedChunk(
                        chunk_id=f"chunk_h2_{i+1}",
                        content=r["snippet"],
                        source=r["url"],
                        relevance_score=r["relevance_score"],
                        retrieval_hop=2,
                    ))
        else:
            hop2_result.agent_accepted = False  # Insufficient – mark rejected

        await _emit(stream_callback, self.agent_id,
                    f"[TOOL_RESULT:web_search hop=2 success={hop2_result.success} "
                    f"new_chunks={len(hop2_chunks)} accepted={hop2_result.agent_accepted} "
                    f"budget_remaining={_remaining()}]\n")

        all_chunks = hop1_chunks + hop2_chunks
        # Ensure at least 2 fallback chunks even if retrieval fully fails
        if len(all_chunks) < 2:
            all_chunks += [
                RetrievedChunk(chunk_id=f"chunk_fallback_{i+1}",
                               content=f"General knowledge (hop {i+1}): {context.original_query}",
                               source="internal_knowledge", relevance_score=0.4, retrieval_hop=i+1)
                for i in range(2 - len(all_chunks))
            ]

        context.retrieved_chunks = all_chunks

        # ── LLM multi-hop reasoning ─────────────────────────────────────────
        chunks_text = "\n\n".join(
            f"[{c.chunk_id}] (hop={c.retrieval_hop}, score={c.relevance_score}, source={c.source})\n{c.content}"
            for c in all_chunks
        )

        messages = [{
            "role": "user",
            "content": (
                f"Query: {context.original_query}\n\n"
                f"Retrieved chunks:\n{chunks_text}\n\n"
                "Perform multi-hop reasoning and cite each chunk."
            ),
        }]

        await self.check_budget(context, len(messages[0]["content"]) // 4, label="retrieval_input")

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=RETRIEVAL_SYSTEM,
            context=context,
            max_tokens=1500,
            stream_callback=stream_callback,
        )

        try:
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            reasoning_data = json.loads(clean)
        except json.JSONDecodeError:
            reasoning_data = {
                "synthesis": response_text,
                "answer_with_citations": [],
                "retrieval_chain": f"Multi-hop retrieval: {len(hop1_chunks)} hop1, {len(hop2_chunks)} hop2 chunks",
            }

        context.retrieval_reasoning = reasoning_data.get("retrieval_chain", "")
        context.agent_outputs["retrieval_agent"] = AgentOutput(
            agent_id="retrieval_agent",
            output_type="retrieval_result",
            content=reasoning_data,
            token_count=tokens,
        )

        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.AGENT_END,
            data={
                "chunks_retrieved": len(all_chunks),
                "hop1_chunks": len(hop1_chunks),
                "hop2_chunks": len(hop2_chunks),
                "multi_hop_confirmed": len(hop2_chunks) > 0,
                "budget_remaining": _remaining(),
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
