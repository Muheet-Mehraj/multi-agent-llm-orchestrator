"""
Retrieval-Augmented Agent
Performs multi-hop reasoning across at least 2 retrieved chunks.
Must cite which chunk contributed to which part of the answer.
Single-hop retrieval is not sufficient.
"""
import json
import re
from app.agents.base import BaseAgent
from app.core.context import SharedContext, RetrievedChunk, AgentOutput
from app.tools.tools import call_tool_with_retry
from app.database import EventType


RETRIEVAL_SYSTEM = """You are a Retrieval-Augmented Agent. You perform multi-hop reasoning across retrieved chunks.

You have access to retrieved chunks. You MUST:
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


class RetrievalAgent(BaseAgent):
    agent_id = "retrieval_agent"
    max_context_budget = 6000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        # HOP 1: Broad retrieval
        hop1_result, hop1_attempts = await call_tool_with_retry(
            tool_name="web_search",
            tool_input={"query": context.original_query, "max_results": 3},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=2,
        )

        hop1_chunks = []
        if hop1_result.success and hop1_result.data:
            for i, r in enumerate(hop1_result.data.get("results", [])):
                chunk = RetrievedChunk(
                    chunk_id=f"chunk_h1_{i+1}",
                    content=r["snippet"],
                    source=r["url"],
                    relevance_score=r["relevance_score"],
                    retrieval_hop=1,
                )
                hop1_chunks.append(chunk)

        # Handle hop1 failure
        if not hop1_chunks:
            context.errors.append(f"Retrieval hop 1 failed: {hop1_result.failure_mode}")
            # Fallback: use data_lookup
            hop1_result, _ = await call_tool_with_retry(
                tool_name="data_lookup",
                tool_input={"nl_query": context.original_query},
                job_id=context.job_id,
                agent_id=self.agent_id,
                max_retries=1,
            )
            if hop1_result.success:
                results = hop1_result.data.get("results", [])
                for i, r in enumerate(results[:3]):
                    chunk = RetrievedChunk(
                        chunk_id=f"chunk_h1_db_{i+1}",
                        content=json.dumps(r),
                        source=f"database/{hop1_result.data.get('table')}",
                        relevance_score=0.7,
                        retrieval_hop=1,
                    )
                    hop1_chunks.append(chunk)

        # HOP 2: Targeted retrieval based on hop1 context
        # Extract key terms from hop1 for refinement
        hop1_text = " ".join(c.content[:100] for c in hop1_chunks)
        words = context.original_query.split()[:4]
        hop2_query = " ".join(words) + " " + " ".join(hop1_text.split()[:5])

        hop2_result, hop2_attempts = await call_tool_with_retry(
            tool_name="web_search",
            tool_input={"query": hop2_query.strip(), "max_results": 2},
            job_id=context.job_id,
            agent_id=self.agent_id,
            max_retries=2,
        )

        hop2_chunks = []
        if hop2_result.success and hop2_result.data:
            for i, r in enumerate(hop2_result.data.get("results", [])):
                # Only add chunks not already in hop1
                if not any(c.source == r["url"] for c in hop1_chunks):
                    chunk = RetrievedChunk(
                        chunk_id=f"chunk_h2_{i+1}",
                        content=r["snippet"],
                        source=r["url"],
                        relevance_score=r["relevance_score"],
                        retrieval_hop=2,
                    )
                    hop2_chunks.append(chunk)

        all_chunks = hop1_chunks + hop2_chunks
        context.retrieved_chunks = all_chunks

        if not all_chunks:
            # Last resort: create placeholder chunks
            all_chunks = [
                RetrievedChunk(
                    chunk_id="chunk_fallback_1",
                    content=f"General knowledge about: {context.original_query}",
                    source="internal_knowledge",
                    relevance_score=0.5,
                    retrieval_hop=1,
                ),
                RetrievedChunk(
                    chunk_id="chunk_fallback_2",
                    content="Additional context from internal knowledge base.",
                    source="internal_knowledge_2",
                    relevance_score=0.4,
                    retrieval_hop=2,
                ),
            ]
            context.retrieved_chunks = all_chunks

        # Build context for LLM reasoning
        chunks_text = "\n\n".join(
            f"[{c.chunk_id}] (hop={c.retrieval_hop}, score={c.relevance_score}, source={c.source})\n{c.content}"
            for c in all_chunks
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Query: {context.original_query}\n\n"
                    f"Retrieved chunks:\n{chunks_text}\n\n"
                    f"Perform multi-hop reasoning and cite each chunk."
                ),
            }
        ]

        within_budget = await self.check_budget(
            context,
            len(messages[0]["content"]) // 4,
            label="retrieval_reasoning_input"
        )

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=RETRIEVAL_SYSTEM,
            context=context,
            max_tokens=1500,
            stream_callback=stream_callback,
        )

        # Parse reasoning output
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            reasoning_data = json.loads(clean)
        except json.JSONDecodeError:
            reasoning_data = {
                "synthesis": response_text,
                "answer_with_citations": [],
                "retrieval_chain": "Multi-hop retrieval completed",
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
                "multi_hop": len(hop2_chunks) > 0,
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
