"""
Dynamic Router
Analyzes query and decides which agents to invoke, in what order, and with what budget.
Routing is done via LLM structured reasoning - NOT keyword matching.
"""
import json
import re
from app.core.context import SharedContext


async def dynamic_route(context: SharedContext, llm_client=None) -> list[dict]:
    """
    Dynamically route a query to agents using LLM reasoning.
    Returns ordered routing plan with budget allocations.
    Falls back to full pipeline if LLM unavailable.
    """
    if llm_client is None:
        # Fallback: always run full pipeline
        return _default_plan()

    system = """You are a routing agent. Analyze the query and decide which agents to invoke.
Available: decomposition_agent, retrieval_agent, critique_agent, synthesis_agent
Output JSON: {"routing_plan": [{"agent": "...", "reason": "...", "context_budget": 4000}],
              "routing_justification": "...", "query_complexity": "simple|moderate|complex",
              "adversarial_detected": false}"""

    try:
        response = await llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": f"Route this query: {context.original_query}"}],
        )
        text = response.content[0].text
        clean = re.sub(r"```json\s*|\s*```", "", text).strip()
        data = json.loads(clean)
        context.routing_justification = data.get("routing_justification", "")
        return data.get("routing_plan", _default_plan())
    except Exception:
        context.routing_justification = "Fallback routing: full pipeline"
        return _default_plan()


def _default_plan() -> list[dict]:
    return [
        {"agent": "decomposition_agent", "reason": "Break query into sub-tasks", "context_budget": 4000},
        {"agent": "retrieval_agent", "reason": "Multi-hop retrieval with citations", "context_budget": 6000},
        {"agent": "critique_agent", "reason": "Per-claim confidence scoring", "context_budget": 4000},
        {"agent": "synthesis_agent", "reason": "Resolve contradictions, build provenance map", "context_budget": 6000},
    ]
