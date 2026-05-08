"""
Decomposition Agent
Breaks ambiguous queries into typed sub-tasks with explicit dependency graphs.
Dependent sub-tasks must not execute until their dependencies resolve.
"""
import json
import re
from app.agents.base import BaseAgent
from app.core.context import SharedContext, SubTask
from app.database import EventType


DECOMPOSITION_SYSTEM = """You are a Decomposition Agent. Your role is to break down complex queries into structured sub-tasks with explicit dependencies.

For each query, output a JSON object with this exact schema:
{
  "sub_tasks": [
    {
      "id": "task_1",
      "task_type": "research|compute|synthesize|validate",
      "description": "Clear description of what this sub-task does",
      "dependencies": [],
      "assigned_agent": "retrieval_agent|orchestrator|synthesis_agent"
    }
  ],
  "dependency_graph": {
    "task_1": [],
    "task_2": ["task_1"]
  },
  "reasoning": "Why you decomposed this way"
}

Rules:
- task_type must be one of: research, compute, synthesize, validate
- dependencies list IDs of tasks that must complete first
- Tasks with empty dependencies run first (in parallel if possible)
- Output ONLY valid JSON, no markdown fences
"""


class DecompositionAgent(BaseAgent):
    agent_id = "decomposition_agent"
    max_context_budget = 4000

    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        await self.declare_budget(context)

        messages = [
            {
                "role": "user",
                "content": f"Decompose this query into sub-tasks:\n\n{context.original_query}",
            }
        ]

        response_text, tokens = await self.call_llm(
            messages=messages,
            system=DECOMPOSITION_SYSTEM,
            context=context,
            max_tokens=1500,
            stream_callback=stream_callback,
        )

        # Parse response
        try:
            # Strip possible markdown fences
            clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            data = json.loads(clean)

            sub_tasks = []
            for t in data.get("sub_tasks", []):
                st = SubTask(
                    id=t.get("id", f"task_{len(sub_tasks)+1}"),
                    task_type=t.get("task_type", "research"),
                    description=t.get("description", ""),
                    dependencies=t.get("dependencies", []),
                    assigned_agent=t.get("assigned_agent"),
                )
                sub_tasks.append(st)

            context.sub_tasks = sub_tasks
            context.dependency_graph = data.get("dependency_graph", {})

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create a single task
            context.errors.append(f"Decomposition parse error: {e}. Using fallback single-task decomposition.")
            context.sub_tasks = [
                SubTask(
                    id="task_1",
                    task_type="research",
                    description=f"Research and answer: {context.original_query}",
                    dependencies=[],
                )
            ]
            context.dependency_graph = {"task_1": []}

        # Store agent output
        from app.core.context import AgentOutput
        context.agent_outputs["decomposition_agent"] = AgentOutput(
            agent_id="decomposition_agent",
            output_type="sub_tasks",
            content={
                "sub_tasks": [t.model_dump() for t in context.sub_tasks],
                "dependency_graph": context.dependency_graph,
                "reasoning": response_text[:500],
            },
            token_count=tokens,
        )

        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.AGENT_END,
            data={
                "num_subtasks": len(context.sub_tasks),
                "dependency_graph": context.dependency_graph,
            },
            output_content=response_text,
            token_count=tokens,
        )

        return context
