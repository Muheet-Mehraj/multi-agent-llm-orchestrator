"""
Base agent. All agents inherit from this.
Enforces budget constraints and structured logging.
"""
import time
import asyncio
from typing import Optional, AsyncIterator, Any
from abc import ABC, abstractmethod
import anthropic

from app.config import settings
from app.core.context import SharedContext
from app.core.tokens import count_tokens, count_tokens_for_messages, hash_content
from app.core.logger import StructuredLogger
from app.database import EventType

client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


class BaseAgent(ABC):
    agent_id: str = "base_agent"
    max_context_budget: int = 4000

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    async def declare_budget(self, context: SharedContext):
        """Agents must declare budget before execution."""
        context.register_budget(self.agent_id, self.max_context_budget)
        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.AGENT_START,
            data={"budget_declared": self.max_context_budget},
        )

    async def check_budget(self, context: SharedContext, tokens_needed: int, label: str = "") -> bool:
        """
        Check if budget allows adding more tokens.
        Returns False and logs a policy violation if over budget.
        """
        budget = context.get_budget(self.agent_id)
        if not budget:
            await self.declare_budget(context)
            budget = context.get_budget(self.agent_id)

        ok = budget.check_and_consume(tokens_needed, label)
        if not ok:
            violation = (
                f"Agent {self.agent_id} budget overflow: needed {tokens_needed} "
                f"tokens for '{label}' but only {budget.remaining()} remaining"
            )
            context.policy_violations.append(violation)
            await self.logger.log(
                agent_id=self.agent_id,
                event_type=EventType.BUDGET_VIOLATION,
                data={"violation": violation, "tokens_needed": tokens_needed, "remaining": budget.remaining()},
                policy_violation=violation,
            )
        return ok

    async def call_llm(
        self,
        messages: list[dict],
        system: str,
        context: SharedContext,
        max_tokens: int = 1024,
        stream_callback=None,
    ) -> tuple[str, int]:
        """
        Call Claude with budget enforcement and streaming support.
        Returns (response_text, tokens_used).
        """
        input_tokens = count_tokens_for_messages(messages) + count_tokens(system)

        # Check budget before calling
        within_budget = await self.check_budget(context, input_tokens, label="llm_input")
        if not within_budget:
            # Trigger compression if over budget
            await self._trigger_compression(context, messages)
            input_tokens = count_tokens_for_messages(messages) + count_tokens(system)

        start = time.time()
        full_response = ""
        output_tokens = 0

        if stream_callback:
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    await stream_callback(self.agent_id, text)
            usage = (await stream.get_final_message()).usage
            output_tokens = usage.output_tokens
        else:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            full_response = response.content[0].text
            output_tokens = response.usage.output_tokens

        latency_ms = (time.time() - start) * 1000
        total_tokens = input_tokens + output_tokens

        await self.check_budget(context, output_tokens, label="llm_output")

        await self.logger.log(
            agent_id=self.agent_id,
            event_type=EventType.TOKEN_STREAM,
            data={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": "claude-sonnet-4-20250514",
            },
            input_content=messages,
            output_content=full_response,
            latency_ms=latency_ms,
            token_count=total_tokens,
        )

        return full_response, total_tokens

    async def _trigger_compression(self, context: SharedContext, messages: list[dict]):
        """Compress older conversational context to free up budget."""
        from app.core.tokens import compress_conversational, serialize_structured
        # Find non-structured messages and compress them
        for msg in messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                original = msg["content"]
                if count_tokens(original) > 200:
                    compressed = compress_conversational(original, 100)
                    msg["content"] = compressed
                    event = {
                        "original_tokens": count_tokens(original),
                        "compressed_tokens": count_tokens(compressed),
                        "agent": self.agent_id,
                    }
                    context.compression_events.append(event)
                    await self.logger.log(
                        agent_id=self.agent_id,
                        event_type=EventType.CONTEXT_COMPRESS,
                        data=event,
                    )

    @abstractmethod
    async def run(self, context: SharedContext, stream_callback=None) -> SharedContext:
        """Execute the agent's primary task."""
        ...
