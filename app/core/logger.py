"""
Structured logging with consistent schema:
timestamp, agent_id, event_type, input_hash, output_hash, latency, token_count, policy_violations
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Optional, Any
from app.database import AsyncSessionLocal, ExecutionLog, EventType
from app.core.tokens import hash_content

logger = logging.getLogger("mega_ai")


class StructuredLogger:
    """Thread-safe structured logger that writes to DB and stdout."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def log(
        self,
        agent_id: str,
        event_type: EventType,
        data: dict,
        input_content: Any = None,
        output_content: Any = None,
        latency_ms: Optional[float] = None,
        token_count: int = 0,
        policy_violation: Optional[str] = None,
    ) -> ExecutionLog:
        async with self._lock:
            self._sequence += 1
            seq = self._sequence

        entry = ExecutionLog(
            job_id=self.job_id,
            sequence_num=seq,
            agent_id=agent_id,
            event_type=event_type,
            input_hash=hash_content(input_content) if input_content else None,
            output_hash=hash_content(output_content) if output_content else None,
            latency_ms=latency_ms,
            token_count=token_count,
            policy_violation=policy_violation,
            data=data,
        )

        # Stdout structured log
        log_line = {
            "ts": datetime.utcnow().isoformat(),
            "job_id": self.job_id,
            "seq": seq,
            "agent": agent_id,
            "event": event_type.value,
            "tokens": token_count,
            "latency_ms": latency_ms,
            "violation": policy_violation,
        }
        if policy_violation:
            logger.warning(json.dumps(log_line))
        else:
            logger.info(json.dumps(log_line))

        async with AsyncSessionLocal() as session:
            session.add(entry)
            await session.commit()
            await session.refresh(entry)

        return entry
