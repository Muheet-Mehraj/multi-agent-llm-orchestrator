"""
Background worker that processes agent jobs asynchronously.
Polls Redis for queued jobs and executes them.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime

import redis.asyncio as aioredis

from app.config import settings
from app.database import init_db, AsyncSessionLocal, Job, JobStatus

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("mega_ai.worker")

QUEUE_KEY = "mega_ai:job_queue"


async def process_job(job_id: str, query: str):
    """Process a single job."""
    from app.agents.orchestrator import run_job
    logger.info(f"Worker processing job {job_id}")
    try:
        await run_job(job_id, query)
        logger.info(f"Job {job_id} completed successfully")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await session.commit()


async def main():
    """Worker main loop."""
    await init_db()
    logger.info("Worker started, waiting for jobs...")

    redis_client = aioredis.from_url(settings.redis_url)

    while True:
        try:
            # Blocking pop from Redis queue
            result = await redis_client.blpop(QUEUE_KEY, timeout=5)
            if result:
                _, payload_bytes = result
                payload = json.loads(payload_bytes)
                job_id = payload.get("job_id")
                query = payload.get("query")
                if job_id and query:
                    asyncio.create_task(process_job(job_id, query))
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
