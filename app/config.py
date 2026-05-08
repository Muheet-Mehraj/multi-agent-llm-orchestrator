from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    anthropic_api_key: str
    database_url: str = "postgresql+asyncpg://megaai:megaai_secret@db:5432/megaai"
    redis_url: str = "redis://redis:6379"
    log_level: str = "INFO"
    environment: str = "development"

    # Agent context budgets (in tokens)
    orchestrator_budget: int = 8000
    decomposition_budget: int = 4000
    retrieval_budget: int = 6000
    critique_budget: int = 4000
    synthesis_budget: int = 6000
    compression_budget: int = 3000

    # Tool settings
    tool_timeout_seconds: int = 30
    tool_max_retries: int = 2

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
