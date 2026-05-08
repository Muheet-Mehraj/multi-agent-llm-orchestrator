-- Postgres initialization script
-- Creates indexes for fast log querying

-- Run after tables are created by SQLAlchemy init_db()
-- These are applied on first container start

-- Enable faster log queries by agent_id and event_type
DO $$
BEGIN
    -- Index for querying logs by job
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_exec_logs_job_seq') THEN
        CREATE INDEX idx_exec_logs_job_seq ON execution_logs(job_id, sequence_num);
    END IF;

    -- Index for policy violation queries
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_exec_logs_violation') THEN
        CREATE INDEX idx_exec_logs_violation ON execution_logs(policy_violation) WHERE policy_violation IS NOT NULL;
    END IF;

    -- Index for tool call queries by job
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_tool_calls_job') THEN
        CREATE INDEX idx_tool_calls_job ON tool_call_logs(job_id, agent_id);
    END IF;

    -- Index for eval results by run and pass/fail
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_eval_results_run_passed') THEN
        CREATE INDEX idx_eval_results_run_passed ON eval_results(run_id, passed);
    END IF;

    -- Index for prompt rewrite status queries
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_rewrites_status') THEN
        CREATE INDEX idx_rewrites_status ON prompt_rewrites(status, proposed_at);
    END IF;
EXCEPTION WHEN OTHERS THEN
    -- Tables may not exist yet on very first boot; SQLAlchemy will create them
    RAISE NOTICE 'Index creation skipped (tables not yet created): %', SQLERRM;
END $$;
