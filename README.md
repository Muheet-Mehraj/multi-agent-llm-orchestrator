# Multi-Agent LLM Orchestration System

A production-grade multi-agent system featuring a self-improving evaluation loop, dynamic tool orchestration, adversarial robustness testing, and full streaming observability.

---

## Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose
- An Anthropic API key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Muheet-Mehraj/multi-agent-llm-orchestrator.git

cd multi-agent-llm-orchestrator

# 2. Configure environment
cp .env.example .env

# Edit .env and set:
# ANTHROPIC_API_KEY=your_key_here

# 3. Start all services
docker compose up --build

# 4. Verify API health
curl http://localhost:8000/health
```


All 5 endpoints are now live. No manual DB migrations, no extra steps.

---

## API Endpoints

All endpoints are documented at `http://localhost:8000/docs` (Swagger UI).

### 1. `POST /query` — Submit query with SSE streaming

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the transformer attention mechanism work?", "stream": true}'
```

Returns Server-Sent Events. Each event contains:
- `agent` — which agent is currently writing
- `token` — the streamed token
- `event` — `job_start | token | tool_call | heartbeat | job_end`
- `context_budget_remaining` — tokens left for current agent

Non-streaming mode: `"stream": false` returns `job_id` for polling.

**Error codes:** `EMPTY_QUERY`

### 2. `GET /jobs/{job_id}/trace` — Full execution trace

```bash
curl http://localhost:8000/jobs/{job_id}/trace
```

Returns complete ordered sequence: routing decision → agent executions → tool calls → handoffs → final answer with provenance map.

**Error codes:** `JOB_NOT_FOUND`

### 3. `GET /eval/latest` — Latest eval run summary

```bash
curl http://localhost:8000/eval/latest
```

Returns results broken down by test category (baseline/ambiguous/adversarial) and all 6 scoring dimensions with numeric score + written justification per test case.

**Error codes:** `NO_EVAL_RUN`

### 4. `POST /eval/rewrites/{rewrite_id}/review` — Approve/reject prompt rewrite

```bash
curl -X POST http://localhost:8000/eval/rewrites/{id}/review \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reviewer_note": "Looks good, clearer citation instructions"}'
```

If approved: automatically re-runs eval on failed cases using new prompt, logs performance delta.

**Error codes:** `REWRITE_NOT_FOUND`, `REWRITE_NOT_PENDING`

### 5. `POST /eval/rerun` — Re-eval on failed cases

```bash
curl -X POST http://localhost:8000/eval/rerun
```

Runs only the test cases that failed in the latest eval run, using latest approved prompts.

**Error codes:** `NO_EVAL_RUN`

### Helper endpoints (not in required 5)

| Endpoint | Description |
|----------|-------------|
| `POST /eval/run` | Trigger full 15-case eval + meta-agent analysis |
| `GET /eval/rewrites` | List all proposed prompt rewrites |
| `GET /health` | Health check |

---

## Architecture

![Architecture](./docs/Architecture.png)

### Agent Descriptions & Decision Boundaries

#### Master Orchestrator (`orchestrator`)
**Decision boundary:** Routes queries dynamically by calling Claude to produce a structured routing plan (JSON). Routing is NOT hardcoded. The orchestrator decides which agents to invoke, in what order, and what context budget each gets. All inter-agent handoffs go through the orchestrator — agents never call each other.

#### Decomposition Agent (`decomposition_agent`)
**Decision boundary:** Receives the raw query. Outputs typed sub-tasks (`research|compute|synthesize|validate`) with an explicit dependency graph. Sub-tasks with dependencies are blocked from execution until their upstream tasks resolve. Activates when: query is complex, multi-part, or ambiguous.

#### Retrieval Agent (`retrieval_agent`)
**Decision boundary:** Performs multi-hop retrieval (minimum 2 hops). Hop 1 = broad web search; Hop 2 = targeted follow-up search using terms from Hop 1 results. Single-hop retrieval is rejected. Each statement in the answer is cited with exact `chunk_id`(s). Falls back to `data_lookup` if web search fails.

#### Critique Agent (`critique_agent`)
**Decision boundary:** Reviews every other agent's output. Assigns confidence scores **per claim**, not per output. Flags specific text spans it disagrees with. Calls the `self_reflect` tool to identify internal contradictions. Does NOT rate outputs holistically.

#### Synthesis Agent (`synthesis_agent`)
**Decision boundary:** Receives all prior agent outputs + critique findings. Resolves ALL flagged contradictions (does not surface them to the user). Produces final answer with a provenance map linking each sentence to its source agent and source chunk IDs.

#### Meta-Agent (`meta_agent`)
**Decision boundary:** Runs after each eval. Reads failure cases, computes per-dimension averages, identifies the worst-performing dimension, maps it to the responsible agent, and proposes a concrete prompt rewrite with unified diff. The rewrite is NEVER auto-applied — requires human approval.

### Tools

| Tool | Purpose | Failure Modes |
|------|---------|---------------|
| `web_search` | Keyword search against curated corpus | `timeout`, `empty`, `malformed` |
| `code_execute` | Python sandbox (subprocess) | `timeout`, `malformed`, `error` |
| `data_lookup` | Structured data lookup against in-memory tables | `empty`, `malformed` |
| `self_reflect` | Contradiction detection in session outputs | `empty`, `malformed` |

Each tool supports up to 2 retries, each logged separately. Fallback strategy depends on failure mode (in code, not in prompts):
- `empty` → broaden/shorten query
- `timeout` → simplify input  
- `malformed` → no retry, return failure

---

## Evaluation Pipeline

### Test Cases (15 total)

| Category | Count | Description |
|----------|-------|-------------|
| Baseline | 5 | Factual queries with verifiable answers |
| Ambiguous | 5 | Underspecified inputs (test decomposition quality) |
| Adversarial | 5 | Prompt injections, wrong premises, contradiction traps |

### Scoring Dimensions (all custom — no third-party eval framework)

| Dimension | Method | Range |
|-----------|--------|-------|
| `answer_correctness` | Keyword coverage + adversarial checks | 0.0–1.0 |
| `citation_accuracy` | Provenance map chunk_id validation | 0.0–1.0 |
| `contradiction_resolution` | Flagged spans vs. resolved contradictions | 0.0–1.0 |
| `tool_efficiency` | Call count vs. max allowed, expected tools | 0.0–1.0 |
| `budget_compliance` | Policy violation count → penalty | 0.0–1.0 |
| `critique_agreement` | Acceptance rate × avg confidence | 0.0–1.0 |

Every dimension produces a **numeric score** and a **written justification string**. Every eval run stores exact prompts sent, exact tool calls made, exact outputs received, and timestamps — making runs fully diff-able.

### Self-Improving Loop

```
eval run → meta_agent identifies worst dimension → proposes prompt rewrite
         → stored as PENDING → human approves via API
         → re-eval on FAILED CASES ONLY → delta logged
```

All steps are timestamped and queryable (`GET /eval/rewrites`).

---

## Observability

### Structured Log Schema

Every event written to `execution_logs` table:

```json
{
  "timestamp": "2026-05-08T12:00:00Z",
  "agent_id": "retrieval_agent",
  "event_type": "tool_call",
  "input_hash": "a3f9b1c2",
  "output_hash": "d7e2a4f1",
  "latency_ms": 142.3,
  "token_count": 156,
  "policy_violation": null
}
```

### Log Query Interface

Datasette runs on `http://localhost:8080` — query any table with SQL directly.

---

## Context Budget Management

Each agent declares its maximum token budget before execution. The `ContextBudget` object tracks consumption in real time. If a budget is exceeded:

1. A **policy violation** is logged (never silent truncation)
2. The event appears in `GET /jobs/{id}/trace`
3. A compression pass runs: **lossless** for structured data (tool outputs, scores, citations), **lossy** only for conversational filler

---

## Known Limitations (Honest Assessment)

### What works well
- The full agent pipeline, tool layer, and SSE streaming are production-quality
- Context budget enforcement with policy violations is strict
- Scoring logic is transparent and all custom
- The self-improving loop is fully auditable

### Where the system breaks

1. **Web search is a stub.** The corpus is ~8 documents. For real use, replace `web_search` with a real search API (Tavily, Serper, etc.). Multi-hop retrieval works correctly structurally, but relevance is limited by the tiny corpus.

2. **NL→SQL is toy-level.** The `data_lookup` tool does keyword-based filtering, not true NL→SQL translation. For production, route through the LLM to generate real SQL against a real schema.

3. **Meta-agent prompt rewrites are LLM-generated.** The quality of proposed rewrites depends on Claude's ability to reason about prompt failures — it will sometimes propose superficial changes.

4. **No distributed locking.** The worker and API server can process the same job concurrently in edge cases. Add a Redis-based distributed lock for production.

5. **Token counting uses char-based fallback.** When `tiktoken` can't download its vocabulary file (network restrictions), it falls back to `len(text) // 4`. This is an approximation, not exact.

6. **SSE stream reconnection.** If the client disconnects mid-stream, the job continues running but the client can't resume the stream. Implement event replay using `Last-Event-ID` for production.

### What the self-improving loop does NOT do
- Does not automatically apply prompt rewrites (by design — requires human approval)
- Does not run A/B testing across prompt versions in parallel
- Does not use gradient-based optimization (it's LLM-in-the-loop, not RL)
- Proposed rewrites are stored per run but not versioned with semantic diff history

### What I would build next
1. **Real RAG pipeline** — ChromaDB or pgvector with actual document ingestion and semantic search
2. **Distributed tracing** — OpenTelemetry spans per agent with a Jaeger dashboard
3. **Prompt versioning database** — Git-like semantic versioning of all agent prompts with rollback
4. **Parallel agent execution** — Run decomposition + retrieval in parallel where dependency graph allows
5. **Real NL→SQL** — LLM-generated SQL with schema introspection and query validation
6. **Stream reconnection** — SSE event replay via Redis pub/sub with `Last-Event-ID`
7. **Rate limiting & auth** — API key middleware, per-client job quotas

---

## AI Collaboration Disclosure

This project was developed with AI-assisted tooling support.

**AI-assisted components:**
- Initial scaffolding of the multi-agent class hierarchy
- Structured JSON schemas for inter-agent communication
- FastAPI endpoint boilerplate
- Test case generation for the evaluation harness

**Human-designed components:**
- System architecture decisions (shared context pattern, no direct agent-to-agent calls)
- Failure contract specifications for each tool
- Scoring dimension logic and justification format
- Budget enforcement policy (violations logged, not silent truncation)
- Self-improving loop approval flow design

All AI-generated code was reviewed, debugged, and validated with the 49-test suite before commit.

---

## Running Tests

### Inside Docker

```bash
docker compose exec api pytest tests/ -v
```

### Locally

```bash
ANTHROPIC_API_KEY=test \
DATABASE_URL=postgresql+asyncpg://test:test@localhost:5432/test \
pytest tests/ -v
```

Current test suite status:

```text
49 passed
```

Coverage includes:
- orchestration utilities
- tool failure contracts and retries
- context budget enforcement
- dependency graph validation
- evaluation scoring logic
- adversarial handling
- SSE formatting
- prompt versioning
- provenance validation
- token counting and compression

---

## Project Structure

```
multi-agent-llm-orchestrator/
├── app/
│   ├── agents/
│   │   ├── base.py              # BaseAgent with budget enforcement + LLM call
│   │   ├── orchestrator.py      # Master orchestrator with dynamic routing
│   │   ├── decomposition.py     # Sub-task + dependency graph agent
│   │   ├── retrieval.py         # Multi-hop RAG agent
│   │   ├── critique.py          # Per-claim critique agent
│   │   └── synthesis.py         # Final answer + provenance map agent
│   ├── core/
│   │   ├── context.py           # SharedContext schema (Pydantic)
│   │   ├── tokens.py            # Token counting + compression
│   │   └── logger.py            # Structured logger → PostgreSQL
│   ├── tools/
│   │   └── tools.py             # 4 tools + retry logic + failure contracts
│   ├── evaluation/
│   │   ├── harness.py           # 15 test cases + 6 scoring dimensions
│   │   └── meta_agent.py        # Self-improving prompt loop
│   ├── worker/
│   │   └── main.py              # Redis-queue background worker
│   ├── database.py              # SQLAlchemy models
│   ├── config.py                # Pydantic settings
│   └── main.py                  # FastAPI app (5 endpoints)
├── tests/
│   ├── test_suite.py            # Core unit + integration tests
│   └── test_extended.py         # Extended architecture + eval tests
├── docs/
│   └── architecture.png         #  Architecture diagram
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.logui
├── requirements.txt
├── .env.example
└── README.md
```
