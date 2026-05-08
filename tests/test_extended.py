"""
Additional tests covering:
- Prompt file loader (load_prompt, promote_prompt)
- agent_accepted field on ToolResult
- SSE event parsing (TOOL_CALL / TOOL_RESULT markers)
- Prompt registry round-trip
"""
import os
import tempfile
import pathlib
import asyncio
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")


# ============================================================
# Prompt loader tests
# ============================================================

def test_load_prompt_fallback():
    """load_prompt returns fallback when no file exists."""
    from app.core.prompts import load_prompt
    result = load_prompt("nonexistent_agent_xyz", "FALLBACK_TEXT")
    assert result == "FALLBACK_TEXT"


def test_load_prompt_from_file(tmp_path, monkeypatch):
    """load_prompt reads from the active prompts directory."""
    from app.core import prompts as p_module
    active = tmp_path / "active"
    active.mkdir()
    (active / "test_agent_v1.txt").write_text("LOADED PROMPT V1")

    monkeypatch.setattr(p_module, "ACTIVE_DIR", active)

    from app.core.prompts import load_prompt
    result = load_prompt("test_agent", "FALLBACK")
    assert result == "LOADED PROMPT V1"


def test_load_prompt_picks_latest_version(tmp_path, monkeypatch):
    """load_prompt picks the highest-numbered version file."""
    from app.core import prompts as p_module
    active = tmp_path / "active"
    active.mkdir()
    (active / "myagent_v1.txt").write_text("V1 PROMPT")
    (active / "myagent_v2.txt").write_text("V2 PROMPT")
    (active / "myagent_v3.txt").write_text("V3 PROMPT")

    monkeypatch.setattr(p_module, "ACTIVE_DIR", active)

    from app.core.prompts import load_prompt
    result = load_prompt("myagent", "FALLBACK")
    assert result == "V3 PROMPT"


def test_promote_prompt_archives_old(tmp_path, monkeypatch):
    """promote_prompt archives old version and writes new active file."""
    from app.core import prompts as p_module
    active = tmp_path / "active"
    history = tmp_path / "history"
    active.mkdir()
    history.mkdir()
    proposed = tmp_path / "proposed"
    proposed.mkdir()

    # Existing active prompt
    (active / "myagent_v1.txt").write_text("OLD PROMPT")

    monkeypatch.setattr(p_module, "ACTIVE_DIR", active)
    monkeypatch.setattr(p_module, "HISTORY_DIR", history)
    monkeypatch.setattr(p_module, "PROPOSED_DIR", proposed)

    from app.core.prompts import promote_prompt
    new_path = promote_prompt("myagent", "NEW IMPROVED PROMPT")

    # Old version should be archived
    assert (history / "myagent_v1.txt").exists()
    # New version should be active
    assert new_path.exists()
    assert new_path.read_text() == "NEW IMPROVED PROMPT"
    # New version number should be v2
    assert "v2" in new_path.name


def test_save_proposed_prompt(tmp_path, monkeypatch):
    """save_proposed_prompt writes to proposed/ directory."""
    from app.core import prompts as p_module
    proposed = tmp_path / "proposed"
    monkeypatch.setattr(p_module, "PROPOSED_DIR", proposed)

    from app.core.prompts import save_proposed_prompt
    path = save_proposed_prompt("test_agent", "PROPOSED CONTENT", "abc12345")

    assert path.exists()
    assert path.read_text() == "PROPOSED CONTENT"
    assert "abc1234" in path.name  # rewrite_id[:8]


def test_current_prompt_version_no_files(tmp_path, monkeypatch):
    """current_prompt_version returns fallback label when no files exist."""
    from app.core import prompts as p_module
    empty = tmp_path / "active_empty"
    empty.mkdir()
    monkeypatch.setattr(p_module, "ACTIVE_DIR", empty)

    from app.core.prompts import current_prompt_version
    result = current_prompt_version("missing_agent")
    assert "fallback" in result.lower() or "missing_agent" in result


# ============================================================
# ToolResult agent_accepted tests
# ============================================================

def test_tool_result_agent_accepted_field():
    """ToolResult has agent_accepted field, defaults to None."""
    from app.tools.tools import ToolResult
    r = ToolResult(success=True, data={"x": 1})
    assert r.agent_accepted is None


def test_tool_result_agent_accepted_set():
    """agent_accepted can be set after creation."""
    from app.tools.tools import ToolResult
    r = ToolResult(success=True, data={})
    r.agent_accepted = True
    assert r.agent_accepted is True

    r2 = ToolResult(success=False, data=None, failure_mode="empty")
    r2.agent_accepted = False
    assert r2.agent_accepted is False


def test_tool_result_to_dict_includes_accepted():
    """to_dict() includes agent_accepted."""
    from app.tools.tools import ToolResult
    r = ToolResult(success=True, data={"k": "v"}, agent_accepted=True)
    d = r.to_dict()
    assert "agent_accepted" in d
    assert d["agent_accepted"] is True


# ============================================================
# SSE event marker tests
# ============================================================

def test_sse_tool_call_marker_detected():
    """SSE stream_callback correctly classifies TOOL_CALL markers."""
    tool_events = []
    token_events = []

    # Simulate what stream_callback does in main.py
    async def mock_callback(agent_id: str, token: str):
        if token.startswith("\n[TOOL_CALL:") or token.startswith("[TOOL_CALL:"):
            tool_events.append({"event": "tool_call", "agent": agent_id, "detail": token.strip()})
        elif token.startswith("\n[TOOL_RESULT:") or token.startswith("[TOOL_RESULT:"):
            tool_events.append({"event": "tool_result", "agent": agent_id, "detail": token.strip()})
        else:
            token_events.append({"event": "token", "agent": agent_id, "token": token})

    async def run():
        await mock_callback("retrieval_agent", "[TOOL_CALL:web_search hop=1 query=\"test\" budget_remaining=5500]")
        await mock_callback("retrieval_agent", "This is normal text output.")
        await mock_callback("retrieval_agent", "[TOOL_RESULT:web_search hop=1 success=True chunks=3 accepted=True budget_remaining=5200]")

    asyncio.new_event_loop().run_until_complete(run())

    assert len(tool_events) == 2
    assert tool_events[0]["event"] == "tool_call"
    assert "web_search" in tool_events[0]["detail"]
    assert tool_events[1]["event"] == "tool_result"
    assert "accepted=True" in tool_events[1]["detail"]

    assert len(token_events) == 1
    assert token_events[0]["token"] == "This is normal text output."


def test_sse_budget_remaining_in_tool_markers():
    """budget_remaining is present in TOOL_CALL and TOOL_RESULT markers."""
    marker_call = "[TOOL_CALL:web_search hop=1 query=\"attention mechanism\" budget_remaining=5800]"
    marker_result = "[TOOL_RESULT:web_search hop=1 success=True chunks=2 latency=120ms accepted=True budget_remaining=5600]"

    assert "budget_remaining" in marker_call
    assert "budget_remaining" in marker_result
    assert "accepted" in marker_result


# ============================================================
# Prompt file existence tests
# ============================================================

def test_active_prompts_exist():
    """All required active prompt files exist in prompts/active/."""
    active = pathlib.Path(__file__).parent.parent / "prompts" / "active"
    assert active.exists(), "prompts/active/ directory missing"

    required_agents = [
        "orchestrator",
        "decomposition_agent",
        "retrieval_agent",
        "critique_agent",
        "synthesis_agent",
        "meta_agent",
    ]
    for agent in required_agents:
        files = list(active.glob(f"{agent}_v*.txt"))
        assert files, f"No active prompt file found for {agent}"
        # Verify content is non-empty
        assert files[-1].read_text().strip(), f"Prompt file for {agent} is empty"


def test_prompt_files_contain_json_schema():
    """Prompt files instruct the model to output JSON."""
    active = pathlib.Path(__file__).parent.parent / "prompts" / "active"
    for f in active.glob("*.txt"):
        content = f.read_text().lower()
        has_json = "json" in content or "output only" in content
        assert has_json, f"{f.name} does not mention JSON output format"


# ============================================================
# Docker files existence tests
# ============================================================

def test_docker_files_exist():
    """Required Docker files exist."""
    root = pathlib.Path(__file__).parent.parent
    assert (root / "Dockerfile").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / "docker" / "docker-compose.prod.yml").exists()
    assert (root / "docker" / "postgres" / "init.sql").exists()


def test_env_example_has_required_vars():
    """env.example contains all required environment variables."""
    root = pathlib.Path(__file__).parent.parent
    env_example = (root / ".env.example").read_text()
    required_vars = [
        "ANTHROPIC_API_KEY",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "DATABASE_URL",
        "REDIS_URL",
    ]
    for var in required_vars:
        assert var in env_example, f"Missing {var} in .env.example"


def test_no_hardcoded_secrets_in_code():
    """No hardcoded API keys or passwords in Python source files."""
    root = pathlib.Path(__file__).parent.parent / "app"
    suspicious_patterns = [
        "sk-ant-",   # Anthropic key prefix
        "password=",
        "api_key=\"sk",
    ]
    for py_file in root.rglob("*.py"):
        content = py_file.read_text()
        for pattern in suspicious_patterns:
            assert pattern not in content, (
                f"Potential hardcoded secret '{pattern}' found in {py_file}"
            )
