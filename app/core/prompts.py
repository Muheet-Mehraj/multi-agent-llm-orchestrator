"""
Prompt loader. Loads agent system prompts from prompts/active/ directory.
Falls back to inline defaults if file not found.
This makes prompts editable without code changes and supports the
self-improving loop (approved rewrites update files in prompts/active/).
"""
import pathlib

PROMPTS_DIR = pathlib.Path(__file__).parent.parent.parent / "prompts"
ACTIVE_DIR = PROMPTS_DIR / "active"
PROPOSED_DIR = PROMPTS_DIR / "proposed"
HISTORY_DIR = PROMPTS_DIR / "history"


def load_prompt(agent_id: str, fallback: str) -> str:
    """
    Load the active prompt for an agent from file.
    Falls back to the provided inline string if no file exists.
    File naming: prompts/active/<agent_id>_v<N>.txt (loads highest N).
    """
    if not ACTIVE_DIR.exists():
        return fallback

    # Find the latest version file for this agent
    matches = sorted(ACTIVE_DIR.glob(f"{agent_id}_v*.txt"))
    if not matches:
        return fallback

    latest = matches[-1]
    try:
        content = latest.read_text().strip()
        if content:
            return content
    except Exception:
        pass

    return fallback


def save_proposed_prompt(agent_id: str, content: str, rewrite_id: str) -> pathlib.Path:
    """Save a meta-agent proposed prompt rewrite to prompts/proposed/."""
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROPOSED_DIR / f"{agent_id}_proposed_{rewrite_id[:8]}.txt"
    path.write_text(content)
    return path


def promote_prompt(agent_id: str, content: str) -> pathlib.Path:
    """
    Promote an approved prompt to active/.
    Archives the previous version to prompts/history/.
    """
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Archive existing active prompts for this agent
    for old in ACTIVE_DIR.glob(f"{agent_id}_v*.txt"):
        archive_path = HISTORY_DIR / old.name
        old.rename(archive_path)

    # Determine new version number
    history_files = list(HISTORY_DIR.glob(f"{agent_id}_v*.txt"))
    max_version = 0
    for f in history_files:
        try:
            v = int(f.stem.split("_v")[-1])
            max_version = max(max_version, v)
        except ValueError:
            pass

    new_version = max_version + 1
    new_path = ACTIVE_DIR / f"{agent_id}_v{new_version}.txt"
    new_path.write_text(content)
    return new_path


def current_prompt_version(agent_id: str) -> str:
    """Return the filename of the currently active prompt version."""
    if not ACTIVE_DIR.exists():
        return f"{agent_id}_v1 (inline fallback)"
    matches = sorted(ACTIVE_DIR.glob(f"{agent_id}_v*.txt"))
    return matches[-1].name if matches else f"{agent_id}_v1 (inline fallback)"
