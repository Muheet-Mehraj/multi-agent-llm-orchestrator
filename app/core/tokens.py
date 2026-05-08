"""
Context budget manager. Tracks token consumption per agent per turn.
Catches budget overflows as policy violations - never silently truncates.
"""
import hashlib
from typing import Any
import json

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens. Uses tiktoken if available, else ~4 chars per token."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_tokens_for_messages(messages: list[dict]) -> int:
    """Count tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # role + content markers
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += count_tokens(block["text"])
    return total


def hash_content(content: Any) -> str:
    """Create a SHA-256 hash of content for logging."""
    if isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def serialize_structured(data: Any) -> str:
    """
    Lossless serialization for structured data (tool outputs, scores, citations).
    Used during compression to preserve critical information.
    """
    return json.dumps(data, default=str)


def compress_conversational(text: str, max_tokens: int) -> str:
    """
    Lossy compression for conversational filler.
    Keeps first and last portions, summarizes middle.
    """
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    # Keep ~40% from start, ~40% from end, drop middle
    chars = len(text)
    keep_chars = int(chars * (max_tokens / tokens) * 0.8)
    half = keep_chars // 2
    return text[:half] + "\n[...context compressed...]\n" + text[-half:]
