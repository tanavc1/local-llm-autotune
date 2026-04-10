"""
Conversation chunking for autotune's recall system.

Strategy: pair consecutive user+assistant turns into single chunks.
A user message and its assistant reply form the natural semantic unit of
information — the question gives context, the answer gives content. Storing
them together keeps embeddings meaningful and makes retrieved chunks
self-contained when injected back into a prompt.

System messages are excluded; they describe the setup, not the exchange.
Chunks shorter than min_chars are dropped as noise.
"""

_USER_LIMIT = 1000
_ASST_LIMIT = 2000


def extract_chunks(messages: list[dict], min_chars: int = 60) -> list[dict]:
    """Split a conversation into memory-worthy chunks.

    Args:
        messages:  OpenAI-style list of {"role": ..., "content": ...} dicts.
        min_chars: Minimum total character length for a chunk to be kept.

    Returns:
        List of {"text": str, "turn_start": int, "turn_end": int} dicts.
        Indices reference positions in the filtered (non-system) message list.
    """
    # Strip system messages first; track original indices would complicate
    # nothing — callers only need positions within the conversational turns.
    turns = [m for m in messages if m.get("role") != "system"]

    chunks: list[dict] = []
    i = 0
    while i < len(turns):
        msg = turns[i]
        role = msg.get("role", "")

        if role == "user":
            user_text = (msg.get("content") or "")[:_USER_LIMIT]
            # Check if the next turn is an assistant reply.
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                asst_text = (turns[i + 1].get("content") or "")[:_ASST_LIMIT]
                text = f"User: {user_text}\nAssistant: {asst_text}"
                turn_start, turn_end = i, i + 1
                i += 2
            else:
                # Orphaned user message — no following assistant reply.
                text = f"User: {user_text}"
                turn_start = turn_end = i
                i += 1

            if len(text) >= min_chars:
                chunks.append({"text": text, "turn_start": turn_start, "turn_end": turn_end})
        else:
            # Assistant or unknown role not preceded by a user message; skip.
            i += 1

    return chunks


def estimate_conversation_value(messages: list[dict]) -> bool:
    """Return True if the conversation contains at least one meaningful exchange.

    A meaningful exchange requires a user message of at least 20 characters
    followed immediately by an assistant message of at least 50 characters.
    Conversations consisting only of very short back-and-forths are rejected.
    """
    turns = [m for m in messages if m.get("role") != "system"]

    for i in range(len(turns) - 1):
        if turns[i].get("role") != "user":
            continue
        if turns[i + 1].get("role") != "assistant":
            continue
        user_len = len(turns[i].get("content") or "")
        asst_len = len(turns[i + 1].get("content") or "")
        if user_len >= 20 and asst_len >= 50:
            return True

    return False
