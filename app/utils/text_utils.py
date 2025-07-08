import re

def is_logical_break(text: str) -> bool:
    return bool(re.search(r"[.!?…]['”\"]?\s*$", text.strip()))

def chunk_text(text_blocks, max_chars=500):
    """
    text_blocks: list of dicts like [{"text": "..."}, ...]
    Returns: list of {"text": "..."}
    """
    chunks = []
    current_text = ""

    for block in text_blocks:
        seg_text = block["text"].strip()
        if not seg_text:
            continue

        proposed_text = (current_text + " " + seg_text).strip() if current_text else seg_text
        proposed_length = len(proposed_text)

        if (
            proposed_length >= max_chars
            and is_logical_break(seg_text)
        ):
            chunks.append({"text": proposed_text})
            current_text = ""
        else:
            current_text = proposed_text

    if current_text:
        chunks.append({"text": current_text})

    return chunks

def clean_metadata(meta: dict):
    return {k: v for k, v in meta.items() if v is not None}