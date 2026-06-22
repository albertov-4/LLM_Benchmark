"""Shared text cleanup for model adapter outputs."""

from __future__ import annotations

from typing import Any
import re


THINKING_TAG_PATTERN = re.compile(
    r"<(?P<tag>think|thinking|reasoning|analysis)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_TAG_PATTERN = re.compile(
    r"^\s*<(?P<tag>think|thinking|reasoning|analysis)\b[^>]*>(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
FINAL_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(?:final\s+answer|final\s+plan|answer|plan|actions?)\s*:?\s*$"
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def join_nonempty(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip()


def extract_reasoning_from_text(raw_text: Any, explicit_reasoning: Any = "") -> dict[str, Any]:
    answer_text = clean_text(raw_text)
    reasoning_parts: list[str] = []
    notes: list[str] = []

    provider_reasoning = clean_text(explicit_reasoning)
    if provider_reasoning:
        reasoning_parts.append(provider_reasoning)
        notes.append("provider reasoning captured separately from raw_text.")

    inline_reasoning_parts: list[str] = []

    def replace_thinking_block(match: re.Match[str]) -> str:
        inline_reasoning = clean_text(match.group("body"))
        if inline_reasoning:
            inline_reasoning_parts.append(inline_reasoning)
        return "\n"

    answer_text = THINKING_TAG_PATTERN.sub(replace_thinking_block, answer_text)

    if inline_reasoning_parts:
        reasoning_parts.extend(inline_reasoning_parts)
        notes.append("inline reasoning extracted from raw_text.")
    else:
        open_match = OPEN_THINKING_TAG_PATTERN.match(answer_text)
        if open_match:
            body = clean_text(open_match.group("body"))
            marker = FINAL_MARKER_PATTERN.search(body)
            if marker:
                inline_reasoning = clean_text(body[: marker.start()])
                final_answer = clean_text(body[marker.end() :])
                if inline_reasoning and final_answer:
                    reasoning_parts.append(inline_reasoning)
                    answer_text = final_answer
                    notes.append("inline reasoning extracted from raw_text.")

    return {
        "raw_text": normalize_text(answer_text),
        "reasoning_text": join_nonempty(reasoning_parts),
        "notes": notes,
    }
