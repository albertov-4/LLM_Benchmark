"""Small sequence-alignment helpers shared by reporting and repair feedback."""

from __future__ import annotations

from collections import Counter
from typing import Any


def normalize_action_sequence(actions: list[Any], lowercase: bool = True) -> list[str]:
    result: list[str] = []
    for action in actions or []:
        if not isinstance(action, str):
            continue
        normalized = " ".join(action.strip().split())
        if lowercase:
            normalized = normalized.lower()
        if normalized:
            result.append(normalized)
    return result


def common_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_item == right_item else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def action_bag_overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    return overlap / max(len(left), len(right), 1)


def contiguous_repetition_blocks(actions: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start = 0
    while start < len(actions):
        end = start
        while end + 1 < len(actions) and actions[end + 1] == actions[start]:
            end += 1
        if end > start:
            blocks.append({"action": actions[start], "start": start, "end": end, "length": end - start + 1})
        start = end + 1
    return blocks


def repetition_similarity(left: list[str], right: list[str]) -> float:
    left_repeated = sum(count - 1 for count in Counter(left).values() if count > 1)
    right_repeated = sum(count - 1 for count in Counter(right).values() if count > 1)
    denominator = max(left_repeated, right_repeated, 1)
    return 1.0 - (abs(left_repeated - right_repeated) / denominator)


def detect_adjacent_swaps(left: list[str], right: list[str]) -> list[dict[str, Any]]:
    swaps: list[dict[str, Any]] = []
    max_index = min(len(left), len(right)) - 1
    index = 0
    while index < max_index:
        if left[index] == right[index + 1] and left[index + 1] == right[index] and left[index] != left[index + 1]:
            swaps.append({"index": index, "raw_pair": [left[index], left[index + 1]], "reasoning_pair": [right[index], right[index + 1]]})
            index += 2
        else:
            index += 1
    return swaps


def compute_sequence_alignment(raw_actions: list[Any], reasoning_actions: list[Any]) -> dict[str, Any]:
    raw = normalize_action_sequence(raw_actions)
    reasoning = normalize_action_sequence(reasoning_actions)
    max_len = max(len(raw), len(reasoning), 1)
    min_len = min(len(raw), len(reasoning))
    prefix = common_prefix_length(raw, reasoning)
    lcs = lcs_length(raw, reasoning)
    length_ratio = min_len / max_len
    prefix_ratio = prefix / max(min_len, 1)
    lcs_ratio = lcs / max_len
    bag_score = action_bag_overlap_score(raw, reasoning)
    repeat_similarity = repetition_similarity(raw, reasoning)
    adjacent_swaps = detect_adjacent_swaps(raw, reasoning)

    first_mismatch_index = None
    mismatch_examples: list[dict[str, Any]] = []
    displaced_actions: list[dict[str, Any]] = []
    for index in range(max(len(raw), len(reasoning))):
        raw_action = raw[index] if index < len(raw) else None
        reasoning_action = reasoning[index] if index < len(reasoning) else None
        if raw_action == reasoning_action:
            continue
        if first_mismatch_index is None:
            first_mismatch_index = index
        if len(mismatch_examples) < 5:
            mismatch_examples.append({"index": index, "raw_action": raw_action, "reasoning_action": reasoning_action})
        if raw_action in reasoning or reasoning_action in raw:
            displaced_actions.append({"index": index, "raw_action": raw_action, "reasoning_action": reasoning_action})

    raw_counts = Counter(raw)
    reasoning_counts = Counter(reasoning)
    structural_alignment = (
        0.35 * lcs_ratio
        + 0.25 * prefix_ratio
        + 0.20 * bag_score
        + 0.10 * length_ratio
        + 0.10 * repeat_similarity
    )
    raw_blocks = contiguous_repetition_blocks(raw)
    reasoning_blocks = contiguous_repetition_blocks(reasoning)
    return {
        "raw_actions": raw,
        "reasoning_actions": reasoning,
        "raw_plan_length": len(raw),
        "reasoning_plan_length": len(reasoning),
        "length_ratio": length_ratio,
        "exact_sequence_match": raw == reasoning,
        "common_prefix_length": prefix,
        "common_prefix_ratio": prefix_ratio,
        "lcs_length": lcs,
        "lcs_ratio": lcs_ratio,
        "action_bag_overlap_score": bag_score,
        "repetition_similarity": repeat_similarity,
        "structural_alignment": max(0.0, min(float(structural_alignment), 1.0)),
        "raw_distinct_action_count": len(raw_counts),
        "reasoning_distinct_action_count": len(reasoning_counts),
        "raw_action_frequencies": dict(raw_counts),
        "reasoning_action_frequencies": dict(reasoning_counts),
        "raw_repeated_action_total": sum(count - 1 for count in raw_counts.values() if count > 1),
        "reasoning_repeated_action_total": sum(count - 1 for count in reasoning_counts.values() if count > 1),
        "raw_repetition_block_count": len(raw_blocks),
        "reasoning_repetition_block_count": len(reasoning_blocks),
        "raw_repetition_blocks": raw_blocks,
        "reasoning_repetition_blocks": reasoning_blocks,
        "adjacent_swaps": adjacent_swaps,
        "adjacent_swap_count": len(adjacent_swaps),
        "displaced_actions": displaced_actions,
        "displaced_action_count": len(displaced_actions),
        "first_mismatch_index": first_mismatch_index,
        "mismatch_examples": mismatch_examples,
    }
