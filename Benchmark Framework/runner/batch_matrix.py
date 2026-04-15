"""Helpers to build benchmark matrices."""


def build_default_matrix() -> dict[str, list[str]]:
    """A minimal benchmark matrix useful as a starting point."""
    return {
        "tiers": ["easy", "medium", "hard"],
        "protocols": [
            "direct_plan",
            "direct_plan_with_rationale",
            "iterative_repair",
        ],
    }
