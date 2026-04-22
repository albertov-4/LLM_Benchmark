"""Mock adapter utilities for benchmark smoke tests."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MockAdapter:
    """Return scripted outputs across successive generate calls."""

    scripted_outputs: list[str] = field(default_factory=lambda: [""])
    model_id: str = "mock-model"
    _call_count: int = field(init=False, default=0)
    last_messages: list[dict[str, str]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._call_count = 0
        self.last_messages = []

    def generate(self, messages: list[dict[str, str]]) -> dict[str, object]:
        self.last_messages = [dict(message) for message in messages]
        if self._call_count < len(self.scripted_outputs):
            raw_text = self.scripted_outputs[self._call_count]
        else:
            raw_text = self.scripted_outputs[-1] if self.scripted_outputs else ""

        self._call_count += 1
        return {
            "model_id": self.model_id,
            "raw_text": raw_text,
            "usage": {},
            "latency_s": 0.0,
            "message_count": len(messages),
            "call_index": self._call_count,
        }


def build_mock_adapter_for_suite(model_entry: dict[str, object], protocol_config) -> MockAdapter:
    """Factory compatible with `run_suite(adapter_factory=...)`."""
    model_id = str(model_entry.get("model_id", "mock-model"))

    if getattr(protocol_config, "max_iterations", 1) > 1:
        return MockAdapter(
            scripted_outputs=["", "(move a b)"],
            model_id=model_id,
        )

    return MockAdapter(
        scripted_outputs=["(move a b)"],
        model_id=model_id,
    )
