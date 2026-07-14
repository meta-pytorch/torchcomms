# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""JSON-serializable schemas for Triton comm tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

WorkItemKind = Literal["candidate", "baseline"]


@dataclass(frozen=True)
class TuningWorkItem:
    work_item_id: int
    kind: WorkItemKind
    spec_type: str
    spec: dict[str, Any]
    key_type: str
    key: dict[str, Any]
    config_type: str | None = None
    config: dict[str, Any] | None = None
    baseline: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "TuningWorkItem":
        return cls(
            work_item_id=int(data["work_item_id"]),
            kind=data["kind"],
            spec_type=str(data["spec_type"]),
            spec=dict(data["spec"]),
            key_type=str(data["key_type"]),
            key=dict(data["key"]),
            config_type=(
                None if data.get("config_type") is None else str(data["config_type"])
            ),
            config=(None if data.get("config") is None else dict(data["config"])),
            baseline=(None if data.get("baseline") is None else str(data["baseline"])),
        )
