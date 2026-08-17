from __future__ import annotations

from agentic_fitting.run import (
    _origins_after_accepted_action,
    _resolve_action_ids,
)


def _merge(ids: list[int]) -> dict[str, object]:
    return {
        "type": "merge",
        "primitive_ids": ids,
        "target_shape": "surface",
        "confidence": 0.9,
        "rationale": "test",
    }


def test_original_ids_survive_an_interleaved_multi_merge() -> None:
    origins = [{index} for index in range(9)]
    first, error = _resolve_action_ids(_merge([0, 2, 3, 6]), origins)
    assert not error
    assert first is not None
    origins = _origins_after_accepted_action(
        origins,
        first,
        proposed_count=6,
    )

    second, error = _resolve_action_ids(_merge([1, 4, 5]), origins)
    assert not error
    assert second is not None
    assert second["primitive_ids"] == [1, 2, 3]


def test_split_preserves_original_id_for_later_resolution() -> None:
    origins = [{0}, {1}, {2}]
    split = {
        "type": "split",
        "primitive_ids": [1],
        "target_shape": "surface",
        "confidence": 0.9,
        "rationale": "test",
    }
    origins = _origins_after_accepted_action(
        origins,
        split,
        proposed_count=4,
    )
    assert origins == [{0}, {1}, {1}, {2}]
    resolved, error = _resolve_action_ids(_merge([1, 2]), origins)
    assert resolved is None
    assert "ambiguous after split" in error
