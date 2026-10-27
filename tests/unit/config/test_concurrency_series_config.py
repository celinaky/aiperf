# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.concurrency_series import ConcurrencySeriesConfig
from aiperf.config.phases import ConcurrencyPhase
from aiperf.plugin.enums import PhaseType

POINTS = [
    {"time_s": 0.0, "concurrency": 1},
    {"time_s": 60.0, "concurrency": 16},
    {"time_s": 120.0, "concurrency": 4},
]


def test_concurrency_series_config_loads_json_path(tmp_path: Path) -> None:
    json_path = tmp_path / "series.json"
    json_path.write_text(
        '{"points":[{"time_s":0,"concurrency":1},{"time_s":60,"concurrency":16}]}',
        encoding="utf-8",
    )

    series = ConcurrencySeriesConfig(path=str(json_path))

    assert series.path is None
    assert [p.concurrency for p in series.points] == [1, 16]
    assert series.max_concurrency == 16


def test_concurrency_series_config_accepts_inline_points() -> None:
    series = ConcurrencySeriesConfig(points=POINTS)
    assert [p.time_s for p in series.points] == [0.0, 60.0, 120.0]
    assert series.max_concurrency == 16


def test_concurrency_series_config_rejects_single_point() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        ConcurrencySeriesConfig(points=POINTS[:1])


def test_concurrency_series_config_rejects_non_increasing_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ConcurrencySeriesConfig(points=[POINTS[0], {"time_s": 0.0, "concurrency": 2}])


def test_concurrency_series_config_rejects_zero_concurrency() -> None:
    with pytest.raises(ValueError):
        ConcurrencySeriesConfig(points=[POINTS[0], {"time_s": 10.0, "concurrency": 0}])


def test_phase_fills_concurrency_from_series_peak() -> None:
    phase = ConcurrencyPhase(
        name="profiling",
        type=PhaseType.CONCURRENCY,
        duration=300,
        concurrency_series=POINTS,
    )
    assert phase.concurrency == 16


def test_phase_accepts_concurrency_equal_to_series_peak() -> None:
    phase = ConcurrencyPhase(
        name="profiling",
        type=PhaseType.CONCURRENCY,
        duration=300,
        concurrency=16,
        concurrency_series=POINTS,
    )
    assert phase.concurrency == 16


def test_phase_rejects_concurrency_different_from_series_peak() -> None:
    with pytest.raises(ValueError, match="must equal the concurrency_series maximum"):
        ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            duration=300,
            concurrency=8,
            concurrency_series=POINTS,
        )


def test_phase_rejects_series_with_ramp() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            duration=300,
            concurrency_ramp=30,
            concurrency_series=POINTS,
        )
