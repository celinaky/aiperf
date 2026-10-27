# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--concurrency-series`` routing through the CLI converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _write_series(tmp_path: Path) -> Path:
    json_path = tmp_path / "concurrency.json"
    json_path.write_text(
        '{"points":[{"time_s":0,"concurrency":1},{"time_s":60,"concurrency":10},'
        '{"time_s":120,"concurrency":2}]}',
        encoding="utf-8",
    )
    return json_path


def _cli(**loadgen: object) -> CLIConfig:
    return CLIConfig(
        url="http://localhost:8000/test",
        model_names=["test-model"],
        benchmark_duration=120.0,
        **loadgen,
    )


def test_concurrency_series_lands_on_profiling_phase(tmp_path: Path) -> None:
    prof = build_profiling(
        _cli(concurrency=10, concurrency_series=_write_series(tmp_path))
    )

    assert prof["concurrency"] == 10
    assert prof["concurrency_series"]["points"][2] == {
        "time_s": 120.0,
        "concurrency": 2,
    }


def test_concurrency_series_with_ramp_duration_raises(tmp_path: Path) -> None:
    cli = _cli(
        concurrency=10,
        concurrency_ramp_duration=30.0,
        concurrency_series=_write_series(tmp_path),
    )
    with pytest.raises(
        ValueError, match=r"concurrency-ramp-duration.*concurrency-series"
    ):
        build_profiling(cli)


def test_concurrency_series_resolves_into_phase_config(tmp_path: Path) -> None:
    config = convert_cli_to_aiperf(
        _cli(concurrency=10, concurrency_series=_write_series(tmp_path))
    )

    phase = next(p for p in config.benchmark.phases if p.kind == "profiling")
    assert phase.concurrency == 10
    assert phase.concurrency_series is not None
    assert phase.concurrency_series.max_concurrency == 10


def test_concurrency_series_without_concurrency_fills_peak(tmp_path: Path) -> None:
    config = convert_cli_to_aiperf(_cli(concurrency_series=_write_series(tmp_path)))

    phase = next(p for p in config.benchmark.phases if p.kind == "profiling")
    assert phase.concurrency == 10
