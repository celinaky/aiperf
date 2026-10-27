# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-concurrency time series configuration."""

from __future__ import annotations

import math
from typing import Any

import orjson
from pydantic import ConfigDict, Field, model_validator

from aiperf.common.path_safety import safe_read_template_path
from aiperf.config.base import BaseConfig


class ConcurrencySeriesPoint(BaseConfig):
    """One session-concurrency control point."""

    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(
        ge=0.0,
        description="Elapsed seconds from the start of the profiling phase.",
    )
    concurrency: int = Field(
        ge=1,
        description="Session concurrency limit at this point.",
    )


class ConcurrencySeriesConfig(BaseConfig):
    """Piecewise-linear session-concurrency schedule.

    The limit is interpolated between points and rounded to the nearest
    integer; before the first point and after the last it holds their values.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description="JSON file path containing concurrency control points.",
    )
    points: list[ConcurrencySeriesPoint] = Field(
        default_factory=list,
        description="Control points with strictly increasing time_s.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_shorthand(cls, data: Any) -> Any:
        """Allow `concurrencySeries: path.json` and `concurrencySeries: [{...}, ...]` shorthand."""
        if isinstance(data, str):
            return {"path": data}
        if isinstance(data, list):
            return {"points": data}
        return data

    @model_validator(mode="after")
    def load_path_points(self) -> ConcurrencySeriesConfig:
        """Load JSON points when a path is supplied."""
        if self.path is None:
            _validate_points(self.points)
            return self
        if self.points:
            raise ValueError(
                "concurrency_series.path and concurrency_series.points are mutually exclusive"
            )
        self.points = read_concurrency_series_json(self.path)
        self.path = None
        return self

    @property
    def max_concurrency(self) -> int:
        """Return the peak concurrency, which sizes the phase's session lanes."""
        return max(point.concurrency for point in self.points)


def read_concurrency_series_json(path: str) -> list[ConcurrencySeriesPoint]:
    """Read and validate a JSON concurrency series file."""
    text = safe_read_template_path(path)
    if text is None:
        raise ValueError(f"Cannot read concurrency series JSON '{path}'")

    try:
        document = orjson.loads(text)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid concurrency series JSON '{path}'") from exc

    if isinstance(document, list):
        points_data = document
    elif isinstance(document, dict) and set(document) == {"points"}:
        points_data = document["points"]
    else:
        raise ValueError(
            "Concurrency series JSON must be an object with points or a points array"
        )
    if not isinstance(points_data, list):
        raise ValueError("Concurrency series JSON points must be an array")

    points: list[ConcurrencySeriesPoint] = []
    for index, point_data in enumerate(points_data):
        try:
            points.append(ConcurrencySeriesPoint.model_validate(point_data))
        except ValueError as exc:
            raise ValueError(
                f"Invalid concurrency series point {index}: {exc}"
            ) from exc

    _validate_points(points)
    return points


def _validate_points(points: list[ConcurrencySeriesPoint]) -> None:
    if len(points) < 2:
        raise ValueError("Concurrency series requires at least two points")
    for point in points:
        if not math.isfinite(point.time_s):
            raise ValueError("Concurrency series time_s values must be finite")
    previous_time = points[0].time_s
    for point in points[1:]:
        if point.time_s <= previous_time:
            raise ValueError(
                "Concurrency series time_s values must be strictly increasing"
            )
        previous_time = point.time_s
