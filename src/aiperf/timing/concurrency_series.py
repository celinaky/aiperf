# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuous session-concurrency series controller."""

from __future__ import annotations

import logging
from collections.abc import Callable

from aiperf.config.concurrency_series import ConcurrencySeriesConfig
from aiperf.timing.rate_series import RateSeriesController

logger = logging.getLogger(__name__)


class ConcurrencySeriesController(RateSeriesController):
    """Apply piecewise-linear session-limit updates until stopped.

    Reuses the rate-series interpolation loop; values are rounded to the
    nearest integer and the setter is only called when the limit changes.
    """

    def __init__(
        self,
        setter: Callable[[int], None],
        config: ConcurrencySeriesConfig,
        update_interval: float,
    ) -> None:
        self._setter = setter
        self._update_interval = update_interval
        self._start_delay = 0.0
        self._times = [point.time_s for point in config.points]
        self._qps = [float(point.concurrency) for point in config.points]
        self._task = None
        self._current: int | None = None

    def _set_rate(self, rate: float) -> bool:
        limit = max(1, round(rate))
        if limit == self._current:
            return True
        try:
            self._setter(limit)
        except Exception:
            logger.exception("Concurrency series update failed for limit=%s", limit)
            return False
        self._current = limit
        return True
