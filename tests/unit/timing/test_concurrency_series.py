# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib

import pytest

from aiperf.config.concurrency_series import ConcurrencySeriesConfig
from aiperf.timing.concurrency_series import ConcurrencySeriesController
from tests.harness.time_traveler import TimeTraveler


def series_config() -> ConcurrencySeriesConfig:
    return ConcurrencySeriesConfig(
        points=[
            {"time_s": 0.0, "concurrency": 1},
            {"time_s": 10.0, "concurrency": 5},
            {"time_s": 20.0, "concurrency": 1},
        ]
    )


class TestConcurrencySeriesController:
    def test_value_at_interpolates_up_and_down(self) -> None:
        controller = ConcurrencySeriesController(
            setter=lambda value: None, config=series_config(), update_interval=0.1
        )

        assert controller.value_at(0.0) == pytest.approx(1.0)
        assert controller.value_at(5.0) == pytest.approx(3.0)
        assert controller.value_at(10.0) == pytest.approx(5.0)
        assert controller.value_at(15.0) == pytest.approx(3.0)
        assert controller.value_at(30.0) == pytest.approx(1.0)

    def test_set_rate_rounds_and_skips_unchanged_limits(self) -> None:
        values: list[int] = []
        controller = ConcurrencySeriesController(
            setter=values.append, config=series_config(), update_interval=0.1
        )

        for value in (1.0, 1.2, 1.6, 2.4, 2.0, 0.2):
            assert controller._set_rate(value)

        assert values == [1, 2, 1]
        assert all(isinstance(v, int) for v in values)

    @pytest.mark.asyncio
    async def test_start_follows_series_then_holds_final_value(
        self, time_traveler: TimeTraveler
    ) -> None:
        values: list[int] = []
        controller = ConcurrencySeriesController(
            setter=values.append, config=series_config(), update_interval=1.0
        )

        task = controller.start()
        await time_traveler.sleep(0.1)
        assert values == [1]

        await time_traveler.sleep(10.0)
        assert values[-1] == 5
        assert values == sorted(values)

        await time_traveler.sleep(11.0)
        assert values[-1] == 1
        assert all(a != b for a, b in zip(values, values[1:], strict=False))
        assert task.done()

        controller.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
