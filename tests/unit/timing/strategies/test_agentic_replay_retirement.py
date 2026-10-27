# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session retirement when the session limit drops below the live count."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.timing.strategies._shared_helpers import (
    _build_real_trajectory_source,
    _make_credit,
    _make_dataset,
)


def _make_strategy(
    *, debt: int, phase: CreditPhase = CreditPhase.PROFILING
) -> tuple[AgenticReplayStrategy, list[tuple[str, int]], MagicMock]:
    ds = _make_dataset(num_traces=2, turns_per_trace=5)
    src = _build_real_trajectory_source(
        dataset=ds,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
    )
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = 1
    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    issuer.replay_gate = MagicMock()
    manager = MagicMock()
    manager.session_debt.return_value = debt
    scheduler = MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        session_tree_registry=MagicMock(),
        concurrency_manager=manager,
    )
    return strategy, issued, scheduler


def test_should_retire_true_only_for_non_final_root_turn_under_debt() -> None:
    strategy, _, _ = _make_strategy(debt=3)

    assert strategy.should_retire(
        _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    )
    assert not strategy.should_retire(
        _make_credit(conversation_id="trace_0", turn_index=4, num_turns=5)
    )


def test_should_retire_false_without_debt() -> None:
    strategy, _, _ = _make_strategy(debt=0)

    assert not strategy.should_retire(
        _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    )
    assert not strategy._retired_roots


def test_should_retire_false_in_warmup() -> None:
    strategy, _, _ = _make_strategy(debt=3, phase=CreditPhase.WARMUP)
    credit = _make_credit(
        conversation_id="trace_0", turn_index=2, num_turns=5, phase=CreditPhase.WARMUP
    )

    assert not strategy.should_retire(credit)


def test_should_retire_false_without_concurrency_manager() -> None:
    strategy, _, _ = _make_strategy(debt=3)
    strategy._concurrency_manager = None

    assert not strategy.should_retire(
        _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    )


@pytest.mark.asyncio
async def test_retired_root_does_not_dispatch_next_turn() -> None:
    strategy, issued, scheduler = _make_strategy(debt=1)
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._root_to_lane["xcorr"] = 0
    credit = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    assert strategy.should_retire(credit)
    issued.clear()

    await strategy.handle_credit_return(credit)

    assert issued == []
    scheduler.schedule_later.assert_not_called()
    assert "xcorr" not in strategy._retired_roots


@pytest.mark.asyncio
async def test_unretired_root_dispatches_next_turn() -> None:
    strategy, issued, scheduler = _make_strategy(debt=0)
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._root_to_lane["xcorr"] = 0
    credit = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    assert not strategy.should_retire(credit)
    issued.clear()

    await strategy.handle_credit_return(credit)

    assert ("trace_0", 3) in issued or scheduler.schedule_later.called


@pytest.mark.asyncio
async def test_retired_tree_drain_recycles_lane() -> None:
    """After the registry drains a retired tree, the lane is recycled like a finished one."""
    strategy, _, scheduler = _make_strategy(debt=1)
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._root_to_lane["xcorr"] = 0
    credit = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    strategy.should_retire(credit)
    await strategy.handle_credit_return(credit)

    strategy._on_tree_drained("xcorr", CreditPhase.PROFILING)

    scheduler.schedule_later.assert_called_once()
    assert "xcorr" not in strategy._root_to_lane


# --- Idle retirement -----------------------------------------------------------


async def _noop() -> None:
    return None


def test_delayed_root_turn_is_tracked_as_idle() -> None:
    strategy, _, scheduler = _make_strategy(debt=0)
    src = strategy.conversation_source
    real = src.get_next_turn_metadata
    src.get_next_turn_metadata = lambda credit: real(credit).model_copy(
        update={"delay_ms": 5000}
    )
    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=5)

    asyncio.run(strategy.handle_credit_return(credit))

    delay, wrapper = scheduler.schedule_later.call_args[0]
    assert delay == pytest.approx(5.0)
    handle, _due, inner = strategy._idle_next_turns["xcorr"]
    assert handle is scheduler.schedule_later.return_value
    wrapper.close()
    inner.close()


@pytest.mark.asyncio
async def test_fired_idle_turn_leaves_the_idle_table() -> None:
    strategy, _, _ = _make_strategy(debt=0)
    fired = []

    async def turn() -> None:
        fired.append(True)

    strategy._idle_next_turns["r"] = (1, 0.0, turn())
    await strategy._fire_idle_turn("r", strategy._idle_next_turns["r"][2])

    assert fired == [True]
    assert "r" not in strategy._idle_next_turns


def test_retire_idle_sessions_ends_most_idle_first() -> None:
    strategy, _, scheduler = _make_strategy(debt=3)
    registry = strategy._session_tree_registry
    registry.has_tree.return_value = True
    coros = {root: _noop() for root in ("soon", "later", "latest")}
    strategy._idle_next_turns = {
        "soon": (11, 10.0, coros["soon"]),
        "later": (22, 30.0, coros["later"]),
        "latest": (33, 50.0, coros["latest"]),
    }

    assert strategy.retire_idle_sessions(2) == 2

    assert [c.args[0] for c in scheduler.cancel_handle_id.call_args_list] == [33, 22]
    assert [c.args[0] for c in registry.on_root_terminal.call_args_list] == [
        "latest",
        "later",
    ]
    assert list(strategy._idle_next_turns) == ["soon"]
    coros["soon"].close()


def test_retire_idle_sessions_skips_roots_without_a_tree() -> None:
    strategy, _, scheduler = _make_strategy(debt=1)
    registry = strategy._session_tree_registry
    registry.has_tree.return_value = False
    strategy._idle_next_turns = {"gone": (5, 1.0, _noop())}

    assert strategy.retire_idle_sessions(1) == 0

    scheduler.cancel_handle_id.assert_called_once_with(5)
    registry.on_root_terminal.assert_not_called()
    assert not strategy._idle_next_turns


def test_retire_idle_sessions_noop_without_registry_or_demand() -> None:
    strategy, _, _ = _make_strategy(debt=2)
    strategy._idle_next_turns = {"r": (1, 1.0, _noop())}
    assert strategy.retire_idle_sessions(0) == 0
    strategy._session_tree_registry = None
    assert strategy.retire_idle_sessions(2) == 0
    strategy._idle_next_turns["r"][2].close()
