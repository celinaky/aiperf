# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session retirement: a non-final root turn ends its tree when the strategy asks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from tests.unit.credit.test_callback_handler import make_credit, make_credit_return


@pytest.fixture
def mock_concurrency():
    mock = MagicMock()
    mock.release_session_slot = MagicMock()
    mock.release_prefill_slot = MagicMock()
    return mock


@pytest.fixture
def mock_progress():
    mock = MagicMock()
    mock.increment_returned = MagicMock(return_value=False)
    mock.increment_prefill_released = MagicMock()
    mock.all_credits_returned_event = asyncio.Event()
    mock.in_flight_sessions = 0
    return mock


@pytest.fixture
def mock_lifecycle():
    mock = MagicMock()
    mock.is_complete = False
    return mock


@pytest.fixture
def mock_stop_checker():
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_strategy():
    mock = MagicMock()
    mock.handle_credit_return = AsyncMock()
    return mock


def _handler_with_registry(
    mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker, mock_strategy
):
    registry = MagicMock()
    registry.has_tree.return_value = True
    handler = CreditCallbackHandler(mock_concurrency, session_tree_registry=registry)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    return handler, registry


@pytest.mark.asyncio
async def test_retired_root_turn_calls_registry_on_root_terminal(
    mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker, mock_strategy
):
    handler, registry = _handler_with_registry(
        mock_concurrency,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
    )
    mock_strategy.should_retire = MagicMock(return_value=True)
    credit = make_credit(turn_index=1, num_turns=5)

    await handler.on_credit_return("worker-1", make_credit_return(credit))

    mock_strategy.should_retire.assert_called_once_with(credit)
    registry.on_root_terminal.assert_called_once_with(
        credit.effective_root_correlation_id
    )
    mock_strategy.handle_credit_return.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_retired_root_turn_keeps_tree_open(
    mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker, mock_strategy
):
    handler, registry = _handler_with_registry(
        mock_concurrency,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
    )
    mock_strategy.should_retire = MagicMock(return_value=False)
    credit = make_credit(turn_index=1, num_turns=5)

    await handler.on_credit_return("worker-1", make_credit_return(credit))

    registry.on_root_terminal.assert_not_called()


@pytest.mark.asyncio
async def test_strategy_without_retirement_support_is_ignored(
    mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker, mock_strategy
):
    """A strategy hook that does not return exactly True (including MagicMock
    auto-attributes) must never retire a session."""
    handler, registry = _handler_with_registry(
        mock_concurrency,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
    )
    credit = make_credit(turn_index=1, num_turns=5)

    await handler.on_credit_return("worker-1", make_credit_return(credit))

    registry.on_root_terminal.assert_not_called()


@pytest.mark.asyncio
async def test_retirement_not_consulted_without_registry(
    mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker, mock_strategy
):
    handler = CreditCallbackHandler(mock_concurrency)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    mock_strategy.should_retire = MagicMock(return_value=True)

    await handler.on_credit_return(
        "worker-1", make_credit_return(make_credit(turn_index=1, num_turns=5))
    )

    mock_strategy.should_retire.assert_not_called()
    mock_concurrency.release_session_slot.assert_not_called()
