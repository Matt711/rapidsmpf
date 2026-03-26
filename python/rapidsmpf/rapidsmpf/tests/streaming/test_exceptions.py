# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.error import BadAlloc, OutOfMemory, ReservationError
from rapidsmpf.streaming._detail.libcoro_spawn_task import _make_exception
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.streaming.core.actor import CppActor, PyActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.message import Payload


@define_actor()
async def task_that_throws(ctx: Context, ch_in: Channel, ch_out: Channel) -> None:
    raise RuntimeError("Throwing in task")


@define_actor()
async def task_that_spins(ctx: Context, ch_in: Channel) -> None:
    while await ch_in.recv(ctx) is not None:
        pass


def test_task_exceptions(context: Context, py_executor: ThreadPoolExecutor) -> None:
    ch1: Channel[Payload] = context.create_channel()
    ch2: Channel[Payload] = context.create_channel()
    ch3: Channel[Payload] = context.create_channel()

    pull_task, deferred = pull_from_channel(context, ch3)

    actors: list[CppActor | PyActor] = [
        task_that_throws(context, ch1, ch2),
        task_that_spins(context, ch3),
        pull_task,
    ]

    with pytest.raises(RuntimeError, match="Throwing in task"):
        run_actor_network(actors=actors, py_executor=py_executor)

    messages = deferred.release()
    assert len(messages) == 0


@pytest.mark.parametrize(
    ("error_code", "expected_type"),
    [
        (-1, RuntimeError),
        (0, MemoryError),
        (1, TypeError),
        (2, ValueError),
        (3, IOError),
        (4, IndexError),
        (5, OverflowError),
        (6, ArithmeticError),
        (7, ReservationError),
        (8, OutOfMemory),
        (9, BadAlloc),
        (999, RuntimeError),  # Unknown code falls back to RuntimeError.
    ],
)
def test_make_exception_type_mapping(error_code: int, expected_type: type) -> None:
    """_make_exception preserves exception type from the async bridge error code."""
    exc = _make_exception(error_code, "test message")
    assert isinstance(exc, expected_type)
    assert "test message" in str(exc)
