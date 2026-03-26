# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel

if TYPE_CHECKING:
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
        run_actor_network(actors=actors, py_executor=py_executor, context=context)

    messages = deferred.release()
    assert len(messages) == 0


def test_cancel_network_unblocks_unconnected_actor(
    context: Context,
    py_executor: ThreadPoolExecutor,
) -> None:
    """
    cancel_network() must shut down all registered channels, unblocking any
    actor that is blocked on a recv with no producer.

    actor_calls_cancel calls ctx.cancel_network() and exits normally (no
    exception raised). actor_waits is blocked on ch_in.recv() which has no
    producer. Without cancel_network() being implemented, asyncio.gather
    waits for both actors and hangs forever because actor_waits never
    unblocks. With cancel_network(), ch_in is shut down and actor_waits
    sees None and exits.
    """

    @define_actor()
    async def actor_calls_cancel(ctx: Context, ch_dummy: Channel) -> None:
        # Broadcast shutdown to all channels — does not raise.
        ctx.cancel_network()

    @define_actor()
    async def actor_waits(ctx: Context, ch_in: Channel) -> None:
        # Blocks until ch_in is shut down by cancel_network().
        msg = await ch_in.recv(ctx)
        assert msg is None

    ch_dummy: Channel[Payload] = context.create_channel()
    ch_in: Channel[Payload] = context.create_channel()

    run_actor_network(
        actors=[actor_calls_cancel(context, ch_dummy), actor_waits(context, ch_in)],
        py_executor=py_executor,
        context=context,
    )
