# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.error import BadAlloc, OutOfMemory, ReservationError
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.pinned_memory_resource import (
    PinnedPoolProperties,
    is_pinned_memory_resources_supported,
)

if TYPE_CHECKING:
    import rmm.mr


@pytest.mark.parametrize(
    "error",
    [
        MemoryError,
        ReservationError,
        OutOfMemory,
        BadAlloc,
        TypeError,
        ValueError,
        IOError,
        IndexError,
        OverflowError,
        ArithmeticError,
        RuntimeError,
    ],
)
def test_error_handling(
    device_mr: rmm.mr.CudaMemoryResource, error: type[Exception]
) -> None:
    def spill(amount: int) -> int:
        raise error

    br = BufferResource(device_mr, periodic_spill_check=None)
    br.spill_manager.add_spill_function(spill, 0)
    with pytest.raises(error):
        br.spill_manager.spill(10)


def test_spill_function(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr, periodic_spill_check=None)
    track_spilled = [0]

    def spill_unlimited(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    f1 = br.spill_manager.add_spill_function(spill_unlimited, priority=0)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10
    track_spilled[0] = 0

    def spill_not_needed(amount: int) -> int:
        raise ValueError("shouldn't be needed")

    f2 = br.spill_manager.add_spill_function(spill_not_needed, priority=-1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10
    track_spilled[0] = 0

    def spill_limited(amount: int) -> int:
        return 5

    f3 = br.spill_manager.add_spill_function(spill_limited, priority=1)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 5  # Note `track_spilled` doesn't track `spill_limited`.
    track_spilled[0] = 0

    br.spill_manager.remove_spill_function(f3)
    assert br.spill_manager.spill(10) == 10
    assert track_spilled[0] == 10

    br.spill_manager.remove_spill_function(f1)
    with pytest.raises(ValueError, match="shouldn't be needed"):
        br.spill_manager.spill(10)

    br.spill_manager.remove_spill_function(f2)
    assert br.spill_manager.spill(10) == 0
    assert track_spilled[0] == 10


def test_spill_functions_are_isolated_by_memory_type(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    if not is_pinned_memory_resources_supported():
        pytest.skip("Pinned memory resources are not supported on this system")

    br = BufferResource(
        device_mr,
        pinned_pool_properties=PinnedPoolProperties(),
        memory_limits={MemoryType.DEVICE: 10, MemoryType.PINNED_HOST: 10},
        periodic_spill_check=None,
    )

    device_calls = [0]

    def device_spill(amount: int) -> int:
        device_calls[0] += 1
        return amount

    pinned_calls = [0]

    def pinned_spill(amount: int) -> int:
        pinned_calls[0] += 1
        return amount

    br.spill_manager.add_spill_function(device_spill, 0, MemoryType.DEVICE)
    pinned_fid = br.spill_manager.add_spill_function(
        pinned_spill, 0, MemoryType.PINNED_HOST
    )

    assert br.spill_manager.spill(5, MemoryType.DEVICE) == 5
    assert device_calls[0] == 1
    assert pinned_calls[0] == 0

    assert br.spill_manager.spill(5, MemoryType.PINNED_HOST) == 5
    assert device_calls[0] == 1
    assert pinned_calls[0] == 1

    br.spill_manager.remove_spill_function(pinned_fid)
    assert br.spill_manager.spill(5, MemoryType.PINNED_HOST) == 0
    assert br.spill_manager.spill(5, MemoryType.DEVICE) == 5
    assert device_calls[0] == 2
    assert pinned_calls[0] == 1


def test_spill_function_outlive_buffer_resource(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    spill_manager = BufferResource(device_mr, periodic_spill_check=None).spill_manager
    with pytest.raises(ValueError):
        spill_manager.add_spill_function(lambda x: x, 0)
    with pytest.raises(ValueError):
        spill_manager.remove_spill_function(0)
    with pytest.raises(ValueError):
        spill_manager.spill(10)


def test_periodic_spill_check(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    # Create a buffer resource with a negative limit to trigger spilling and
    # a periodic spill check enabled. (memory_available = limit - allocated; with
    # limit = -100 and no allocations, the available value is always negative.)
    br = BufferResource(
        device_mr,
        memory_limits={MemoryType.DEVICE: -100},
        periodic_spill_check=1e-3,
    )

    track_spilled = [0]

    def spill(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    br.spill_manager.add_spill_function(spill, priority=0)
    # After a short sleep, we expect many calls to `spill()` by the periodic check.
    time.sleep(0.1)
    assert track_spilled[0] > 1


def test_spill_to_make_headroom(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    # Create a buffer resource with a fixed limit of 100 bytes.
    br = BufferResource(
        device_mr,
        memory_limits={MemoryType.DEVICE: 100},
        periodic_spill_check=None,
    )

    track_spilled = [0]

    def spill(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    br.spill_manager.add_spill_function(spill, priority=0)
    # We expect to spill on the amount over 100 bytes (the fixed limit).
    assert br.spill_manager.spill_to_make_headroom(10) == 0
    assert br.spill_manager.spill_to_make_headroom(100) == 0
    assert br.spill_manager.spill_to_make_headroom(101) == 1
    assert br.spill_manager.spill_to_make_headroom(110) == 10


def test_reserve_device_memory_and_spill(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    # Create a buffer resource with a fixed limit of 100 bytes.
    br = BufferResource(
        device_mr,
        memory_limits={MemoryType.DEVICE: 100},
        periodic_spill_check=None,
    )

    track_spilled = [0]

    def spill(amount: int) -> int:
        track_spilled[0] += amount
        return amount

    br.spill_manager.add_spill_function(spill, priority=0)

    # We expect to spill on the amount over 100 bytes (the fixed limit).
    res = br.reserve_device_memory_and_spill(100, allow_overbooking=True)
    assert track_spilled[0] == 0
    res = br.reserve_device_memory_and_spill(1000, allow_overbooking=True)
    assert track_spilled[0] == 1000
    del res
