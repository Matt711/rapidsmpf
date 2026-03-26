# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from rapidsmpf.error import BadAlloc, OutOfMemory, ReservationError


async def _set_result(future):
    future.set_result(None)


async def _set_exception(future, exc):
    future.set_exception(exc)


# Maps ExceptionType integer codes (from exception_handling.pyx) to Python exception
# classes.  Code -1 (RuntimeError) is handled as the default fallback.
_EXCEPTION_CODE_MAP = {
    0: MemoryError,
    1: TypeError,
    2: ValueError,
    3: IOError,
    4: IndexError,
    5: OverflowError,
    6: ArithmeticError,
    7: ReservationError,
    8: OutOfMemory,
    9: BadAlloc,
}


def _make_exception(error_code: int, msg: str) -> Exception:
    """Reconstruct a typed Python exception from a bridge error code and message."""
    return _EXCEPTION_CODE_MAP.get(error_code, RuntimeError)(msg)


cdef void cpp_set_py_future_typed(
    void* py_future, int error_code, const char *error_msg
) noexcept nogil:
    """
    Set the result or a typed exception on an asyncio Future from C++ code.

    This function is intended to be called from C++ via the
    ``cython_libcoro_task_wrapper`` helper defined in libcoro_spawn_task.pxd.
    It safely schedules completion of a Python asyncio Future on its associated
    event loop using thread-safe coroutine submission.

    On successful completion (``error_msg == NULL``) the Future is resolved with
    ``None``.  On failure, the Future is completed with an exception whose type
    corresponds to ``error_code`` (see ``_EXCEPTION_CODE_MAP``).

    Parameters
    ----------
    py_future
        Opaque pointer to a Python ``asyncio.Future`` object.
    error_code
        Integer code identifying the exception type (see ExceptionType enum in
        exception_handling.pyx). Use -1 for RuntimeError.
    error_msg
        C string with the exception message, or ``NULL`` on success.
    """
    with gil:
        future = (<object?> py_future)
        if error_msg == NULL:
            asyncio.run_coroutine_threadsafe(_set_result(future), future.get_loop())
        else:
            exc = _make_exception(error_code, error_msg.decode("utf-8"))
            asyncio.run_coroutine_threadsafe(
                _set_exception(future, exc),
                future.get_loop()
            )
