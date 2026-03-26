# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper


cdef extern from * nogil:
    """
    #include <iostream>
    #include <coro/task.hpp>
    #include <rapidsmpf/error.hpp>

    /**
     * @brief Await a C++ coro::task and notify a Python asyncio.Future on completion.
     *
     * This is a C++-only helper used by Cython bindings to bridge C++ coroutines
     * and Python asyncio. The function awaits the provided C++ coro::task and,
     * upon completion, invokes the supplied callback to resolve a Python
     * asyncio.Future.
     *
     * The callback receives an integer error code (matching the ExceptionType enum
     * in exception_handling.pyx) alongside the exception message, so that the Python
     * side can reconstruct the original exception type rather than always raising
     * RuntimeError.
     *
     * Error codes:
     *   -1 = RuntimeError (default / unknown)
     *    0 = MemoryError     (std::bad_alloc)
     *    1 = TypeError       (std::bad_cast)
     *    2 = ValueError      (std::invalid_argument / std::domain_error)
     *    3 = IOError         (std::ios_base::failure)
     *    4 = IndexError      (std::out_of_range)
     *    5 = OverflowError   (std::overflow_error)
     *    6 = ArithmeticError (std::range_error / std::underflow_error)
     *    7 = ReservationError
     *    8 = OutOfMemory
     *    9 = BadAlloc
     *
     * On successful completion the callback is invoked with error_code=-1 and
     * a null error_msg pointer.
     *
     * @param cpp_set_py_future_typed Callback used to resolve or fail the Python
     * asyncio.Future with a typed error code.
     * @param py_future Owning wrapper holding the Python asyncio.Future.
     * @param task C++ coroutine task whose completion is being bridged to Python.
     *
     * @return A coro::task that completes after the underlying task has finished
     * and the Python Future has been notified.
     */
    coro::task<void> cython_libcoro_task_wrapper(
        void (*cpp_set_py_future_typed)(void*, int, const char *),
        rapidsmpf::OwningWrapper py_future,
        coro::task<void> task
    ) {
        try {
            co_await task;
            cpp_set_py_future_typed(py_future.get(), -1, NULL);
        } catch(rapidsmpf::reservation_error const& e) {
            cpp_set_py_future_typed(py_future.get(), 7, e.what());
        } catch(rapidsmpf::out_of_memory const& e) {
            cpp_set_py_future_typed(py_future.get(), 8, e.what());
        } catch(rapidsmpf::bad_alloc const& e) {
            cpp_set_py_future_typed(py_future.get(), 9, e.what());
        } catch(std::bad_alloc const& e) {
            cpp_set_py_future_typed(py_future.get(), 0, e.what());
        } catch(std::bad_cast const& e) {
            cpp_set_py_future_typed(py_future.get(), 1, e.what());
        } catch(std::domain_error const& e) {
            cpp_set_py_future_typed(py_future.get(), 2, e.what());
        } catch(std::invalid_argument const& e) {
            cpp_set_py_future_typed(py_future.get(), 2, e.what());
        } catch(std::ios_base::failure const& e) {
            cpp_set_py_future_typed(py_future.get(), 3, e.what());
        } catch(std::out_of_range const& e) {
            cpp_set_py_future_typed(py_future.get(), 4, e.what());
        } catch(std::overflow_error const& e) {
            cpp_set_py_future_typed(py_future.get(), 5, e.what());
        } catch(std::range_error const& e) {
            cpp_set_py_future_typed(py_future.get(), 6, e.what());
        } catch(std::underflow_error const& e) {
            cpp_set_py_future_typed(py_future.get(), 6, e.what());
        } catch(std::exception const& e) {
            cpp_set_py_future_typed(py_future.get(), -1, e.what());
        } catch(...) {
            cpp_set_py_future_typed(py_future.get(), -1, "Unknown exception");
        }
    }
    """

cdef void cpp_set_py_future_typed(
    void* py_future, int error_code, const char *error_msg
) noexcept nogil
