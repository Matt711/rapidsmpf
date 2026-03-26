/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

class Context;

/**
 * @brief Alias for an actor in a streaming graph.
 *
 * Actors represent coroutine-based asynchronous operations used throughout the streaming
 * graph.
 */
using Actor = coro::task<void>;

/**
 * @brief Runs a list of actors concurrently and waits for all to complete.
 *
 * This function schedules each actor and blocks until all of them have finished
 * execution. Typically used to launch multiple producer/consumer coroutines in parallel.
 *
 * @param actors A vector of actors to run.
 */
void run_actor_network(std::vector<Actor> actors);

/**
 * @brief Runs a list of actors concurrently and waits for all to complete.
 *
 * Like the single-argument overload, but additionally wraps each actor so that
 * the first failure triggers `ctx->cancel_network()`, shutting down all channels
 * and memory reservations. This causes any actors blocked on those operations to
 * unblock and exit, even if they share no channel dependency with the failing actor.
 *
 * @param actors A vector of actors to run.
 * @param ctx The context whose channels and memory reservations are cancelled on
 *            the first actor failure.
 */
void run_actor_network(std::vector<Actor> actors, std::shared_ptr<Context> ctx);

}  // namespace rapidsmpf::streaming
