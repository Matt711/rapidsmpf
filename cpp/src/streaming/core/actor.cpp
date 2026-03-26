/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>

namespace rapidsmpf::streaming {

void run_actor_network(std::vector<Actor> actors) {
    coro_results(coro::sync_wait(coro::when_all(std::move(actors))));
}

namespace {

Actor cancelling_wrapper(Actor inner, std::shared_ptr<Context> ctx) {
    try {
        co_await std::move(inner);
    } catch (...) {
        ctx->cancel_network();
        throw;
    }
}

}  // namespace

void run_actor_network(std::vector<Actor> actors, std::shared_ptr<Context> ctx) {
    std::vector<Actor> wrapped;
    wrapped.reserve(actors.size());
    for (auto& a : actors) {
        wrapped.push_back(cancelling_wrapper(std::move(a), ctx));
    }
    coro_results(coro::sync_wait(coro::when_all(std::move(wrapped))));
}

}  // namespace rapidsmpf::streaming
