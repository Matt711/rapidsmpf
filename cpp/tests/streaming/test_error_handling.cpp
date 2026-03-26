/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingErrorHandling = BaseStreamingFixture;

TEST_F(StreamingErrorHandling, UnhandledException) {
    std::vector<Actor> actors;

    actors.push_back([](Context& ctx) -> Actor {
        co_await ctx.executor()->schedule();
        throw std::runtime_error("unhandled_exception");
    }(*ctx));

    EXPECT_THROW(run_actor_network(std::move(actors)), std::runtime_error);
}

TEST_F(StreamingErrorHandling, ProducerThrows) {
    auto ch = ctx->create_channel();
    std::vector<Actor> actors;

    // Producer actor.
    actors.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_out) -> Actor {
            ShutdownAtExit c{ch_out};
            co_await ctx->executor()->schedule();
            throw std::runtime_error("some unhandled exception");
        }(ctx, ch)
    );

    // Consumer actor.
    actors.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_in) -> Actor {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            std::ignore = ch_in->receive();
        }(ctx, ch)
    );

    EXPECT_THROW(run_actor_network(std::move(actors)), std::runtime_error);
}

TEST_F(StreamingErrorHandling, ConsumerThrows) {
    auto ch = ctx->create_channel();
    std::vector<Actor> actors;

    // Producer actor.
    actors.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_out) -> Actor {
            ShutdownAtExit c{ch_out};
            co_await ctx->executor()->schedule();
            co_await ch_out->send(
                Message{0, std::make_unique<int>(42), ContentDescription{}}
            );
            co_await ch_out->drain(ctx->executor());
        }(ctx, ch)
    );

    // Consumer actor.
    actors.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_in) -> Actor {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            throw std::runtime_error("some unhandled exception");
        }(ctx, ch)
    );

    EXPECT_THROW(run_actor_network(std::move(actors)), std::runtime_error);
}

TEST_F(StreamingErrorHandling, UnconnectedActorCancelledOnFailure) {
    // Actor A fails immediately.  Actor B is unconnected to A but blocks
    // forever on a channel recv.  With cancel_network(), ctx's cancel_network()
    // is broadcast when A fails, which shuts down B's channel and lets B exit.
    // Without this mechanism, run_actor_network would hang forever.
    auto ch_b = ctx->create_channel();
    std::vector<Actor> actors;

    // Actor A: fails immediately.
    actors.push_back([](Context& ctx) -> Actor {
        co_await ctx.executor()->schedule();
        throw std::runtime_error("actor_a_failure");
    }(*ctx));

    // Actor B: unconnected to A, blocks on a recv that would never complete
    // unless cancel_network() shuts down ch_b.  No ShutdownAtExit here because
    // cancel_network() already shuts down ch_b; using ShutdownAtExit would cause
    // a re-entrant Channel::shutdown() from within the inline resume of rb_.shutdown(),
    // leading to a segfault.
    actors.push_back(
        [](std::shared_ptr<Channel> ch_b) -> Actor {
            std::ignore = co_await ch_b->receive();
        }(ch_b)
    );

    try {
        run_actor_network(std::move(actors), ctx);
        FAIL() << "Expected std::runtime_error";
    } catch (std::runtime_error const& e) {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("actor_a_failure"));
    }
}
