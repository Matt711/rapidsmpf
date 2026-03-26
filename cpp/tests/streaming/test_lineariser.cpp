/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <pthread.h>

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace actor = rapidsmpf::streaming::actor;

class StreamingLineariser : public BaseStreamingFixture {
  public:
    // In the following test, a large stack may be required when compiled in debug mode.
    // Therefore, this setup spawns threads with a 64 MiB stack size.
    // See: <https://github.com/rapidsai/rapidsmpf/issues/621>.
    void SetUp() override {
        // Save current default attrs
        pthread_attr_t old_attr;
        pthread_attr_init(&old_attr);
        pthread_getattr_default_np(&old_attr);

        // Set a 64 MiB default stack size for new threads created after this point.
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        constexpr std::size_t big = 1 << 26;
        int err = pthread_attr_setstacksize(&attr, big);
        ASSERT_EQ(err, 0) << "pthread_attr_setstacksize: " << strerror(err);

        err = pthread_setattr_default_np(&attr);
        ASSERT_EQ(err, 0) << "pthread_setattr_default_np: " << strerror(err);

        pthread_attr_destroy(&attr);

        // Create the executor threads, they inherit the big default
        SetUpWithThreads(8);

        // Stash previous default to restore in TearDown
        saved_default_attr_ = old_attr;
    }

    void TearDown() override {
        // Restore previous default.
        if (saved_) {
            pthread_setattr_default_np(&saved_default_attr_);
            pthread_attr_destroy(&saved_default_attr_);
        }
        BaseStreamingFixture::TearDown();
    }

  private:
    pthread_attr_t saved_default_attr_{};
    bool saved_{true};
};

TEST_F(StreamingLineariser, ChOutShutdownUnblocksProducers) {
    // Regression test for the deadlock in Lineariser::drain() when ch_out is shut
    // down mid-stream.
    //
    // Root cause (pre-fix):
    //   When ch_out_->send() returned false (Channel shut down), drain() called
    //   `break` from the inner `for` loop, skipping `co_await receipt`.  The
    //   BoundedQueue semaphore therefore remained at 0.  The outer `while
    //   (!queues_.empty())` loop then re-entered and drain() blocked on
    //   `co_await q->receive()`.  Meanwhile the producer tried to acquire the
    //   next ticket but the semaphore was 0 → permanent deadlock.
    //
    // The fix adds `ch_out_shutdown` so the outer while loop exits, allowing
    // the shutdown section to call q->shutdown() on every BoundedQueue.  That
    // shuts down the semaphore, which unblocks any producer blocked at acquire().
    //
    // Test outline:
    //   - 2 producers each sending 3 messages through a Lineariser.
    //   - A consumer that reads exactly 1 message from ch_out and then calls
    //     cancel_network(), which shuts down ch_out and all other channels.
    //   - Without the fix the network would deadlock; with the fix it completes.

    constexpr std::size_t num_producers = 2;
    constexpr std::size_t messages_per_producer = 3;

    auto ch_out = ctx->create_channel();
    auto lineariser = Lineariser(ctx, ch_out, num_producers);
    auto queues = lineariser.get_queues();

    std::vector<Actor> tasks;

    // Each producer sends messages_per_producer messages in round-robin sequence
    // order, then drains its BoundedQueue.  Having >1 message per producer is
    // what exposes the deadlock: after drain() drops the receipt for message[1]
    // (because ch_out is shut down), the producer tries to acquire() for
    // message[2] and finds the semaphore at 0 → blocks forever without the fix.
    auto make_producer = [](
        std::shared_ptr<Context> ctx,
        std::shared_ptr<BoundedQueue> q,
        std::size_t start,
        std::size_t stride,
        std::size_t end
    ) -> Actor {
        for (auto id = start; id < end; id += stride) {
            auto ticket = co_await q->acquire();
            if (!ticket.has_value()) {
                break;
            }
            co_await ctx->executor()->schedule();
            auto sent = co_await ticket->send(
                Message{id, std::make_unique<std::size_t>(id), ContentDescription{}}
            );
            if (!sent) {
                break;
            }
        }
        co_await q->drain(ctx->executor());
    };

    for (std::size_t i = 0; i < num_producers; i++) {
        tasks.push_back(make_producer(
            ctx, queues[i], i, num_producers, num_producers * messages_per_producer
        ));
    }
    tasks.push_back(lineariser.drain());

    // Consumer: reads exactly 1 message from ch_out, then calls cancel_network()
    // to simulate a downstream failure.  cancel_network() shuts down ch_out (and
    // all other registered channels), which causes drain()'s next ch_out_->send()
    // to return false — the exact condition that triggered the deadlock.
    tasks.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_in) -> Actor {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            // Consume exactly one forwarded message, then tear down the network.
            std::ignore = co_await ch_in->receive();
            ctx->cancel_network();
        }(ctx, ch_out)
    );

    // Must complete without hanging.  cancel_network() does not throw; it just
    // unblocks all blocked actors so the network drains cleanly.
    run_actor_network(std::move(tasks), ctx);
}

TEST_F(StreamingLineariser, ManyProducers) {
    constexpr std::size_t num_producers = 100;
    constexpr std::size_t num_messages = 30'000;

    auto ch_out = ctx->create_channel();
    auto lineariser = Lineariser(ctx, ch_out, num_producers);
    std::vector<Actor> tasks;
    tasks.reserve(num_producers + 2);
    auto make_producer = [end = num_messages, stride = num_producers](
                             std::shared_ptr<Context> ctx,
                             std::shared_ptr<BoundedQueue> ch_out,
                             std::size_t start
                         ) -> Actor {
        for (auto id = start; id < end; id += stride) {
            co_await ctx->executor()->schedule();
            auto ticket = co_await ch_out->acquire();
            if (!ticket.has_value()) {
                break;
            }
            co_await ticket->send(
                Message{id, std::make_unique<std::size_t>(id), ContentDescription{}}
            );
        }
        co_await ch_out->drain(ctx->executor());
    };
    auto queues = lineariser.get_queues();
    EXPECT_EQ(queues.size(), num_producers);
    for (std::size_t i = 0; i < num_producers; i++) {
        tasks.push_back(make_producer(ctx, queues[i], i));
    }
    tasks.push_back(lineariser.drain());
    std::vector<Message> outputs;
    outputs.reserve(num_messages);
    tasks.push_back(actor::pull_from_channel(ctx, ch_out, outputs));
    run_actor_network(std::move(tasks));
    EXPECT_EQ(num_messages, outputs.size());
    for (std::size_t i = 0; i < num_messages; i++) {
        EXPECT_EQ(outputs[i].sequence_number(), i);
        EXPECT_EQ(outputs[i].release<std::size_t>(), i);
    }
}
