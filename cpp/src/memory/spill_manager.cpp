/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <optional>
#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {


SpillManager::SpillManager(
    BufferResource* br, std::optional<Duration> periodic_spill_check
)
    : br_{br} {
    if (periodic_spill_check.has_value()) {
        periodic_spill_thread_.emplace(
            [this]() { spill_to_make_headroom(0); }, *periodic_spill_check
        );
    }
}

SpillManager::~SpillManager() {
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->stop();
    }
}

std::size_t SpillManager::registry_index(MemoryType mem_type) noexcept {
    return static_cast<std::size_t>(mem_type);
}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function, int priority, MemoryType mem_type
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const id = spill_function_id_counter_++;
    auto const idx = registry_index(mem_type);
    RAPIDSMPF_EXPECTS(
        spill_functions_[idx].insert({id, std::move(spill_function)}).second,
        "corrupted id counter"
    );
    spill_function_priorities_[idx].insert({priority, id});

    // Make sure the spill thread is running.
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->resume();
    }
    return id;
}

void SpillManager::remove_spill_function(SpillFunctionID fid) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool any_left = false;
    for (std::size_t idx = 0; idx < MEMORY_TYPES.size(); ++idx) {
        auto& prio = spill_function_priorities_[idx];
        for (auto it = prio.begin(); it != prio.end(); ++it) {
            if (it->second == fid) {
                prio.erase(it);  // Erase the first occurrence
                break;  // Exit after erasing to ensure only the first one is removed
            }
        }
        spill_functions_[idx].erase(fid);
        any_left = any_left || !spill_functions_[idx].empty();
    }

    // Asynchronously pause the spill thread if no spill functions are left.
    if (periodic_spill_thread_.has_value() && !any_left) {
        periodic_spill_thread_->pause_nb();
    }
}

std::size_t SpillManager::spill_unsafe(std::size_t amount, MemoryType mem_type) {
    auto const idx = registry_index(mem_type);
    std::size_t spilled{0};
    for (auto const [_, fid] : spill_function_priorities_[idx]) {
        if (spilled >= amount) {
            break;
        }
        spilled += spill_functions_[idx].at(fid)(amount - spilled);
    }
    return spilled;
}

std::size_t SpillManager::spill_to_make_headroom_unsafe(
    std::int64_t headroom, MemoryType mem_type
) {
    std::int64_t const available = br_->memory_available_for_reservation(mem_type);
    if (headroom <= available) {
        return 0;
    }
    return spill_unsafe(safe_cast<std::size_t>(headroom - available), mem_type);
}

std::size_t SpillManager::spill(std::size_t amount, MemoryType mem_type) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_unsafe(amount, mem_type);
}

std::size_t SpillManager::spill_to_make_headroom(
    std::int64_t headroom, MemoryType mem_type
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_to_make_headroom_unsafe(headroom, mem_type);
}

std::optional<std::size_t> SpillManager::try_spill_to_make_headroom(
    std::int64_t headroom, MemoryType mem_type
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return std::nullopt;
    }
    return spill_to_make_headroom_unsafe(headroom, mem_type);
}

}  // namespace rapidsmpf
