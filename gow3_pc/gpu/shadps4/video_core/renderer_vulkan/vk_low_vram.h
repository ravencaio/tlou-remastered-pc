// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 14 Nov 2026: low video memory mode, for cards of 2 GB (and 3 GB).
//
// At 1080p The Last of Us held 1.6 GB of video memory at the title screen and 2.0 GB in play on a
// GTX 1050 Ti, of which only ~0.5 GB were images and ~0.3-0.5 GB buffer arenas in use: the rest
// were reserves sized for big cards - VMA blocks of 256 MiB plus a 64 MiB spare kept free in
// them (SpareBlockKeeper), a pinned 128 MiB device-local staging buffer, two 64 MiB arena chunks
// allocated ahead, idle staging blocks kept for 10-60 s - and the texture cache's collector
// evicted only images idle for 80-160 frames under pressure. On a 2 GB card that does not fit:
// the driver pages to system memory (frame rate falls) or an allocation fails (the port
// asserted). In this mode:
//   - VMA blocks of 32 MiB, VMA follows the driver's budget (VK_EXT_memory_budget); no spare
//     image block; the image recycler keeps less.
//   - no pinned staging buffer (GOW3_STAGING_LARGE_MB still sets one); depth/color copies
//     take a buffer of their own size; idle device-local staging goes after ~4 s.
//   - arena memory in 16 MiB chunks, one ahead; arenas give back memory unused for 20 s above
//     a quarter of the card (at least 384 MiB).
//   - the texture collector works from the room the driver's budget leaves for images, evicts
//     images idle for fewer frames the higher the pressure, and runs at once after an
//     allocation went over the budget.
// In every mode an image or buffer that does not fit the budget is retried outside it (the driver
// pages) and a device-local buffer then goes to system memory, instead of an assertion.
//
// GOW3_LOW_VRAM=auto (default): on for discrete cards with 3 GiB or less of video memory;
// =1: on; =0: off. The idea and several of the numbers follow the Uncharted port's small-card
// work (b64fc69, "Keep small cards inside their video memory budget").
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "common/types.h"

namespace Vulkan::LowVram {

namespace Detail {
inline std::atomic<int> state{-1}; ///< -1 undecided, 0 off, 1 on
inline std::atomic<bool> urgent{false};
inline std::atomic<u64> failed{0}, over_budget{0}, host_buffers{0};
} // namespace Detail

/// Decides the mode once, from the largest device-local heap (Instance, before the device and
/// the allocator are made).
inline void Decide(u64 device_local_bytes, bool integrated) {
    if (Detail::state.load(std::memory_order_relaxed) >= 0) {
        return;
    }
    const char* env = std::getenv("GOW3_LOW_VRAM");
    const std::string_view value = env && env[0] ? env : "auto";
    bool on;
    const char* why;
    if (value == "1" || value == "on") {
        on = true;
        why = "GOW3_LOW_VRAM=1";
    } else if (value == "0" || value == "off") {
        on = false;
        why = "GOW3_LOW_VRAM=0";
    } else {
        on = !integrated && device_local_bytes != 0 && device_local_bytes <= (u64{3} << 30) + (64u << 20);
        why = integrated ? "integrated GPU" : on ? "3 GiB or less" : "more than 3 GiB";
    }
    Detail::state.store(on ? 1 : 0, std::memory_order_relaxed);
    std::printf("GPU: low video memory mode %s (%llu MiB of video memory, %s; GOW3_LOW_VRAM=1/0 "
                "forces it)\n",
                on ? "on" : "off", static_cast<unsigned long long>(device_local_bytes >> 20), why);
    std::fflush(stdout);
}

/// Whether the mode is on (false until Decide).
inline bool Enabled() {
    return Detail::state.load(std::memory_order_relaxed) == 1;
}

/// An allocation did not fit the driver's budget: the texture collector's next pass is urgent.
inline void NoteOverBudget() {
    Detail::urgent.store(true, std::memory_order_relaxed);
    Detail::failed.fetch_add(1, std::memory_order_relaxed);
}

/// Takes the urgent request (texture collector).
inline bool TakeUrgent() {
    return Detail::urgent.exchange(false, std::memory_order_relaxed);
}

/// Counts for the 5 s memory report: allocations that went over the budget (driver paging), and
/// device-local buffers placed in system memory instead.
inline u64 OverBudgetAllocations() {
    return Detail::over_budget.load(std::memory_order_relaxed);
}
inline u64 HostBuffers() {
    return Detail::host_buffers.load(std::memory_order_relaxed);
}
inline void CountOverBudget() {
    Detail::over_budget.fetch_add(1, std::memory_order_relaxed);
}
inline void CountHostBuffer() {
    Detail::host_buffers.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Vulkan::LowVram
