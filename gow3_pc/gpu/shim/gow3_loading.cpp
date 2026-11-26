// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 26 Nov 2026: the frame limit lifted during the game's loading screens (gow3_loading.h).

#include "gow3_loading.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Gow3Loading {
namespace {

std::atomic<int64_t> started_ns{0};        ///< the loading screen's start, 0: none
std::atomic<int64_t> last_occlusion_ns{0}; ///< the last game frame with an occlusion query
std::atomic<uint32_t> frames{0};

/// Queries recorded this long after the start can still belong to the frame before the load
/// (the draw pipe records a frame late).
constexpr int64_t Grace = 500'000'000;
/// A load that never shows a game frame again (an unknown screen) ends anyway.
constexpr int64_t Longest = 300'000'000'000;

int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void End(int64_t since, int64_t now, const char* why) {
    if (started_ns.compare_exchange_strong(since, 0)) {
        std::printf("Loading screen: %.1f s, %u frames with the frame limit lifted; ended by %s\n",
                    (now - since) / 1e9, frames.exchange(0), why);
        std::fflush(stdout);
    }
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_LOADING_UNCAP");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

uint32_t FrameLimit() {
    static const uint32_t limit = [] {
        const char* env = std::getenv("GOW3_LOADING_FPS");
        const long value = env && *env ? std::strtol(env, nullptr, 10) : 0;
        return value > 0 ? uint32_t(value) : 0u;
    }();
    return limit;
}

bool Active(int64_t now_ns) {
    const int64_t since = started_ns.load(std::memory_order_acquire);
    if (!since) {
        return false;
    }
    if (last_occlusion_ns.load(std::memory_order_relaxed) > since + Grace) {
        End(since, now_ns, "a game frame");
        return false;
    }
    if (now_ns - since > Longest) {
        End(since, now_ns, "the 300 s limit");
        return false;
    }
    return true;
}

void NoteOcclusionQuery() {
    if (started_ns.load(std::memory_order_relaxed)) {
        last_occlusion_ns.store(Now(), std::memory_order_relaxed);
    }
}

void NoteFrame() {
    if (started_ns.load(std::memory_order_relaxed)) {
        frames.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace Gow3Loading

extern "C" void gow3gpu_loading_screen_begin(void) {
    using namespace Gow3Loading;
    if (!Enabled()) {
        return;
    }
    int64_t none = 0;
    const int64_t now = Now();
    if (started_ns.compare_exchange_strong(none, now)) {
        frames.store(0, std::memory_order_relaxed);
        std::printf("Loading screen: world switch, frame limit lifted to %s until the next game "
                    "frame (GOW3_LOADING_UNCAP=0: kept)\n",
                    FrameLimit() ? std::to_string(FrameLimit()).c_str() : "the vblank");
        std::fflush(stdout);
    }
}
