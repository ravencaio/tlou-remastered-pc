// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

// gow3 26 Nov 2026: the game's loading screens. A world load goes forward a fixed amount per
// frame of the loading screen (Continue into world-suburbs: about 5,500 frames, sometimes about
// 2,000), while the process is ~88% idle: at the 60 FPS limit it took 91 s, at 120 46 s, at
// ~220 27 s (replays with the files cached; the hard disk is not what holds it up). While the
// loading screen is up the present thread lifts the frame limit.
//
// Start: the game's own TTY line "----- Switching world: from ..." (runtime_file.c, guest writes
// to stdout). End: the first occlusion query of a game frame (the title menu and gameplay draw
// one per frame; the loading screen none), or a movie, or 300 s.
// GOW3_LOADING_UNCAP=0 (launcher debug menu): the limit stays as set.
// GOW3_LOADING_FPS=<n>: limit during loading screens (default 0 = as fast as the vblank).
namespace Gow3Loading {

bool Enabled();
/// Whether a loading screen is up at `now_ns` (steady clock); ends it on its end conditions.
bool Active(int64_t now_ns);
/// The frame limit while active (0 = none).
uint32_t FrameLimit();
/// A game frame drew its occlusion query (vk_rasterizer.cpp OcclusionQueryEvent).
void NoteOcclusionQuery();
/// A loading screen's frame was presented (statistics only).
void NoteFrame();

} // namespace Gow3Loading

extern "C" void gow3gpu_loading_screen_begin(void);
