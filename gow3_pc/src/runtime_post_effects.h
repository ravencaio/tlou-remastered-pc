/* SPDX-License-Identifier: GPL-2.0-or-later */
/* The Last of Us Remastered 1.11: switches for the game's post-processing effects (26 Nov 2026),
 * live from the launcher (gow3.ini post_effects_off) and Options > PC SETTINGS.
 *
 * Reverse engineered: the renderer reads its settings from three render-settings structs (0x1110
 * bytes each) at 0x1A85E50; slot 2 (0x1A88070) is the target the area's settings are copied into
 * (0xBCE8E0, 0xBCFEA0, 0xBCFEF0, 0xBCFF30), and slot 0, the one in use, is animated every frame
 * toward the area and region settings by the game's animator manager (0x1A7F8A0, updated by
 * 0xBCCC30: each animator's update function writes its field from its own source). A value written
 * into slot 0 from outside therefore lasts one frame. The loader (probe.c, install_post_effects)
 * puts a jump at the updater's exit (0xBCD17D, `mov byte [r10 + 0x6590], 0`, which every path
 * reaches) to a stub that calls runtime_post_effects_frame: each effect switched off gets its
 * fields' "off" values in the three slots, right after the animators and before the frame is
 * drawn. The game's own values are kept, and given back when the effect is switched on again
 * (the animators of fields that have reached their target no longer write them).
 *
 * The motion blur is not in here: the GPU side clears the velocity its two shaders read
 * (vk_rasterizer.cpp, GOW3_NO_MOTION_BLUR), which also follows the switch live. */
#pragma once
#include <stdint.h>
#include "runtime.h"

/* Bits of the switch mask (gow3gpu_post_effects_off): set = the effect is off. */
enum {
    RUNTIME_POST_MOTION_BLUR = 1u << 0,
    RUNTIME_POST_DEPTH_OF_FIELD = 1u << 1,
    RUNTIME_POST_BLOOM = 1u << 2,
    RUNTIME_POST_LENS_FLARE = 1u << 3,
    RUNTIME_POST_LIGHT_SHAFTS = 1u << 4,
    RUNTIME_POST_VIGNETTE = 1u << 5,
    RUNTIME_POST_FILM_GRAIN = 1u << 6,
    RUNTIME_POST_COLOR_GRADING = 1u << 7,
};

/* The image the game runs from (the slots' addresses). */
void runtime_post_effects_set_image(unsigned char *image);
/* The hook's handler, on the game's thread after its animators ran. */
void ABI runtime_post_effects_frame(void);
/* Applies a mask to three slots (tests): off bits get the off values, the others the kept
 * values where this code wrote last. state: RUNTIME_POST_STATE_SIZE bytes, zeroed at first. */
enum { RUNTIME_POST_STATE_SIZE = 4096 };
void runtime_post_effects_apply(unsigned char *slots[3], uint32_t off, void *state);
