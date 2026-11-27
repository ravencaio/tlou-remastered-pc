/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Switches for the game's post-processing effects (see runtime_post_effects.h). */
#include "runtime_post_effects.h"
#include "gpu/gow3gpu.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#define SLOTS_AT 0x1a85e50u
#define SLOT_SIZE 0x1110u

/* A field of render-settings (offsets from tools/tlou_render_settings.py list) and the value
 * that turns its effect off. */
typedef struct {
    uint32_t effect;
    uint16_t offset;
    char kind; /* 'f' float, 'i' 32-bit integer, 'b' byte */
    float value;
} Field;
static const Field fields[] = {
    {RUNTIME_POST_DEPTH_OF_FIELD, 0xe10, 'b', 1},   /* post.dof-disable */
    {RUNTIME_POST_BLOOM, 0xe2c, 'f', 0},            /* post.blend-bloom-layer0..6 */
    {RUNTIME_POST_BLOOM, 0xe30, 'f', 0},
    {RUNTIME_POST_BLOOM, 0xe34, 'f', 0},
    {RUNTIME_POST_BLOOM, 0xe38, 'f', 0},
    {RUNTIME_POST_BLOOM, 0xe3c, 'f', 0},
    {RUNTIME_POST_BLOOM, 0xe40, 'f', 0},
    {RUNTIME_POST_BLOOM, 0xe44, 'f', 0},
    {RUNTIME_POST_LENS_FLARE, 0x634, 'f', 0},       /* sun.sun-flare-enable */
    {RUNTIME_POST_LENS_FLARE, 0xc04, 'f', 0},       /* post.streak-pre-min-scale-bias-exp-1 (scale) */
    {RUNTIME_POST_LIGHT_SHAFTS, 0xd94, 'f', 0},     /* post.light-beam-enable */
    {RUNTIME_POST_VIGNETTE, 0xdd0, 'i', 0},         /* post.vignette-enable */
    {RUNTIME_POST_FILM_GRAIN, 0xdec, 'i', 0},       /* post.film-grain-enable */
    {RUNTIME_POST_FILM_GRAIN, 0xdf4, 'f', 0},       /* post.film-grain-intensity */
    {RUNTIME_POST_COLOR_GRADING, 0xb30, 'f', 0},    /* lut.lut-enable */
};
enum { FIELD_COUNT = sizeof(fields) / sizeof(fields[0]) };

/* Per slot and field: the game's last own value, and whether this code wrote the field last. */
typedef struct {
    uint32_t game[3][FIELD_COUNT];
    uint8_t written[3][FIELD_COUNT];
} State;
_Static_assert(sizeof(State) <= RUNTIME_POST_STATE_SIZE, "state fits");

static uint32_t off_bits(const Field *f) {
    uint32_t bits = 0;
    if (f->kind == 'f') memcpy(&bits, &f->value, 4);
    else bits = (uint32_t)(int32_t)f->value;
    return bits;
}

static uint32_t read_field(const unsigned char *slot, const Field *f) {
    uint32_t bits = 0;
    if (f->kind == 'b') bits = slot[f->offset];
    else memcpy(&bits, slot + f->offset, 4);
    return bits;
}

static void write_field(unsigned char *slot, const Field *f, uint32_t bits) {
    if (f->kind == 'b') slot[f->offset] = (uint8_t)bits;
    else memcpy(slot + f->offset, &bits, 4);
}

void runtime_post_effects_apply(unsigned char *slots[3], uint32_t off, void *state_bytes) {
    State *state = state_bytes;
    for (unsigned s = 0; s < 3; ++s) {
        for (unsigned i = 0; i < FIELD_COUNT; ++i) {
            const Field *f = &fields[i];
            const uint32_t now = read_field(slots[s], f), ours = off_bits(f);
            if (off & f->effect) {
                /* A value other than ours came from the game (or this is the first write). */
                if (!state->written[s][i] || now != ours) state->game[s][i] = now;
                if (now != ours) write_field(slots[s], f, ours);
                state->written[s][i] = 1;
            } else if (state->written[s][i]) {
                /* Switched on again: the game's value back, unless the game wrote a new one. */
                if (now == ours) write_field(slots[s], f, state->game[s][i]);
                state->written[s][i] = 0;
            }
        }
    }
}

static unsigned char *game_image;
static _Alignas(16) unsigned char state[RUNTIME_POST_STATE_SIZE];
static atomic_flag busy = ATOMIC_FLAG_INIT;
static uint32_t last_logged = UINT32_MAX;

void runtime_post_effects_set_image(unsigned char *image) {
    game_image = image;
}

void ABI runtime_post_effects_frame(void) {
    if (!game_image) return;
    /* The updater may run on more than one thread; one apply at a time (a skipped one is
     * redone by the next call, the same frame). */
    if (atomic_flag_test_and_set_explicit(&busy, memory_order_acquire)) return;
    const uint32_t off = (uint32_t)gow3gpu_post_effects_off();
    unsigned char *slots[3] = {game_image + SLOTS_AT, game_image + SLOTS_AT + SLOT_SIZE,
                               game_image + SLOTS_AT + 2 * SLOT_SIZE};
    runtime_post_effects_apply(slots, off, state);
    if (off != last_logged) {
        last_logged = off;
        printf("Post effects: off mask %#x (depth of field %s, bloom %s, lens flare %s, light shafts %s, vignette %s, "
               "film grain %s, color grading %s)\n",
               off, off & RUNTIME_POST_DEPTH_OF_FIELD ? "off" : "on", off & RUNTIME_POST_BLOOM ? "off" : "on",
               off & RUNTIME_POST_LENS_FLARE ? "off" : "on", off & RUNTIME_POST_LIGHT_SHAFTS ? "off" : "on",
               off & RUNTIME_POST_VIGNETTE ? "off" : "on", off & RUNTIME_POST_FILM_GRAIN ? "off" : "on",
               off & RUNTIME_POST_COLOR_GRADING ? "off" : "on");
    }
    atomic_flag_clear_explicit(&busy, memory_order_release);
}
