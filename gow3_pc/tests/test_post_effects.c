/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Post-processing switches (src/runtime_post_effects.c): off values written into the three
 * render-settings slots, the game's own values given back when an effect is switched on again,
 * and values the game writes meanwhile kept. */
#include "runtime_post_effects.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* The runtime's frame handler reads the mask from the GPU library; not called here. */
unsigned gow3gpu_post_effects_off(void) { return 0; }

static unsigned char slot[3][0x1110];
static unsigned char state[RUNTIME_POST_STATE_SIZE];

static float getf(int s, unsigned offset) {
    float v;
    memcpy(&v, slot[s] + offset, 4);
    return v;
}
static void setf(int s, unsigned offset, float v) { memcpy(slot[s] + offset, &v, 4); }
static int geti(int s, unsigned offset) {
    int v;
    memcpy(&v, slot[s] + offset, 4);
    return v;
}
static void seti(int s, unsigned offset, int v) { memcpy(slot[s] + offset, &v, 4); }

int main(void) {
    unsigned char *slots[3] = {slot[0], slot[1], slot[2]};
    for (int s = 0; s < 3; ++s) {
        setf(s, 0xe2c, 0.1f + s);   /* bloom layer 0 */
        seti(s, 0xdec, 1);          /* film grain */
        slot[s][0xe10] = 0;         /* depth of field on */
        setf(s, 0xd94, 1.0f);       /* light shafts */
    }
    /* Nothing off: nothing written. */
    runtime_post_effects_apply(slots, 0, state);
    assert(getf(0, 0xe2c) == 0.1f && geti(1, 0xdec) == 1);

    const unsigned off = RUNTIME_POST_BLOOM | RUNTIME_POST_FILM_GRAIN | RUNTIME_POST_DEPTH_OF_FIELD;
    runtime_post_effects_apply(slots, off, state);
    for (int s = 0; s < 3; ++s) {
        assert(getf(s, 0xe2c) == 0.0f && getf(s, 0xe44) == 0.0f);
        assert(geti(s, 0xdec) == 0 && slot[s][0xe10] == 1);
        assert(getf(s, 0xd94) == 1.0f);   /* light shafts left on */
    }
    /* The game's animators write slot 0 again; the next frame turns it off again. */
    setf(0, 0xe2c, 0.5f);
    seti(0, 0xdec, 1);
    runtime_post_effects_apply(slots, off, state);
    assert(getf(0, 0xe2c) == 0.0f && geti(0, 0xdec) == 0);

    /* Bloom on again: slot 0 gets the game's last value, the others their own. Film grain stays off. */
    runtime_post_effects_apply(slots, off & ~RUNTIME_POST_BLOOM, state);
    assert(getf(0, 0xe2c) == 0.5f && getf(1, 0xe2c) == 1.1f && getf(2, 0xe2c) == 2.1f);
    assert(geti(0, 0xdec) == 0);
    /* A value the game wrote while the effect was being switched on stays. */
    setf(0, 0xd94, 0.25f);
    runtime_post_effects_apply(slots, off | RUNTIME_POST_LIGHT_SHAFTS, state);
    assert(getf(0, 0xd94) == 0.0f);
    setf(0, 0xd94, 0.75f);   /* the game, before the switch goes back on */
    runtime_post_effects_apply(slots, 0, state);
    assert(getf(0, 0xd94) == 0.75f && slot[0][0xe10] == 0 && geti(2, 0xdec) == 1);
    puts("post effects: ok");
    return 0;
}
