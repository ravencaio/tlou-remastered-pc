/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Render settings overrides (see runtime_render_settings.h). */
#include "runtime_render_settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* sid32("render-settings"): CRC-32, MSB first, polynomial 0x04C11DB7, no initial value or xor. */
#define RENDER_SETTINGS_TYPE 0x9ec73b22u

typedef struct {
    uint32_t offset;
    char kind;              /* 'f' float, 'i' 32-bit integer, 'b' byte (a switch) */
    union { float f; int32_t i; uint8_t b; } value;
} Override;

enum { MAX_OVERRIDES = 1024 };
static Override overrides[MAX_OVERRIDES];
static unsigned override_count;
static unsigned structs_seen, modules_seen;

unsigned runtime_render_settings_modules_seen(void) { return __atomic_load_n(&modules_seen, __ATOMIC_RELAXED); }

unsigned runtime_render_settings_load(const char *path) {
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) return 0;
    char line[256];
    unsigned number = 0;
    while (fgets(line, sizeof(line), f)) {
        ++number;
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        unsigned long long offset;
        char kind, text[64];
        if (sscanf(line, "%llx %c %63s", &offset, &kind, text) != 3) continue;
        Override o = {(uint32_t)offset, kind, {0}};
        const unsigned width = kind == 'b' ? 1 : 4;
        char *end = NULL;
        if (kind == 'f') o.value.f = strtof(text, &end);
        else if (kind == 'i') o.value.i = (int32_t)strtol(text, &end, 0);
        else if (kind == 'b') o.value.b = (uint8_t)strtoul(text, &end, 0);
        if (!end || *end || offset + width > RUNTIME_RENDER_SETTINGS_SIZE || (kind != 'b' && offset % 4)) {
            printf("Render settings: %s line %u ignored\n", path, number);
            continue;
        }
        if (override_count == MAX_OVERRIDES) break;
        overrides[override_count++] = o;
    }
    fclose(f);
    return override_count;
}

static void apply(unsigned char *settings) {
    for (unsigned i = 0; i < override_count; ++i) {
        const Override *o = &overrides[i];
        if (o->kind == 'b') settings[o->offset] = o->value.b;
        else memcpy(settings + o->offset, &o->value, 4);
    }
}

void ABI runtime_render_settings_module(const unsigned char *table) {
    if (!table || !override_count) return;
    /* The loader's own path reaches the hook with whatever it read, DC module or not. */
    uint32_t magic, count;
    uint64_t entries;
    memcpy(&magic, table - 16, 4);
    if (magic != 0x44433030u) return;
    memcpy(&count, table + 4, 4);
    memcpy(&entries, table + 8, 8);
    if (!entries || count > 0x10000) return;
    if (__atomic_add_fetch(&modules_seen, 1, __ATOMIC_RELAXED) == 1)
        puts("Render settings: the hook sees the game's DC modules");
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *entry = (const unsigned char *)(uintptr_t)entries + 16 * (uint64_t)i;
        uint32_t name, type;
        uint64_t data;
        memcpy(&name, entry, 4);
        memcpy(&type, entry + 4, 4);
        memcpy(&data, entry + 8, 8);
        if (type != RENDER_SETTINGS_TYPE || !data) continue;
        apply((unsigned char *)(uintptr_t)data);
        /* A few lines only: the handler runs on the game's loading thread. */
        const unsigned seen = __atomic_add_fetch(&structs_seen, 1, __ATOMIC_RELAXED);
        if (seen <= 3 || (seen & (seen - 1)) == 0)
            printf("Render settings: %u overrides written (struct %08x, %u so far)\n", override_count, name, seen);
    }
}
