/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Render settings overrides (src/runtime_render_settings.c): the file's lines and the handler on a
 * relocated DC module table with two render-settings structs and one of another type. */
#include "runtime_render_settings.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static unsigned char a[RUNTIME_RENDER_SETTINGS_SIZE], b[RUNTIME_RENDER_SETTINGS_SIZE], other[16];

static void entry(unsigned char *at, uint32_t name, uint32_t type, void *data) {
    const uint64_t pointer = (uint64_t)(uintptr_t)data;
    memcpy(at, &name, 4);
    memcpy(at + 4, &type, 4);
    memcpy(at + 8, &pointer, 8);
}

int main(int argc, char **argv) {
    assert(argc > 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/render-settings-test.txt", argv[1]);
    FILE *f = fopen(path, "w");
    assert(f);
    fputs("# comment\n"
          "0x1038 f 20000 misc.far-cull-distance\n"
          "0x0217 b 1 lighting.ssao-enable\n"
          "0x0ad0 i -3 tonemap.tonemap-exposure-method\n"
          "0x1110 f 1 past-the-end\n"       /* ignored: outside the struct */
          "0x0002 f 1 misaligned\n"         /* ignored: a float off its 4-byte slot */
          "0x0010 f 1x bad-number\n"        /* ignored */
          "\n", f);
    fclose(f);
    assert(runtime_render_settings_load(path) == 3);
    assert(runtime_render_settings_load(NULL) == 0);  /* no file */
    runtime_render_settings_module(NULL);

    unsigned char entries[3 * 16];
    entry(entries, 0x1111, 0x9ec73b22, a);
    entry(entries + 16, 0x2222, 0x12345678, other);
    entry(entries + 32, 0x3333, 0x9ec73b22, b);
    unsigned char module[32] = {0};             /* "00CD" header, then the table at + 0x10 */
    unsigned char *table = module + 16;
    const uint32_t count = 3;
    const uint64_t pointer = (uint64_t)(uintptr_t)entries;
    memcpy(table + 4, &count, 4);
    memcpy(table + 8, &pointer, 8);
    runtime_render_settings_module(table);      /* not a DC module: nothing written */
    assert(a[0x217] == 0 && b[0x217] == 0 && runtime_render_settings_modules_seen() == 0);
    memcpy(module, "00CD", 4);
    runtime_render_settings_module(table);
    assert(runtime_render_settings_modules_seen() == 1);

    for (unsigned char *s = a; s; s = s == a ? b : NULL) {
        float far;
        int32_t method;
        memcpy(&far, s + 0x1038, 4);
        memcpy(&method, s + 0xad0, 4);
        assert(far == 20000.0f && s[0x217] == 1 && method == -3);
        assert(s[0x216] == 0 && s[0x218] == 0 && s[0x10] == 0);  /* nothing else written */
    }
    for (unsigned i = 0; i < sizeof(other); ++i) assert(other[i] == 0);
    remove(path);
    puts("render settings: ok");
    return 0;
}
