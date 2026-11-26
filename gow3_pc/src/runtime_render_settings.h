/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Render settings overrides, The Last of Us Remastered 1.11 (10 Nov 2026). The game's render
 * settings are the DC struct "render-settings" (0x1110 bytes): a default and one per level area
 * and cutscene, in its bin/dc1 modules; the one in use is copied into the current settings and the
 * renderer's working copies are animated toward it. tools/tlou_render_settings.py turns setting
 * names into a file of (offset, type, value); the loader (probe.c, install_render_settings) puts a
 * jump at the exit of the game's DC module relocation (0xBC69F0, after it has turned the module's
 * offsets into pointers) to a stub that calls runtime_render_settings_module with the module's
 * table, and the handler writes the overrides into every render-settings struct of that module:
 * the values hold in every area and cutscene, and through the game's transitions between them. */
#pragma once
#include <stdint.h>
#include "runtime.h"

enum { RUNTIME_RENDER_SETTINGS_SIZE = 0x1110 };

/* Reads the overrides file (lines "<hex offset> <f|i|b> <value> [name]", '#' comments); the number
 * of overrides, 0 when the file has none or cannot be read. */
unsigned runtime_render_settings_load(const char *path);
/* The hooks' handler, on the game's loading thread: table is the relocated module + 0x10,
 * {u32, u32 entry count, entries} ({u32 name id, u32 type id, pointer} each), or NULL; nothing is
 * written unless the module starts with "00CD". */
void ABI runtime_render_settings_module(const unsigned char *table);
/* DC modules the handler has seen (a check that the hooks are on the game's path). */
unsigned runtime_render_settings_modules_seen(void);
