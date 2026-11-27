/* "PC SETTINGS" in the game's Options page (src/menu_pc_hook.h): the page's rows (the game's own
 * toggles and sliders), their labels in the game's language, their values. GOW3_PC_MENU=0 leaves
 * it out. */
#ifndef GOW3_RUNTIME_MENU_PC_H
#define GOW3_RUNTIME_MENU_PC_H
#include <stdint.h>
#include "menu_pc_hook.h"
#include "runtime.h"

typedef ABI const char *(*Gow3TextLookup)(void *table, uint32_t sid);

/* The rows to add (NULL when GOW3_PC_MENU=0). */
const Gow3PcRow *runtime_menu_pc_rows(unsigned *count);
/* The text of one of the page's ids (button, title, row names), or NULL; lookup is the game's
 * own, for its language. */
const char *runtime_menu_pc_text(Gow3TextLookup lookup, void *table, uint32_t sid);
/* A button press: 1 when it was one of the page's rows (MORE PC SETTINGS opens the overlay). */
int runtime_menu_pc_button(uint32_t hash);
/* The binding factory (menu_pc_hook.h), and the helper its stub calls for our option sids. */
void runtime_menu_pc_set_factory(void *original);
ABI void *runtime_menu_pc_binding(void *menu, uint32_t kind, uint32_t a, uint32_t sid, uint32_t b, uint32_t c,
                                  uint32_t d);
/* "New save" in the language of the last text table seen. */
const char *runtime_menu_pc_new_save_text(void);
/* A text of the game in the language of the last text table seen (NULL before one or if none). */
const char *runtime_menu_pc_game_text(uint32_t sid);
/* The page scrolls (26 Nov 2026: its rows run past the footer). The game lays out a page every
 * frame (update 1.11: 0x2F7390); at 0x2F8026 it has the list's origin (x in r12d, y in r13d, float
 * bits) and is about to add it to each row's position ({x, y} at row + 0xa0) before the rows place
 * themselves. A hook there calls this with the page object ({+0x10 hash, +0xac row count, +0xb0
 * rows}; a row: +0x8 focused, +0x90 {x, y, width, height} as last placed) and the y; it returns the
 * y to use. On our page it keeps the focused row inside RUNTIME_MENU_PC_VIEW units below the
 * origin (the screen is 1280x720 units; the footer starts near 640) and moves the rows outside
 * that window off the screen; other pages get their y back unchanged. */
enum { RUNTIME_MENU_PC_VIEW = 500 };
ABI uint32_t runtime_menu_pc_scroll(unsigned char *page, uint32_t y_bits);
/* Same for tests: the rows' heights in order, the focused row, the current scroll; the new scroll,
 * and visible[i] for each row. */
float runtime_menu_pc_scroll_for(const float *heights, unsigned count, int focused, float scroll, unsigned char *visible);
/* From the pad's sampling: values changed in the page are applied and saved. */
void runtime_menu_pc_tick(void);
#endif
