#include "runtime_menu_exit.h"
#include "gpu/gow3gpu.h"
#include "runtime_menu_pc.h"
#include "runtime_menu_saves.h"
#include "runtime_render_settings.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#define DATA_MAGIC 0x44433030u
#define DATA_MIN 0x1000u
#define DATA_MAX 0x400000u
#define SLOT_SIZE 0x8000u
#define SLOTS_MAX 64

/* The arena is cut in fixed slots, one per address menu data was loaded at (each load takes ~14
 * KiB: the copied page table and lists). A load at an address seen before writes its slot again
 * (the file there was replaced, so nothing points at the old copy any more); a load at a new
 * address takes a free slot, else the one whose data was loaded longest ago (its menu data has
 * long been replaced: the game holds a few menu files at once, far fewer than the slots). Before
 * 21 Nov 2026 every new address took new space and after ~18 loads the arena was full for good. */
typedef struct { uintptr_t base; uint64_t stamp; } Slot;
static unsigned char *arena;
static Slot slots[SLOTS_MAX];
static unsigned slot_count;
static uint64_t load_stamp;
static atomic_flag arena_lock = ATOMIC_FLAG_INIT;
static atomic_int asking, leaving;
static uint32_t label = GOW3_MENU_LABEL_QUIT;

/* "QUIT TO DESKTOP" in the game's languages. The table's language is told by its own "Leave the
 * game?", different in each (Portugal and Brazil included; the two English and the two Spanish
 * tables share one answer). Every character is one the game's text in that language already uses,
 * so its font has it ("desktop" is not among the Chinese ones: "leave the game" there). */
static const struct { const char *leave_game, *label; } desktop_labels[] = {
    {"Leave the game?", "QUIT TO DESKTOP"},
    {"Voulez-vous vraiment quitter ?", "QUITTER VERS LE BUREAU"},
    {"Abbandonare?", "ESCI AL DESKTOP"},
    {"Spiel verlassen?", "ZUM DESKTOP BEENDEN"},
    {"¿Abandonar partida?", "SALIR AL ESCRITORIO"},
    {"¿Abandonar el juego?", "SALIR AL ESCRITORIO"},
    {"Forlad spillet?", "AFSLUT TIL SKRIVEBORDET"},
    {"De certeza que queres sair?", "SAIR PARA O AMBIENTE DE TRABALHO"},
    {"Sair do jogo?", "SAIR PARA A ÁREA DE TRABALHO"},
    {"Spel verlaten?", "AFSLUITEN NAAR BUREAUBLAD"},
    {"ゲームを退出しますか？", "デスクトップに戻る"},
    {"게임에서 나가시겠습니까?", "바탕 화면으로 나가기"},
    {"Haluatko poistua pelistä?", "POISTU TYÖPÖYDÄLLE"},
    {"Vil du gå ut av spillet?", "AVSLUTT TIL SKRIVEBORDET"},
    {"Lämna spelet?", "AVSLUTA TILL SKRIVBORDET"},
    {"確定要離開遊戲嗎？", "離開遊戲"},
    {"Вы уверены, что хотите выйти?",
     "ВЫЙТИ НА РАБОЧИЙ СТОЛ"},
    {"Wyjść z gry?", "WYJDŹ DO PULPITU"},
    {"Έξοδος από το παιχνίδι;",
     "ΕΞΟΔΟΣ ΣΤΗΝ ΕΠΙΦΑΝΕΙΑ ΕΡΓΑΣΙΑΣ"},
    {"Odejít ze hry?", "UKONČIT DO PLOCHY"},
    {"Oyundan çıkılsın mı?", "MASAÜSTÜNE ÇIK"},
};
static Gow3TextLookup text_original;

void runtime_menu_exit_set_text(void *original) {
    text_original = (Gow3TextLookup)original;
    label = GOW3_MENU_LABEL_EXIT_DESKTOP;
}

/* text(table, sid) for sid in [TEXT_FIRST, TEXT_LAST], from the lookup's hook: our labels. */
ABI const char *runtime_menu_exit_text(void *table, uint32_t sid) {
    if (sid != GOW3_MENU_LABEL_EXIT_DESKTOP) {
        const char *text = runtime_menu_pc_text(text_original, table, sid);
        return text ? text : text_original(table, sid);
    }
    const char *leave = text_original(table, GOW3_MENU_TEXT_LEAVE_GAME);
    if (leave)
        for (size_t i = 0; i < sizeof(desktop_labels) / sizeof(desktop_labels[0]); ++i)
            if (!strcmp(leave, desktop_labels[i].leave_game)) return desktop_labels[i].label;
    return desktop_labels[0].label;
}

void runtime_menu_exit_set_arena(unsigned char *memory, size_t size) {
    arena = memory;
    slot_count = (unsigned)(size / SLOT_SIZE);
    if (slot_count > SLOTS_MAX) slot_count = SLOTS_MAX;
}

/* The slot of menu data at base: its own, a free one, else the least recently loaded. */
static unsigned slot_for(uintptr_t base, int *recycled) {
    unsigned pick = 0;
    *recycled = 0;
    for (unsigned i = 0; i < slot_count; ++i)
        if (slots[i].base == base) return i;
    for (unsigned i = 0; i < slot_count; ++i) {
        if (!slots[i].base) return i;
        if (slots[i].stamp < slots[pick].stamp) pick = i;
    }
    *recycled = 1;
    return pick;
}

/* The exit button, then the PC settings page (only with our labels: the text hook in). */
static int patch(unsigned char *data, uint64_t size, unsigned char *space, size_t space_size, size_t *used,
                 Gow3MenuPatched *exit_done, Gow3MenuPcPatched *pc_done) {
    *exit_done = gow3_menu_patch(data, size, space, space_size, used, label);
    *pc_done = (Gow3MenuPcPatched){0, 0, 0};
    unsigned rows_count = 0;
    const Gow3PcRow *rows = text_original ? runtime_menu_pc_rows(&rows_count) : NULL;
    if (rows) *pc_done = gow3_menu_pc_patch(data, size, space, space_size, used, rows, rows_count);
    return exit_done->lobby || exit_done->pause || pc_done->page || pc_done->button;
}

ABI void runtime_menu_exit_data_loaded(unsigned char *data) {
    /* The render settings overrides share this hook: it is on the DC module loader's path. */
    if (data) runtime_render_settings_module(data + 0x10);
    if (!data || !arena || gow3_menu_u32(data) != DATA_MAGIC) return;
    const uint64_t size = gow3_menu_u32(data + 8);
    if (size < DATA_MIN || size > DATA_MAX) return;
    while (atomic_flag_test_and_set(&arena_lock)) {}
    Gow3MenuPatched done = {0, 0};
    Gow3MenuPcPatched pc = {0, 0, 0};
    int recycled = 0;
    size_t used = 0;
    const unsigned slot = slot_count ? slot_for((uintptr_t)data, &recycled) : 0;
    /* A slot is taken only by data that got something: menu files without our lists leave it. */
    const int changed = slot_count && patch(data, size, arena + (size_t)slot * SLOT_SIZE, SLOT_SIZE, &used, &done, &pc);
    if (changed) slots[slot] = (Slot){(uintptr_t)data, ++load_stamp};
    else recycled = 0;
    atomic_flag_clear(&arena_lock);
    if (recycled) printf("Menu: menu data at a new address, slot %u reused\n", slot);
    if (done.lobby || done.pause) printf("Menu: QUIT button added (lobby lists %u, pause lists %u)\n", done.lobby, done.pause);
    if (pc.page || pc.button)
        printf("Menu: PC settings added (page %u, Options lists %u, save list page %u; slot %u, %zu of %u bytes)\n",
               pc.page, pc.button, pc.saves, slot, used, SLOT_SIZE);
    if (pc.saves) runtime_menu_saves_page_added();
}

static void leave(void) {
    if (atomic_exchange(&leaving, 1)) return;
    puts("Menu: quit to desktop");
    gow3gpu_request_exit();
}

typedef ABI void (*ConfirmOpen)(void *menu, uint32_t title, uint32_t yes_label, uint32_t yes_button,
                                 uint32_t no_label, uint32_t no_button, uint32_t extra);
static ConfirmOpen confirm_open;

void runtime_menu_exit_set_confirm(void *open) {
    confirm_open = (ConfirmOpen)open;
    runtime_menu_saves_set_confirm(open);
}

/* Runs on the menu's thread, at the dispatcher's entry: the game's own opener is safe to call. */
ABI void runtime_menu_exit_button(void *menu, uint32_t hash) {
    runtime_menu_saves_note_button(hash);
    if (runtime_menu_saves_button(menu, hash) || runtime_menu_pc_button(hash)) return;
    if (hash == GOW3_MENU_EXIT_LOBBY || hash == GOW3_MENU_EXIT_CONFIRMED) {
        leave();
    } else if (hash == GOW3_MENU_EXIT_PAUSE && confirm_open && menu) {
        confirm_open(menu, GOW3_MENU_TEXT_LEAVE_GAME, GOW3_MENU_TEXT_YES, GOW3_MENU_EXIT_CONFIRMED,
                     GOW3_MENU_TEXT_NO, GOW3_MENU_CONFIRM_NO, 0);
    } else if (hash == GOW3_MENU_EXIT_PAUSE && !atomic_load(&asking)) {
        static const char *const items[] = {"Cancel", "Quit to desktop"};
        if (gow3gpu_choice_begin("Quit to desktop? Unsaved progress is lost.", items, NULL, 2, 0)) atomic_store(&asking, 1);
    }
}

void runtime_menu_exit_tick(void) {
    runtime_menu_pc_tick();
    if (!atomic_load(&asking)) return;
    const int choice = gow3gpu_choice_poll();
    if (choice == -1) return;
    atomic_store(&asking, 0);
    if (choice == 1) leave();
}
