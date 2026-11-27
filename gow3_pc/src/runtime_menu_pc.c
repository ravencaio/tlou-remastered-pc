#include "runtime_menu_pc.h"
#include "gpu/gow3gpu.h"
#include "runtime_menu_saves.h"
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Texts the game has in every language: the headers (its own Options' "DISPLAY" and
 * "CONTROLS") and "Leave the game?" (tells the table's language, as in runtime_menu_exit.c). */
#define TEXT_MENU_VIDEO 0x53126e93u
#define TEXT_MENU_CONTROLS 0xecfffd59u
#define TEXT_MENU_OFF 0x32066ec9u
#define MORE_SETTINGS (-1)

/* Our texts in the game's 21 languages, told apart by the table's own "Leave the game?" (the
 * two English and the two Spanish tables share one; Spain's and Latin America's differ). Every
 * character is one that language's text in the game already uses, so its font has it (checked
 * against text1/<N>.common of update 1.11: the Japanese has no 垂, the Chinese no 滑 or 齒). The
 * game cuts a name wider than about 800 pixels at 1080p (Cyrillic and Greek capitals are about
 * 22 pixels): the longest Russian and Greek names are shortened. */
enum { TEXT_TITLE, TEXT_DISPLAY_MODE, TEXT_VSYNC, TEXT_FPS_LIMIT, TEXT_ANTI_ALIASING, TEXT_SHARPENING,
       TEXT_PERF_OVERLAY, TEXT_MOUSE_SENS, TEXT_MOUSE_AIM_SENS, TEXT_MOUSE_SCOPE_ON, TEXT_MOUSE_SCOPE_SENS, TEXT_MOUSE_VERTICAL, TEXT_MOUSE_INVERT,
       TEXT_VIBRATION, TEXT_MORE, TEXT_WINDOWED, TEXT_BORDERLESS, TEXT_FULLSCREEN, TEXT_NEW_SAVE, TEXT_NO_SAVE, TEXT_OVERWRITE, TEXT_ANISOTROPIC, TEXT_FIRE_SHADOWS, TEXT_COUNT, TEXT_NONE = -1 };
static const struct { const char *leave_game, *texts[TEXT_COUNT]; } languages[] = {
    {"Leave the game?", {"PC SETTINGS", "DISPLAY MODE", "VSYNC", "FRAME RATE LIMIT", "ANTI-ALIASING", "SHARPENING STRENGTH", "PERFORMANCE OVERLAY", "MOUSE SENSITIVITY", "MOUSE AIM SENSITIVITY", "SEPARATE SCOPE SENSITIVITY", "MOUSE SCOPE SENSITIVITY", "MOUSE UP/DOWN SPEED", "INVERT MOUSE UP/DOWN", "CONTROLLER VIBRATION", "MORE PC SETTINGS...", "Windowed", "Borderless", "Fullscreen", "New save", "CONTINUE WITHOUT SAVING", "Overwrite this save?", "ANISOTROPIC FILTERING", "ENABLE EXTRA SHADOWS"}},
    {"Voulez-vous vraiment quitter ?", {"PARAMÈTRES PC", "MODE D'AFFICHAGE", "SYNCHRO VERTICALE", "LIMITE DE FPS", "ANTICRÉNELAGE", "INTENSITÉ DE LA NETTETÉ", "AFFICHAGE DES PERFORMANCES", "SENSIBILITÉ DE LA SOURIS", "SENSIBILITÉ DE VISÉE (SOURIS)", "SENSIBILITÉ DE LUNETTE SÉPARÉE", "SENSIBILITÉ AVEC LUNETTE (SOURIS)", "VITESSE VERTICALE DE LA SOURIS", "INVERSER LA SOURIS (VERTICAL)", "VIBRATIONS DE LA MANETTE", "PLUS DE PARAMÈTRES PC...", "Fenêtré", "Sans bordure", "Plein écran", "Nouvelle sauvegarde", "CONTINUER SANS SAUVEGARDER", "Écraser cette sauvegarde ?", "FILTRAGE ANISOTROPE", "ACTIVER LES OMBRES SUPPLÉMENTAIRES"}},
    {"Abbandonare?", {"IMPOSTAZIONI PC", "MODALITÀ SCHERMO", "VSYNC", "LIMITE FPS", "ANTI-ALIASING", "INTENSITÀ NITIDEZZA", "OVERLAY PRESTAZIONI", "SENSIBILITÀ MOUSE", "SENSIBILITÀ MIRA (MOUSE)", "SENSIBILITÀ MIRINO SEPARATA", "SENSIBILITÀ MIRINO (MOUSE)", "VELOCITÀ VERTICALE MOUSE", "INVERTI MOUSE (VERTICALE)", "VIBRAZIONE CONTROLLER", "ALTRE IMPOSTAZIONI PC...", "Finestra", "Senza bordi", "Schermo intero", "Nuovo salvataggio", "CONTINUA SENZA SALVARE", "Sovrascrivere questo salvataggio?", "FILTRAGGIO ANISOTROPICO", "ATTIVA OMBRE AGGIUNTIVE"}},
    {"Spiel verlassen?", {"PC-EINSTELLUNGEN", "ANZEIGEMODUS", "VSYNC", "BILDRATENLIMIT", "KANTENGLÄTTUNG", "SCHÄRFESTÄRKE", "LEISTUNGSANZEIGE", "MAUSEMPFINDLICHKEIT", "MAUSEMPFINDLICHKEIT BEIM ZIELEN", "EIGENE ZIELFERNROHR-EMPFINDLICHKEIT", "MAUSEMPFINDLICHKEIT MIT ZIELFERNROHR", "VERTIKALE MAUSGESCHWINDIGKEIT", "MAUS VERTIKAL INVERTIEREN", "CONTROLLER-VIBRATION", "WEITERE PC-EINSTELLUNGEN...", "Fenster", "Rahmenlos", "Vollbild", "Neuer Spielstand", "OHNE SPEICHERN FORTFAHREN", "Diesen Spielstand überschreiben?", "ANISOTROPE FILTERUNG", "ZUSÄTZLICHE SCHATTEN AKTIVIEREN"}},
    {"¿Abandonar partida?", {"AJUSTES DE PC", "MODO DE PANTALLA", "VSYNC", "LÍMITE DE FPS", "ANTIALIASING", "INTENSIDAD DE NITIDEZ", "INDICADOR DE RENDIMIENTO", "SENSIBILIDAD DEL RATÓN", "SENSIBILIDAD AL APUNTAR (RATÓN)", "SENSIBILIDAD PROPIA CON VISOR", "SENSIBILIDAD CON VISOR (RATÓN)", "VELOCIDAD VERTICAL DEL RATÓN", "INVERTIR RATÓN (VERTICAL)", "VIBRACIÓN DEL MANDO", "MÁS AJUSTES DE PC...", "Ventana", "Sin bordes", "Pantalla completa", "Nueva partida guardada", "CONTINUAR SIN GUARDAR", "¿Sobrescribir esta partida guardada?", "FILTRADO ANISOTRÓPICO", "ACTIVAR SOMBRAS ADICIONALES"}},
    {"Leave the game?", {"PC SETTINGS", "DISPLAY MODE", "VSYNC", "FRAME RATE LIMIT", "ANTI-ALIASING", "SHARPENING STRENGTH", "PERFORMANCE OVERLAY", "MOUSE SENSITIVITY", "MOUSE AIM SENSITIVITY", "SEPARATE SCOPE SENSITIVITY", "MOUSE SCOPE SENSITIVITY", "MOUSE UP/DOWN SPEED", "INVERT MOUSE UP/DOWN", "CONTROLLER VIBRATION", "MORE PC SETTINGS...", "Windowed", "Borderless", "Fullscreen", "New save", "CONTINUE WITHOUT SAVING", "Overwrite this save?", "ANISOTROPIC FILTERING", "ENABLE EXTRA SHADOWS"}},
    {"Forlad spillet?", {"PC-INDSTILLINGER", "VISNINGSTILSTAND", "VSYNC", "BILLEDHASTIGHEDSGRÆNSE", "KANTUDJÆVNING", "SKARPHEDSSTYRKE", "YDEEVNEVISNING", "MUSEFØLSOMHED", "MUSEFØLSOMHED VED SIGTE", "SEPARAT KIKKERTSIGTE-FØLSOMHED", "MUSEFØLSOMHED MED KIKKERTSIGTE", "LODRET MUSEHASTIGHED", "INVERTER MUS (LODRET)", "CONTROLLERVIBRATION", "FLERE PC-INDSTILLINGER...", "Vindue", "Uden kant", "Fuld skærm", "Nyt gemt spil", "FORTSÆT UDEN AT GEMME", "Overskriv dette gemte spil?", "ANISOTROPISK FILTRERING", "AKTIVÉR EKSTRA SKYGGER"}},
    {"De certeza que queres sair?", {"DEFINIÇÕES DE PC", "MODO DE APRESENTAÇÃO", "VSYNC", "LIMITE DE FPS", "ANTI-ALIASING", "INTENSIDADE DA NITIDEZ", "SOBREPOSIÇÃO DE DESEMPENHO", "SENSIBILIDADE DO RATO", "SENSIBILIDADE DA MIRA (RATO)", "SENSIBILIDADE PRÓPRIA DA LUNETA", "SENSIBILIDADE DA LUNETA (RATO)", "VELOCIDADE VERTICAL DO RATO", "INVERTER RATO (VERTICAL)", "VIBRAÇÃO DO COMANDO", "MAIS DEFINIÇÕES DE PC...", "Janela", "Janela sem margens", "Ecrã inteiro", "Novo jogo guardado", "CONTINUAR SEM GUARDAR", "Substituir este jogo guardado?", "FILTRAGEM ANISOTRÓPICA", "ATIVAR SOMBRAS EXTRAS"}},
    {"Spel verlaten?", {"PC-INSTELLINGEN", "WEERGAVEMODUS", "VSYNC", "FRAMERATELIMIET", "ANTI-ALIASING", "VERSCHERPINGSSTERKTE", "PRESTATIE-OVERLAY", "MUISGEVOELIGHEID", "MUISGEVOELIGHEID BIJ RICHTEN", "APARTE VIZIERGEVOELIGHEID", "MUISGEVOELIGHEID MET VIZIER", "VERTICALE MUISSNELHEID", "MUIS OMKEREN (VERTICAAL)", "CONTROLLERTRILLING", "MEER PC-INSTELLINGEN...", "Venster", "Randloos", "Volledig scherm", "Nieuwe opgeslagen game", "DOORGAAN ZONDER OPSLAAN", "Deze opgeslagen game overschrijven?", "ANISOTROPE FILTERING", "EXTRA SCHADUWEN INSCHAKELEN"}},
    {"ゲームを退出しますか？", {"PC設定", "表示モード", "VSYNC", "フレームレート上限", "アンチエイリアス", "シャープネス強度", "パフォーマンス表示", "マウス感度", "エイム時のマウス感度", "スコープ時の感度を使う", "スコープ時のマウス感度", "マウス上下速度", "マウス上下反転", "コントローラーの振動", "その他のPC設定...", "ウィンドウ", "ボーダーレス", "フルスクリーン", "新規セーブ", "セーブせずに続ける", "このセーブデータに上書きしますか？", "異方性フィルタリング", "追加の影を有効にする"}},
    {"게임에서 나가시겠습니까?", {"PC 설정", "디스플레이 모드", "수직 동기화", "프레임 속도 제한", "안티에일리어싱", "선명도 강도", "성능 오버레이", "마우스 감도", "조준 시 마우스 감도", "스코프 감도 따로 사용", "스코프 사용 시 마우스 감도", "마우스 상하 속도", "마우스 상하 반전", "컨트롤러 진동", "PC 설정 더 보기...", "창 모드", "테두리 없음", "전체 화면", "새로운 저장", "저장하지 않고 계속", "이 저장 데이터를 덮어쓰시겠습니까?", "이방성 필터링", "추가 그림자 켜기"}},
    {"Haluatko poistua pelistä?", {"PC-ASETUKSET", "NÄYTTÖTILA", "VSYNC", "KUVATAAJUUDEN RAJA", "REUNANPEHMENNYS", "TERÄVÖINNIN VOIMAKKUUS", "SUORITUSKYKYNÄYTTÖ", "HIIREN HERKKYYS", "HIIREN HERKKYYS TÄHDÄTESSÄ", "ERILLINEN KIIKARIN HERKKYYS", "HIIREN HERKKYYS KIIKARILLA", "HIIREN PYSTYNOPEUS", "KÄÄNNÄ HIIRI (PYSTY)", "OHJAIMEN VÄRINÄ", "LISÄÄ PC-ASETUKSIA...", "Ikkuna", "Reunaton", "Koko näyttö", "Uusi tallennus", "JATKA TALLENTAMATTA", "Korvataanko tämä tallennus?", "ANISOTROPINEN SUODATUS", "OTA LISÄVARJOT KÄYTTÖÖN"}},
    {"Vil du gå ut av spillet?", {"PC-INNSTILLINGER", "VISNINGSMODUS", "VSYNC", "BILDEFREKVENSGRENSE", "KANTUTJEVNING", "SKARPHETSSTYRKE", "YTELSESVISNING", "MUSEFØLSOMHET", "MUSEFØLSOMHET VED SIKTING", "EGEN KIKKERTSIKTE-FØLSOMHET", "MUSEFØLSOMHET MED KIKKERTSIKTE", "VERTIKAL MUSEHASTIGHET", "INVERTER MUS (VERTIKALT)", "KONTROLLERVIBRASJON", "FLERE PC-INNSTILLINGER...", "Vindu", "Uten ramme", "Fullskjerm", "Ny lagring", "FORTSETT UTEN Å LAGRE", "Vil du overskrive denne lagringen?", "ANISOTROPISK FILTRERING", "AKTIVER EKSTRA SKYGGER"}},
    {"Lämna spelet?", {"PC-INSTÄLLNINGAR", "VISNINGSLÄGE", "VSYNC", "BILDFREKVENSGRÄNS", "KANTUTJÄMNING", "SKÄRPNINGSSTYRKA", "PRESTANDAVISNING", "MUSKÄNSLIGHET", "MUSKÄNSLIGHET VID SIKTE", "EGEN KÄNSLIGHET FÖR KIKARSIKTE", "MUSKÄNSLIGHET MED KIKARSIKTE", "VERTIKAL MUSHASTIGHET", "INVERTERA MUS (VERTIKALT)", "HANDKONTROLLENS VIBRATION", "FLER PC-INSTÄLLNINGAR...", "Fönster", "Kantlöst", "Helskärm", "Ny sparning", "FORTSÄTT UTAN ATT SPARA", "Skriv över den här sparningen?", "ANISOTROPISK FILTRERING", "AKTIVERA EXTRA SKUGGOR"}},
    {"確定要離開遊戲嗎？", {"PC 設定", "顯示模式", "垂直同步", "影格率上限", "邊緣柔化", "銳利化強度", "效能顯示", "鼠標靈敏度", "瞄準時鼠標靈敏度", "使用瞄準鏡靈敏度", "瞄準鏡鼠標靈敏度", "鼠標上下速度", "鼠標上下反轉", "控制器震動", "更多 PC 設定...", "視窗", "無邊框", "全螢幕", "新增存檔", "不存檔繼續", "要覆寫此存檔嗎？", "非等向性過濾", "啟用額外陰影"}},
    {"Вы уверены, что хотите выйти?", {"НАСТРОЙКИ ПК", "РЕЖИМ ЭКРАНА", "ВЕРТИКАЛЬНАЯ СИНХРОНИЗАЦИЯ", "ОГРАНИЧЕНИЕ FPS", "СГЛАЖИВАНИЕ", "СИЛА РЕЗКОСТИ", "ПАНЕЛЬ ПРОИЗВОДИТЕЛЬНОСТИ", "ЧУВСТВИТЕЛЬНОСТЬ МЫШИ", "ЧУВСТВ. МЫШИ ПРИ ПРИЦЕЛИВАНИИ", "ОТДЕЛЬНАЯ ЧУВСТВ. С ПРИЦЕЛОМ", "ЧУВСТВ. МЫШИ С ПРИЦЕЛОМ", "СКОРОСТЬ МЫШИ ПО ВЕРТИКАЛИ", "ИНВЕРСИЯ МЫШИ ПО ВЕРТИКАЛИ", "ВИБРАЦИЯ КОНТРОЛЛЕРА", "ДРУГИЕ НАСТРОЙКИ ПК...", "Окно", "Без рамки", "Полный экран", "Новое сохранение", "ПРОДОЛЖИТЬ БЕЗ СОХРАНЕНИЯ", "Перезаписать это сохранение?", "АНИЗОТРОПНАЯ ФИЛЬТРАЦИЯ", "ВКЛЮЧИТЬ ДОПОЛНИТЕЛЬНЫЕ ТЕНИ"}},
    {"Wyjść z gry?", {"USTAWIENIA PC", "TRYB WYŚWIETLANIA", "VSYNC", "LIMIT KLATEK", "WYGŁADZANIE KRAWĘDZI", "SIŁA WYOSTRZANIA", "WSKAŹNIK WYDAJNOŚCI", "CZUŁOŚĆ MYSZY", "CZUŁOŚĆ MYSZY PRZY CELOWANIU", "OSOBNA CZUŁOŚĆ Z LUNETĄ", "CZUŁOŚĆ MYSZY Z LUNETĄ", "PIONOWA PRĘDKOŚĆ MYSZY", "ODWRÓĆ MYSZ (PION)", "WIBRACJE KONTROLERA", "WIĘCEJ USTAWIEŃ PC...", "Okno", "Bez ramki", "Pełny ekran", "Nowy zapis", "KONTYNUUJ BEZ ZAPISU", "Nadpisać ten zapis?", "FILTROWANIE ANIZOTROPOWE", "WŁĄCZ DODATKOWE CIENIE"}},
    {"¿Abandonar el juego?", {"CONFIGURACIÓN DE PC", "MODO DE PANTALLA", "VSYNC", "LÍMITE DE FPS", "ANTIALIASING", "INTENSIDAD DE NITIDEZ", "INDICADOR DE RENDIMIENTO", "SENSIBILIDAD DEL MOUSE", "SENSIBILIDAD AL APUNTAR (MOUSE)", "SENSIBILIDAD PROPIA CON MIRA", "SENSIBILIDAD CON MIRA (MOUSE)", "VELOCIDAD VERTICAL DEL MOUSE", "INVERTIR MOUSE (VERTICAL)", "VIBRACIÓN DEL CONTROL", "MÁS CONFIGURACIÓN DE PC...", "Ventana", "Sin bordes", "Pantalla completa", "Nueva partida guardada", "CONTINUAR SIN GUARDAR", "¿Sobrescribir esta partida guardada?", "FILTRADO ANISOTRÓPICO", "ACTIVAR SOMBRAS ADICIONALES"}},
    {"Sair do jogo?", {"CONFIGURAÇÕES DE PC", "MODO DE EXIBIÇÃO", "VSYNC", "LIMITE DE FPS", "ANTISSERRILHAMENTO", "INTENSIDADE DA NITIDEZ", "OVERLAY DE DESEMPENHO", "SENSIBILIDADE DO MOUSE", "SENSIBILIDADE DA MIRA (MOUSE)", "SENSIBILIDADE PRÓPRIA DA LUNETA", "SENSIBILIDADE DA LUNETA (MOUSE)", "VELOCIDADE VERTICAL DO MOUSE", "INVERTER MOUSE (VERTICAL)", "VIBRAÇÃO DO CONTROLE", "MAIS CONFIGURAÇÕES DE PC...", "Janela", "Janela sem bordas", "Tela cheia", "Novo jogo salvo", "CONTINUAR SEM SALVAR", "Sobrescrever este jogo salvo?", "FILTRAGEM ANISOTRÓPICA", "ATIVAR SOMBRAS EXTRAS"}},
    {"Έξοδος από το παιχνίδι;", {"ΡΥΘΜΙΣΕΙΣ PC", "ΛΕΙΤΟΥΡΓΙΑ ΟΘΟΝΗΣ", "VSYNC", "ΟΡΙΟ FPS", "ΕΞΟΜΑΛΥΝΣΗ ΑΚΜΩΝ", "ΕΝΤΑΣΗ ΟΞΥΤΗΤΑΣ", "ΕΝΔΕΙΞΗ ΑΠΟΔΟΣΗΣ", "ΕΥΑΙΣΘΗΣΙΑ ΠΟΝΤΙΚΙΟΥ", "ΕΥΑΙΣΘ. ΣΤΟΧΕΥΣΗΣ (ΠΟΝΤΙΚΙ)", "ΞΕΧΩΡΙΣΤΗ ΕΥΑΙΣΘ. ΔΙΟΠΤΡΑΣ", "ΕΥΑΙΣΘ. ΔΙΟΠΤΡΑΣ (ΠΟΝΤΙΚΙ)", "ΚΑΤΑΚΟΡΥΦΗ ΤΑΧΥΤΗΤΑ ΠΟΝΤΙΚΙΟΥ", "ΑΝΤΙΣΤΡΟΦΗ ΠΟΝΤΙΚΙΟΥ (ΚΑΘΕΤΑ)", "ΔΟΝΗΣΗ ΧΕΙΡΙΣΤΗΡΙΟΥ", "ΠΕΡΙΣΣΟΤΕΡΕΣ ΡΥΘΜΙΣΕΙΣ PC...", "Παράθυρο", "Χωρίς περίγραμμα", "Πλήρης οθόνη", "Νέα αποθήκευση", "ΣΥΝΕΧΕΙΑ ΧΩΡΙΣ ΑΠΟΘΗΚΕΥΣΗ", "Αντικατάσταση αυτής της αποθήκευσης;", "ΑΝΙΣΟΤΡΟΠΙΚΟ ΦΙΛΤΡΑΡΙΣΜΑ", "ΕΝΕΡΓΟΠΟΙΗΣΗ ΠΡΟΣΘΕΤΩΝ ΣΚΙΩΝ"}},
    {"Odejít ze hry?", {"NASTAVENÍ PC", "REŽIM ZOBRAZENÍ", "VSYNC", "OMEZENÍ SNÍMKOVÉ FREKVENCE", "VYHLAZOVÁNÍ HRAN", "SÍLA ZOSTŘENÍ", "UKAZATEL VÝKONU", "CITLIVOST MYŠI", "CITLIVOST MYŠI PŘI MÍŘENÍ", "VLASTNÍ CITLIVOST S OPTIKOU", "CITLIVOST MYŠI S OPTIKOU", "SVISLÁ RYCHLOST MYŠI", "OBRÁTIT MYŠ (SVISLE)", "VIBRACE OVLADAČE", "DALŠÍ NASTAVENÍ PC...", "Okno", "Bez okrajů", "Celá obrazovka", "Nová uložená pozice", "POKRAČOVAT BEZ ULOŽENÍ", "Přepsat tuto uloženou pozici?", "ANIZOTROPNÍ FILTROVÁNÍ", "ZAPNOUT DALŠÍ STÍNY"}},
    {"Oyundan çıkılsın mı?", {"PC AYARLARI", "EKRAN MODU", "VSYNC", "KARE HIZI SINIRI", "KENAR YUMUŞATMA", "KESKİNLEŞTİRME GÜCÜ", "PERFORMANS GÖSTERGESİ", "FARE HASSASİYETİ", "NİŞAN ALIRKEN FARE HASSASİYETİ", "AYRI DÜRBÜN HASSASİYETİ", "DÜRBÜNLE FARE HASSASİYETİ", "FARE DİKEY HIZI", "FAREYİ TERS ÇEVİR (DİKEY)", "KUMANDA TİTREŞİMİ", "DAHA FAZLA PC AYARI...", "Pencere", "Kenarlıksız", "Tam ekran", "Yeni kayıt", "KAYDETMEDEN DEVAM ET", "Bu kayıt üzerine yazılsın mı?", "ANİZOTROPİK FİLTRELEME", "EK GÖLGELERİ ETKİNLEŞTİR"}},
};
enum { LANGUAGE_COUNT = sizeof(languages) / sizeof(languages[0]) };

/* Sliders: the setting in percent, from min to max in steps of step. */
typedef struct {
    Gow3PcRowKind kind;
    int setting; /* GOW3_PC_* (gow3gpu.h), MORE_SETTINGS, or unused */
    uint32_t game_text; /* a header's label */
    int text; /* TEXT_*: the row's name */
    int min, max, step;
} RowDef;
static const RowDef defs[] = {
    {GOW3_PC_ROW_HEADER, 0, TEXT_MENU_VIDEO, TEXT_NONE, 0, 0, 0},
    {GOW3_PC_ROW_LIST, GOW3_PC_DISPLAY_MODE, 0, TEXT_DISPLAY_MODE, 0, 0, 0},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_VSYNC, 0, TEXT_VSYNC, 0, 0, 0},
    {GOW3_PC_ROW_LIST, GOW3_PC_FPS_LIMIT, 0, TEXT_FPS_LIMIT, 0, 0, 0},
    {GOW3_PC_ROW_LIST, GOW3_PC_ANTI_ALIASING, 0, TEXT_ANTI_ALIASING, 0, 0, 0},
    {GOW3_PC_ROW_LIST, GOW3_PC_ANISOTROPIC_FILTER, 0, TEXT_ANISOTROPIC, 0, 0, 0},
    {GOW3_PC_ROW_SLIDER, GOW3_PC_SHARPENING, 0, TEXT_SHARPENING, 0, 100, 5},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_PERF_OVERLAY, 0, TEXT_PERF_OVERLAY, 0, 0, 0},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_FIRE_SHADOWS, 0, TEXT_FIRE_SHADOWS, 0, 0, 0},
    {GOW3_PC_ROW_HEADER, 0, TEXT_MENU_CONTROLS, TEXT_NONE, 0, 0, 0},
    {GOW3_PC_ROW_SLIDER, GOW3_PC_MOUSE_SENS, 0, TEXT_MOUSE_SENS, 10, 500, 10},
    {GOW3_PC_ROW_SLIDER, GOW3_PC_MOUSE_AIM_SENS, 0, TEXT_MOUSE_AIM_SENS, 10, 500, 10},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_MOUSE_SCOPE_ON, 0, TEXT_MOUSE_SCOPE_ON, 0, 0, 0},
    {GOW3_PC_ROW_SLIDER, GOW3_PC_MOUSE_SCOPE_SENS, 0, TEXT_MOUSE_SCOPE_SENS, 10, 200, 10},
    {GOW3_PC_ROW_SLIDER, GOW3_PC_MOUSE_VERTICAL, 0, TEXT_MOUSE_VERTICAL, 25, 300, 25},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_MOUSE_INVERT, 0, TEXT_MOUSE_INVERT, 0, 0, 0},
    {GOW3_PC_ROW_TOGGLE, GOW3_PC_VIBRATION, 0, TEXT_VIBRATION, 0, 0, 0},
    {GOW3_PC_ROW_SPACER, 0, 0, TEXT_NONE, 0, 0, 0},
    {GOW3_PC_ROW_BUTTON, MORE_SETTINGS, 0, TEXT_MORE, 0, 0, 0},
};
enum { ROW_COUNT = sizeof(defs) / sizeof(defs[0]) };
/* The cast to int: ROW_COUNT and the GOW3_MENU_PC_MAX_* constants are different anonymous enums
 * (menu_pc_hook.h); GCC's -Wenum-compare rejects their comparison (clang, the Windows build,
 * does not warn). */
_Static_assert(ROW_COUNT <= (int)GOW3_MENU_PC_MAX_ROWS && ROW_COUNT <= (int)GOW3_MENU_PC_MAX_OPTIONS,
               "rows fit the page's ids");

static Gow3PcRow rows[ROW_COUNT];
/* The values the game's widgets read and write (a toggle's int, a slider's float in percent),
 * and the setting's value as last applied or read (the tick compares the two). */
static _Alignas(16) int32_t toggle_values[ROW_COUNT];
static _Alignas(16) float slider_values[ROW_COUNT];
static int applied[ROW_COUNT];
static atomic_int bound[ROW_COUNT];
static int save_countdown;

const Gow3PcRow *runtime_menu_pc_rows(unsigned *count) {
    const char *env = getenv("GOW3_PC_MENU");
    if (env && env[0] == '0') return NULL;
    for (unsigned i = 0; i < ROW_COUNT; ++i)
        rows[i] = (Gow3PcRow){defs[i].kind,
                              defs[i].kind == GOW3_PC_ROW_HEADER ? defs[i].game_text : GOW3_MENU_PC_ROW_LABEL + i,
                              GOW3_MENU_PC_OPTION + i};
    *count = ROW_COUNT;
    return rows;
}

/* The table's language (index into languages; English when unknown). */
static int language_of(Gow3TextLookup lookup, void *table) {
    const char *leave = lookup(table, GOW3_MENU_TEXT_LEAVE_GAME);
    if (leave)
        for (int i = 0; i < LANGUAGE_COUNT; ++i)
            if (!strcmp(leave, languages[i].leave_game)) return i;
    return 0;
}

/* The language of the last table the text hook saw, and that table with the game's lookup (the
 * list labels have no table of their own). */
static atomic_int last_language;
static Gow3TextLookup last_lookup;
static void *_Atomic last_table;

const char *runtime_menu_pc_text(Gow3TextLookup lookup, void *table, uint32_t sid) {
    const int language = language_of(lookup, table);
    atomic_store(&last_language, language);
    last_lookup = lookup;
    atomic_store(&last_table, table);
    if (sid == GOW3_MENU_PC_LABEL || sid == GOW3_MENU_PC_TITLE) return languages[language].texts[TEXT_TITLE];
    if (sid == GOW3_MENU_SAVES_TITLE) return runtime_menu_saves_text(lookup, table, sid);
    if (sid == GOW3_MENU_SAVES_OVERWRITE_TEXT) return languages[language].texts[TEXT_OVERWRITE];
    /* The footer's Square prompt: the cinematics' SUBTITLES ON/OFF, on the save list "CONTINUE
     * WITHOUT SAVING" while it offers that (runtime_menu_saves.c). */
    if (sid == GOW3_MENU_TEXT_SUBTITLES_ON || sid == GOW3_MENU_TEXT_SUBTITLES_OFF)
        return runtime_menu_saves_no_save_offered() ? languages[language].texts[TEXT_NO_SAVE] : NULL;
    if (sid < GOW3_MENU_PC_ROW_LABEL || sid >= GOW3_MENU_PC_ROW_LABEL + ROW_COUNT) return NULL;
    const RowDef *def = &defs[sid - GOW3_MENU_PC_ROW_LABEL];
    if (def->text == TEXT_NONE) return NULL;
    return languages[language].texts[def->text];
}

int runtime_menu_pc_button(uint32_t hash) {
    if (hash < GOW3_MENU_PC_ROW || hash >= GOW3_MENU_PC_ROW + ROW_COUNT) return 0;
    const RowDef *def = &defs[hash - GOW3_MENU_PC_ROW];
    if (def->kind == GOW3_PC_ROW_BUTTON && def->setting == MORE_SETTINGS) {
        puts("Menu: PC settings, more settings (overlay menu)");
        gow3gpu_settings_menu_open();
    }
    return 1;
}

/* --- Bindings ------------------------------------------------------------------------------- */
typedef ABI void *(*Factory)(void *menu, uint32_t kind, uint32_t a, uint32_t sid, uint32_t b, uint32_t c, uint32_t d);
static Factory factory;
/* A copy of the int binding's vtable whose label slot names list values (display modes, FPS). */
static uint64_t list_vtable[GOW3_BINDING_VTABLE_SLOTS];
static atomic_int list_vtable_ready;

static int row_of_value(const void *value) {
    const int32_t *v = value;
    return v >= toggle_values && v < toggle_values + ROW_COUNT ? (int)(v - toggle_values) : -1;
}

/* The int binding's label(self, index) for lists: the value's name, in the page's language. */
static ABI const char *list_label(void *self, int index) {
    const int row = row_of_value((const void *)(uintptr_t)gow3_menu_u64((unsigned char *)self + GOW3_BINDING_VALUE));
    if (row < 0) return "";
    const int setting = defs[row].setting;
    if (setting == GOW3_PC_DISPLAY_MODE) {
        const int language = atomic_load(&last_language);
        return index >= 0 && index < 3 ? languages[language].texts[TEXT_WINDOWED + index] : "";
    }
    /* Anti-aliasing and anisotropic filtering off: the game's own Off ("Desligado"). */
    void *table = atomic_load(&last_table);
    if ((setting == GOW3_PC_ANTI_ALIASING || setting == GOW3_PC_ANISOTROPIC_FILTER) && index == 0 && last_lookup &&
        table) {
        const char *off = last_lookup(table, TEXT_MENU_OFF);
        if (off) return off;
    }
    const char *text = gow3gpu_pc_setting_choice(setting, index);
    return text ? text : "";
}

void runtime_menu_pc_set_factory(void *original) {
    factory = (Factory)original;
}

/* Called instead of the factory for our sids (same arguments): the game's binding for a sid of
 * the same kind, pointed at our value. */
ABI void *runtime_menu_pc_binding(void *menu, uint32_t kind, uint32_t a, uint32_t sid, uint32_t b, uint32_t c,
                                  uint32_t d) {
    const unsigned row = sid - GOW3_MENU_PC_OPTION;
    if (factory && sid == GOW3_MENU_SAVES_OPTION) { /* the save list: an int binding of ours */
        unsigned char *binding = factory(menu, kind, a, GOW3_MENU_OPTION_LOCK_30, b, c, d);
        if (binding) runtime_menu_saves_bind(binding);
        return binding;
    }
    if (!factory || row >= ROW_COUNT) return NULL;
    const RowDef *def = &defs[row];
    const int toggle = def->kind == GOW3_PC_ROW_TOGGLE || def->kind == GOW3_PC_ROW_LIST;
    const int slider = def->kind == GOW3_PC_ROW_SLIDER;
    if (!toggle && !slider) return NULL;
    unsigned char *binding = factory(menu, kind, a, toggle ? GOW3_MENU_OPTION_LOCK_30 : GOW3_MENU_OPTION_SENSITIVITY, b, c, d);
    if (!binding) return NULL;
    const int value = gow3gpu_pc_setting(def->setting);
    applied[row] = value;
    if (toggle) {
        const int count = gow3gpu_pc_setting_count(def->setting);
        toggle_values[row] = value;
        gow3_menu_put64(binding + GOW3_BINDING_VALUE, (uint64_t)(uintptr_t)&toggle_values[row]);
        gow3_menu_put32(binding + GOW3_BINDING_MIN, 0);
        gow3_menu_put32(binding + GOW3_BINDING_MAX, (uint32_t)(count > 1 ? count - 1 : 0));
        if (def->kind == GOW3_PC_ROW_LIST) { /* our labels */
            if (!atomic_load(&list_vtable_ready)) {
                memcpy(list_vtable, (const void *)(uintptr_t)gow3_menu_u64(binding), sizeof(list_vtable));
                list_vtable[GOW3_BINDING_LABEL_SLOT] = (uint64_t)(uintptr_t)list_label;
                atomic_store(&list_vtable_ready, 1);
            }
            gow3_menu_put64(binding, (uint64_t)(uintptr_t)list_vtable);
        }
    } else {
        slider_values[row] = (float)value;
        gow3_menu_put64(binding + GOW3_BINDING_VALUE, (uint64_t)(uintptr_t)&slider_values[row]);
        gow3_menu_put32(binding + GOW3_BINDING_STEPS, (uint32_t)((def->max - def->min) / def->step));
        /* Linear: the middle of the bar is the middle of the range. */
        const float min = (float)def->min, max = (float)def->max, middle = (min + max) * 0.5f;
        memcpy(binding + GOW3_BINDING_FLOAT_MIN, &min, 4);
        memcpy(binding + GOW3_BINDING_FLOAT_MID, &middle, 4);
        memcpy(binding + GOW3_BINDING_FLOAT_MAX, &max, 4);
    }
    atomic_store_explicit(&bound[row], 1, memory_order_release);
    return binding;
}

/* From the pad's sampling: values the menu changed go to the settings (saved half a second
 * after the last change); settings changed elsewhere (the overlay menu) show in the page. */
void runtime_menu_pc_tick(void) {
    int changed = 0;
    for (unsigned i = 0; i < ROW_COUNT; ++i) {
        if (!atomic_load_explicit(&bound[i], memory_order_acquire)) continue;
        const RowDef *def = &defs[i];
        int shown;
        if (def->kind != GOW3_PC_ROW_SLIDER) {
            shown = toggle_values[i];
        } else {
            const float v = slider_values[i];
            shown = (int)lroundf((v < def->min ? def->min : v > def->max ? def->max : v) / def->step) * def->step;
        }
        if (shown != applied[i]) {
            gow3gpu_pc_setting_set(def->setting, shown);
            applied[i] = shown;
            changed = 1;
            continue;
        }
        const int current = gow3gpu_pc_setting(def->setting);
        if (current != applied[i]) {
            applied[i] = current;
            if (def->kind != GOW3_PC_ROW_SLIDER) toggle_values[i] = current;
            else slider_values[i] = (float)current;
        }
    }
    if (changed) save_countdown = 30;
    else if (save_countdown > 0 && --save_countdown == 0) gow3gpu_pc_settings_save();
}

const char *runtime_menu_pc_game_text(uint32_t sid) {
    void *table = atomic_load(&last_table);
    return last_lookup && table ? last_lookup(table, sid) : NULL;
}

const char *runtime_menu_pc_new_save_text(void) {
    return languages[atomic_load(&last_language)].texts[TEXT_NEW_SAVE];
}
