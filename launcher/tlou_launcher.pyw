"""The Last of Us Remastered: game launcher (Tkinter, nothing to install).

Reads and writes the settings the game reads (analysis/tlou-user/tlou.ini, other keys stay as they
are), keeps its own switches in launcher.json beside it and starts tools/probe_tlou.py.
Usage: tlou_launcher.pyw [--seconds N] [--close-after S]   (for tests: a time limit for the game, and
for the launcher's window; default none)
"""
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import updater  # noqa: E402  (launcher/updater.py: updates from the GitHub releases)
ROOT = HERE.parent
ICON = HERE / "assets/icon.ico"   # made from the player's own picture, not in the repository
GAME_ICON = "sce_sys/icon0.png"   # without it, the icon of the player's own dump, read at run time
USER = ROOT / "analysis/tlou-user"
INI = USER / "tlou.ini"
SWITCHES = USER / "launcher.json"
LOG = ROOT / "analysis/tlou/boot.log"
RUN_LOG = ROOT / "analysis/tlou/launcher-run.log"
GAME = ROOT.parent / "CUSA00552"

FPS = ["30", "40", "45", "50", "55", "60", "75", "90", "100", "120"]
DISPLAY = {"borderless": "borderless", "fullscreen": "fullscreen", "windowed": "windowed"}   # text key: value
UPSCALERS = {"off": "off", "FSR 3": "fsr3", "TAA": "taa", "DLSS": "dlss"}
PRESETS = {"q_native": "0", "q_quality": "1", "q_balanced": "2", "q_performance": "3", "q_ultra": "4"}
# The size the game draws at (output_res in tlou.ini; tools/probe_tlou.py patches the game's
# render size to it at start). The ultrawide sizes (21:9, 32:9) also get the aspect ratio patches.
RESOLUTIONS = ["1280x720", "1600x900", "1920x1080", "2560x1440", "2880x1620", "3200x1800", "3840x2160",
               "2560x1080", "3440x1440", "3840x1600", "3840x1080", "5120x1440"]
DEFAULT_SWITCHES = {"motion_blur": False, "job_sleep": True, "profile": False, "background_shaders": True,
                    "pro_shadows": True, "ambient_half": True, "quarter_depth": True, "low_vram": "auto"}
# Video memory (GOW3_LOW_VRAM, gpu/shadps4/video_core/renderer_vulkan/vk_low_vram.h): auto turns the
# low video memory mode on for cards of 3 GB or less; 2-3 GB forces it, 4 GB or more turns it off.
LOW_VRAM = {"vram_auto": "auto", "vram_low": "on", "vram_normal": "off"}
DEFAULT_LANGUAGE = "en"
# What Start-TLOU-Profile.cmd sets, for the "profiler" switch.
PROFILE_ENV = {"GOW3_PROFILE": "10", "GOW3_THREAD_IDS": "1", "GOW3_COMPUTE_WAIT_STATS": "1",
               "GOW3_FAULT_STATS": "1", "GOW3_FAULT_TIMES": "1",
               "GOW3_PROFILE_GUEST": "main,NdJobWorkerThread",
               "GOW3_SHARE_CHECK": "1", "GOW3_PROFILE_STALLS": "1",
               # Checkpoints only: the device fault report itself caused device loss at the TLOU menu.
               "GOW3_GPU_CHECKPOINTS": "4",
               # Keep buffer ranges for the dispatches immediately around the water/pallet loss.
               "GOW3_TRACE_DISPATCH": "82c46b8b,e44070db,106beafd,15d34dcb,951b339e",
               "GOW3_DUMP_DIR": str(ROOT / "analysis/tlou/profile-user")}

# Debug menu. Live: optimizations the game re-reads every 250 ms (gow3_toggles.h, a bit set in the
# file switches one off). The file lasts one run: probe_tlou.py deletes it at the start.
TOGGLE_FILE = USER / "disabled-optimizations.txt"
TOGGLES = [
    ("Memória", [
        ("Cache de regiões de memória", 1), ("Saída rápida do rastreio de páginas", 4),
        ("Limite de consulta de operações pendentes", 8), ("Leitura rastreada por página", 2048),
        ("Páginas quentes", 32768), ("Janela em torno da falha de escrita", 65536),
        ("Escritas de memória na gravação de draws", 1 << 42), ("Escritas do jogo em ordem", 1 << 49),
    ]),
    ("Threads e gravação", [
        ("Pipeline de draws em dois estágios", 1 << 37), ("Dispatches na thread de draws", 1 << 40),
        ("Draws indiretos na thread de draws", 1 << 46), ("Tarefas na thread de draws", 1 << 38),
        ("Gravação Vulkan em thread própria", 16), ("Preparação de draws em workers", 8192),
        ("Recursos preparados nos workers", 4096), ("Entradas de draw adiantadas", 1 << 35),
        ("Ajudante de texturas (causou bugs gráficos)", 1 << 34), ("Prefetch de gravação", 1 << 32),
    ]),
    ("Cópias e sincronização", [
        ("Uploads na thread de gravação", 256), ("Cópias curtas na thread de gravação", 16384),
        ("Cópias grandes em threads paralelas", 131072), ("Cópias curtas em lote", 524288),
        ("Fences assíncronas", 262144), ("Esperas de fence adiadas", 1 << 39), ("Fences na thread Vulkan", 1 << 41),
        ("Upload sem trava", 64), ("Anel de constantes", 1 << 36), ("Shader de cópia múltipla", 1 << 43),
    ]),
    ("Texturas e estado", [
        ("Cache de fetch shaders", 2), ("Cache de descritores de textura", 32), ("Cache de busca de imagens", 128),
        ("Memo de barreiras", 512), ("Memo de bind de texturas", 1024), ("Memo de views de textura", 1 << 33),
        ("Memo de estado de render", 1 << 44), ("Memo de conjuntos de textura", 1 << 45),
        ("Caminho rápido de UpdateImage", 1 << 30),
    ]),
]

# Startup: environment of the next run; only what differs from probe_tlou.py's defaults is passed.
# (label, variable, value on, value off, on by default)
STARTUP = [
    ("Threads", [
        ("Pipeline de draws", "GOW3_DRAW_PIPE", "1", "0", True),
        ("Gravação Vulkan em thread própria", "GOW3_VK_RECORD_THREAD", "1", "0", True),
        ("Prioridade alta nas threads da GPU", "GOW3_GPU_THREAD_PRIORITY", "above", "normal", True),
        ("Prioridades das threads do jogo", "GOW3_GUEST_PRIORITY", "1", "0", True),
        ("Quadros em andamento", "GOW3_FRAMES_IN_FLIGHT", "1", "0", True),
        ("Menos esperas entre estágios", "GOW3_FEWER_DRAINS", "1", "0", True),
        ("Busca de dados sem bloquear", "GOW3_FETCH_NO_BLOCK", "1", "0", True),
        ("Timer de 1 ms", "GOW3_TIMER_RES", "1", "0", True),
        ("Sem limitação de energia", "GOW3_POWER_THROTTLE", "0", "1", True),
        ("Condition variables rápidas", "GOW3_FUTEX_COND", "1", "0", True),
        ("Reaproveitar buffers de draws", "GOW3_PREP_POOL", "1", "0", True),
        ("Economias por draw", "GOW3_CPU_TRIMS", "1", "0", True),
        ("Ajudante de texturas (causou bugs gráficos)", "GOW3_TEXTURE_HELPER", "1", "0", False),
    ]),
    ("Memória e leitura da GPU", [
        ("Escritas ao lado de dados da GPU sem espera", "GOW3_SHARE_UNFINISHED_WRITES", "1", "0", True),
        ("Resultados de DMA no fim do pipe", "GOW3_DMA_EOP_READBACK", "1", "0", True),
        ("Cópia de imagens minúsculas (foco da mira)", "GOW3_TINY_IMAGE_READBACK", "1", "0", True),
        # GitHub #57/#63: NPCs jumping or bobbing (vk_rasterizer.cpp TlouMemCpyEopReadback), and
        # the cloth joint results of every cloth size (TlouClothResults).
        ("Leitura das cópias da GPU (NPCs pulando)", "GOW3_TLOU_MEMCPY_READBACK", "1", "0", True),
        ("Juntas do tecido em todos os tamanhos", "GOW3_CLOTH_ALL_SIZES", "1", "0", True),
        ("Acesso direto à memória", "GOW3_DIRECT_MEMORY_ACCESS", "1", "0", True),
        ("TSC nativo", "GOW3_NATIVE_TSC", "1", "0", True),
        ("Buffer grande de staging reservado", "GOW3_STAGING_LARGE_MB", "128", "0", True),
    ]),
    ("Shaders e proteção", [
        ("Cache de pipelines", "GOW3_PIPELINE_CACHE", "1", "0", True),
        ("Binários de pipeline", "GOW3_PIPELINE_BINARIES", "1", "0", True),
        ("Prefetch de pipelines", "GOW3_PREFETCH_PIPELINES", "1", "0", True),
        ("Proteção da red zone", "GOW3_RED_ZONE_REWRITE", "1", "0", True),
        ("Correção da chuva branca", "GOW3_RAIN_SCALE", "0.12", "1", True),
        # On (auto): on NVIDIA cards before Turing the largest cloth solver runs as 256 threads
        # with 2 vertices each (lane_coarsening.cpp); off: as it is, which loses the device there.
        ("Contorno do tecido na GTX 900/10 (evita crash)", "GOW3_PASCAL_CLOTH512", "auto", "1", True),
    ]),
    ("Tela", [
        # GitHub #139: on a Resolution wider than 16:9, the interface's full-screen layers (menus'
        # dimming, fades, damage vignette, the rifle scope's black frame) reach the screen's edges
        # (gow3_pc/src/runtime_hud.c); off: they stop at the 16:9 edges as before.
        ("Camadas do HUD até as bordas (ultrawide)", "GOW3_HUD_BARS", "1", "0", True),
    ]),
]

# The debug menu in the launcher's languages: {Portuguese text: the other languages, in this order}.
DEBUG_LANGUAGES = ("en", "es", "fr", "de", "it", "pl", "ru", "ja")
DEBUG_TEXT = {
    'Memória': (
        'Memory',
        'Memoria',
        'Mémoire',
        'Speicher',
        'Memoria',
        'Pamięć',
        'Память',
        'メモリ'),
    'Threads e gravação': (
        'Threads and recording',
        'Hilos y grabación',
        'Threads et enregistrement',
        'Threads und Aufzeichnung',
        'Thread e registrazione',
        'Wątki i nagrywanie',
        'Потоки и запись',
        'スレッドと記録'),
    'Cópias e sincronização': (
        'Copies and synchronization',
        'Copias y sincronización',
        'Copies et synchronisation',
        'Kopien und Synchronisierung',
        'Copie e sincronizzazione',
        'Kopie i synchronizacja',
        'Копирование и синхронизация',
        'コピーと同期'),
    'Texturas e estado': (
        'Textures and state',
        'Texturas y estado',
        'Textures et état',
        'Texturen und Zustand',
        'Texture e stato',
        'Tekstury i stan',
        'Текстуры и состояние',
        'テクスチャと状態'),
    'Threads': (
        'Threads',
        'Hilos',
        'Threads',
        'Threads',
        'Thread',
        'Wątki',
        'Потоки',
        'スレッド'),
    'Memória e leitura da GPU': (
        'Memory and GPU readback',
        'Memoria y lectura de la GPU',
        'Mémoire et relecture GPU',
        'Speicher und GPU-Rücklesen',
        'Memoria e lettura dalla GPU',
        'Pamięć i odczyt z GPU',
        'Память и чтение из GPU',
        'メモリと GPU リードバック'),
    'Shaders e proteção': (
        'Shaders and protection',
        'Shaders y protección',
        'Shaders et protection',
        'Shader und Schutz',
        'Shader e protezione',
        'Shadery i ochrona',
        'Шейдеры и защита',
        'シェーダーと保護'),
    'Cache de regiões de memória': (
        'Memory region cache',
        'Caché de regiones de memoria',
        'Cache des régions mémoire',
        'Speicherregionen-Cache',
        'Cache delle regioni di memoria',
        'Pamięć podręczna regionów pamięci',
        'Кэш областей памяти',
        'メモリ領域キャッシュ'),
    'Saída rápida do rastreio de páginas': (
        'Fast exit from page tracking',
        'Salida rápida del rastreo de páginas',
        'Sortie rapide du suivi des pages',
        'Schneller Ausstieg aus der Seitenverfolgung',
        'Uscita rapida dal tracciamento pagine',
        'Szybkie wyjście ze śledzenia stron',
        'Быстрый выход из отслеживания страниц',
        'ページ追跡の高速終了'),
    'Limite de consulta de operações pendentes': (
        'Pending operations query limit',
        'Límite de consulta de operaciones pendientes',
        'Limite de requêtes des opérations en attente',
        'Abfragelimit für ausstehende Operationen',
        'Limite di interrogazione delle operazioni in sospeso',
        'Limit zapytań o oczekujące operacje',
        'Лимит запросов ожидающих операций',
        '保留中の操作の照会制限'),
    'Leitura rastreada por página': (
        'Tracked read per page',
        'Lectura rastreada por página',
        'Lecture suivie par page',
        'Verfolgtes Lesen pro Seite',
        'Lettura tracciata per pagina',
        'Śledzony odczyt na stronę',
        'Отслеживаемое чтение по страницам',
        'ページ単位の追跡読み取り'),
    'Páginas quentes': (
        'Hot pages',
        'Páginas calientes',
        'Pages chaudes',
        'Heiße Seiten',
        'Pagine calde',
        'Gorące strony',
        'Горячие страницы',
        'ホットページ'),
    'Janela em torno da falha de escrita': (
        'Window around the write fault',
        'Ventana alrededor del fallo de escritura',
        "Fenêtre autour de la faute d'écriture",
        'Fenster um den Schreibfehler',
        "Finestra attorno all'errore di scrittura",
        'Okno wokół błędu zapisu',
        'Окно вокруг ошибки записи',
        '書き込みフォールト周辺のウィンドウ'),
    'Escritas de memória na gravação de draws': (
        'Memory writes while recording draws',
        'Escrituras de memoria al grabar draws',
        "Écritures mémoire pendant l'enregistrement des draws",
        'Speicherschreibvorgänge bei der Draw-Aufzeichnung',
        'Scritture di memoria durante la registrazione dei draw',
        'Zapisy pamięci podczas nagrywania draws',
        'Запись в память при записи draw',
        'ドロー記録中のメモリ書き込み'),
    'Escritas do jogo em ordem': (
        'Game writes in order',
        'Escrituras del juego en orden',
        "Écritures du jeu dans l'ordre",
        'Spiel-Schreibvorgänge in Reihenfolge',
        'Scritture del gioco in ordine',
        'Zapisy gry po kolei',
        'Запись игры по порядку',
        'ゲームの書き込みを順序どおりに'),
    'Pipeline de draws em dois estágios': (
        'Two-stage draw pipeline',
        'Pipeline de draws en dos etapas',
        'Pipeline de draws à deux étages',
        'Zweistufige Draw-Pipeline',
        'Pipeline dei draw a due stadi',
        'Dwuetapowy potok draws',
        'Двухэтапный конвейер draw',
        '2 段階のドローパイプライン'),
    'Dispatches na thread de draws': (
        'Dispatches on the draw thread',
        'Dispatches en el hilo de draws',
        'Dispatches sur le thread de draws',
        'Dispatches im Draw-Thread',
        'Dispatch sul thread dei draw',
        'Dispatche w wątku draws',
        'Dispatch в потоке draw',
        'ドロースレッドでのディスパッチ'),
    'Draws indiretos na thread de draws': (
        'Indirect draws on the draw thread',
        'Draws indirectos en el hilo de draws',
        'Draws indirects sur le thread de draws',
        'Indirekte Draws im Draw-Thread',
        'Draw indiretti sul thread dei draw',
        'Pośrednie draws w wątku draws',
        'Косвенные draw в потоке draw',
        'ドロースレッドでの間接ドロー'),
    'Tarefas na thread de draws': (
        'Tasks on the draw thread',
        'Tareas en el hilo de draws',
        'Tâches sur le thread de draws',
        'Aufgaben im Draw-Thread',
        'Attività sul thread dei draw',
        'Zadania w wątku draws',
        'Задачи в потоке draw',
        'ドロースレッドでのタスク'),
    'Gravação Vulkan em thread própria': (
        'Vulkan recording on its own thread',
        'Grabación Vulkan en su propio hilo',
        'Enregistrement Vulkan sur son propre thread',
        'Vulkan-Aufzeichnung in eigenem Thread',
        'Registrazione Vulkan su thread dedicato',
        'Nagrywanie Vulkan we własnym wątku',
        'Запись Vulkan в отдельном потоке',
        '専用スレッドでの Vulkan 記録'),
    'Preparação de draws em workers': (
        'Draw preparation on workers',
        'Preparación de draws en workers',
        'Préparation des draws sur les workers',
        'Draw-Vorbereitung auf Workern',
        'Preparazione dei draw sui worker',
        'Przygotowanie draws w workerach',
        'Подготовка draw в рабочих потоках',
        'ワーカーでのドロー準備'),
    'Recursos preparados nos workers': (
        'Resources prepared on workers',
        'Recursos preparados en los workers',
        'Ressources préparées sur les workers',
        'Auf Workern vorbereitete Ressourcen',
        'Risorse preparate sui worker',
        'Zasoby przygotowane w workerach',
        'Ресурсы, подготовленные в рабочих потоках',
        'ワーカーで準備されたリソース'),
    'Entradas de draw adiantadas': (
        'Draw inputs ahead of time',
        'Entradas de draw adelantadas',
        'Entrées de draw anticipées',
        'Vorgezogene Draw-Eingaben',
        'Input dei draw anticipati',
        'Wcześniejsze dane wejściowe draws',
        'Опережающие входные данные draw',
        'ドロー入力の先行処理'),
    'Ajudante de texturas (causou bugs gráficos)': (
        'Texture helper (caused graphics bugs)',
        'Ayudante de texturas (causó errores gráficos)',
        'Assistant de textures (a causé des bugs graphiques)',
        'Textur-Helfer (verursachte Grafikfehler)',
        'Assistente texture (ha causato bug grafici)',
        'Pomocnik tekstur (powodował błędy grafiki)',
        'Помощник текстур (вызывал графические ошибки)',
        'テクスチャヘルパー(グラフィックバグの原因)'),
    'Prefetch de gravação': (
        'Recording prefetch',
        'Prefetch de grabación',
        "Prefetch d'enregistrement",
        'Aufzeichnungs-Prefetch',
        'Prefetch di registrazione',
        'Prefetch nagrywania',
        'Предвыборка записи',
        '記録のプリフェッチ'),
    'Uploads na thread de gravação': (
        'Uploads on the recording thread',
        'Subidas en el hilo de grabación',
        "Envois sur le thread d'enregistrement",
        'Uploads im Aufzeichnungs-Thread',
        'Upload sul thread di registrazione',
        'Przesyłanie w wątku nagrywania',
        'Загрузки в потоке записи',
        '記録スレッドでのアップロード'),
    'Cópias curtas na thread de gravação': (
        'Short copies on the recording thread',
        'Copias cortas en el hilo de grabación',
        "Copies courtes sur le thread d'enregistrement",
        'Kurze Kopien im Aufzeichnungs-Thread',
        'Copie brevi sul thread di registrazione',
        'Krótkie kopie w wątku nagrywania',
        'Короткие копии в потоке записи',
        '記録スレッドでの短いコピー'),
    'Cópias grandes em threads paralelas': (
        'Large copies on parallel threads',
        'Copias grandes en hilos paralelos',
        'Grandes copies sur des threads parallèles',
        'Große Kopien in parallelen Threads',
        'Copie grandi su thread paralleli',
        'Duże kopie w równoległych wątkach',
        'Большие копии в параллельных потоках',
        '並列スレッドでの大きなコピー'),
    'Cópias curtas em lote': (
        'Short copies in batches',
        'Copias cortas por lotes',
        'Copies courtes par lots',
        'Kurze Kopien im Stapel',
        'Copie brevi in batch',
        'Krótkie kopie w partiach',
        'Короткие копии пакетами',
        '短いコピーのバッチ処理'),
    'Fences assíncronas': (
        'Asynchronous fences',
        'Fences asíncronas',
        'Fences asynchrones',
        'Asynchrone Fences',
        'Fence asincrone',
        'Asynchroniczne fence',
        'Асинхронные fence',
        '非同期フェンス'),
    'Esperas de fence adiadas': (
        'Deferred fence waits',
        'Esperas de fence aplazadas',
        'Attentes de fence différées',
        'Verzögertes Warten auf Fences',
        'Attese di fence differite',
        'Odroczone oczekiwanie na fence',
        'Отложенное ожидание fence',
        'フェンス待機の遅延'),
    'Fences na thread Vulkan': (
        'Fences on the Vulkan thread',
        'Fences en el hilo Vulkan',
        'Fences sur le thread Vulkan',
        'Fences im Vulkan-Thread',
        'Fence sul thread Vulkan',
        'Fence w wątku Vulkan',
        'Fence в потоке Vulkan',
        'Vulkan スレッドでのフェンス'),
    'Upload sem trava': (
        'Lock-free upload',
        'Subida sin bloqueo',
        'Envoi sans verrou',
        'Upload ohne Sperre',
        'Upload senza blocco',
        'Przesyłanie bez blokady',
        'Загрузка без блокировки',
        'ロックなしアップロード'),
    'Anel de constantes': (
        'Constant ring',
        'Anillo de constantes',
        'Anneau de constantes',
        'Konstanten-Ring',
        'Anello di costanti',
        'Pierścień stałych',
        'Кольцо констант',
        '定数リング'),
    'Shader de cópia múltipla': (
        'Multi-copy shader',
        'Shader de copia múltiple',
        'Shader de copie multiple',
        'Mehrfachkopie-Shader',
        'Shader di copia multipla',
        'Shader wielokrotnego kopiowania',
        'Шейдер множественного копирования',
        'マルチコピーシェーダー'),
    'Cache de fetch shaders': (
        'Fetch shader cache',
        'Caché de fetch shaders',
        'Cache des fetch shaders',
        'Fetch-Shader-Cache',
        'Cache dei fetch shader',
        'Pamięć podręczna fetch shaderów',
        'Кэш fetch-шейдеров',
        'フェッチシェーダーキャッシュ'),
    'Cache de descritores de textura': (
        'Texture descriptor cache',
        'Caché de descriptores de textura',
        'Cache des descripteurs de texture',
        'Texturdeskriptor-Cache',
        'Cache dei descrittori di texture',
        'Pamięć podręczna deskryptorów tekstur',
        'Кэш дескрипторов текстур',
        'テクスチャ記述子キャッシュ'),
    'Cache de busca de imagens': (
        'Image lookup cache',
        'Caché de búsqueda de imágenes',
        "Cache de recherche d'images",
        'Bildsuch-Cache',
        'Cache di ricerca delle immagini',
        'Pamięć podręczna wyszukiwania obrazów',
        'Кэш поиска изображений',
        '画像検索キャッシュ'),
    'Memo de barreiras': (
        'Barrier memo',
        'Memo de barreras',
        'Mémo des barrières',
        'Barrieren-Memo',
        'Memo delle barriere',
        'Memo barier',
        'Мемо барьеров',
        'バリアのメモ'),
    'Memo de bind de texturas': (
        'Texture bind memo',
        'Memo de bind de texturas',
        'Mémo des binds de textures',
        'Textur-Bind-Memo',
        'Memo dei bind delle texture',
        'Memo wiązania tekstur',
        'Мемо привязки текстур',
        'テクスチャバインドのメモ'),
    'Memo de views de textura': (
        'Texture view memo',
        'Memo de vistas de textura',
        'Mémo des vues de texture',
        'Texturansicht-Memo',
        'Memo delle view delle texture',
        'Memo widoków tekstur',
        'Мемо представлений текстур',
        'テクスチャビューのメモ'),
    'Memo de estado de render': (
        'Render state memo',
        'Memo de estado de render',
        "Mémo de l'état de rendu",
        'Renderzustand-Memo',
        'Memo dello stato di rendering',
        'Memo stanu renderowania',
        'Мемо состояния рендера',
        'レンダー状態のメモ'),
    'Memo de conjuntos de textura': (
        'Texture set memo',
        'Memo de conjuntos de textura',
        'Mémo des ensembles de textures',
        'Textursatz-Memo',
        'Memo dei set di texture',
        'Memo zestawów tekstur',
        'Мемо наборов текстур',
        'テクスチャセットのメモ'),
    'Caminho rápido de UpdateImage': (
        'UpdateImage fast path',
        'Ruta rápida de UpdateImage',
        "Chemin rapide d'UpdateImage",
        'Schneller UpdateImage-Pfad',
        'Percorso rapido di UpdateImage',
        'Szybka ścieżka UpdateImage',
        'Быстрый путь UpdateImage',
        'UpdateImage の高速パス'),
    'Pipeline de draws': (
        'Draw pipeline',
        'Pipeline de draws',
        'Pipeline de draws',
        'Draw-Pipeline',
        'Pipeline dei draw',
        'Potok draws',
        'Конвейер draw',
        'ドローパイプライン'),
    'Prioridade alta nas threads da GPU': (
        'High priority on GPU threads',
        'Prioridad alta en los hilos de la GPU',
        'Priorité élevée pour les threads GPU',
        'Hohe Priorität für GPU-Threads',
        'Priorità alta sui thread GPU',
        'Wysoki priorytet wątków GPU',
        'Высокий приоритет потоков GPU',
        'GPU スレッドの高優先度'),
    'Prioridades das threads do jogo': (
        "Game thread priorities",
        'Prioridades de los hilos del juego',
        'Priorités des threads du jeu',
        'Prioritäten der Spiel-Threads',
        'Priorità dei thread del gioco',
        'Priorytety wątków gry',
        'Приоритеты потоков игры',
        'ゲームスレッドの優先度'),
    'Quadros em andamento': (
        'Frames in flight',
        'Fotogramas en curso',
        'Images en cours',
        'Frames in Bearbeitung',
        'Fotogrammi in corso',
        'Klatki w toku',
        'Кадры в обработке',
        '処理中のフレーム'),
    'Menos esperas entre estágios': (
        'Fewer waits between stages',
        'Menos esperas entre etapas',
        "Moins d'attentes entre étages",
        'Weniger Wartezeiten zwischen Stufen',
        'Meno attese tra gli stadi',
        'Mniej oczekiwania między etapami',
        'Меньше ожиданий между этапами',
        'ステージ間の待機を削減'),
    'Busca de dados sem bloquear': (
        'Non-blocking data fetch',
        'Búsqueda de datos sin bloqueo',
        'Lecture de données non bloquante',
        'Datenabruf ohne Blockieren',
        'Lettura dei dati non bloccante',
        'Pobieranie danych bez blokowania',
        'Получение данных без блокировки',
        'ブロックしないデータ取得'),
    'Timer de 1 ms': (
        '1 ms timer',
        'Temporizador de 1 ms',
        'Minuteur de 1 ms',
        '1-ms-Timer',
        'Timer da 1 ms',
        'Zegar 1 ms',
        'Таймер 1 мс',
        '1 ms タイマー'),
    'Sem limitação de energia': (
        'No power throttling',
        'Sin limitación de energía',
        "Sans limitation d'énergie",
        'Keine Energiedrosselung',
        'Nessuna limitazione energetica',
        'Bez ograniczania energii',
        'Без ограничения питания',
        '電力制限なし'),
    'Condition variables rápidas': (
        'Fast condition variables',
        'Condition variables rápidas',
        'Condition variables rapides',
        'Schnelle Condition Variables',
        'Condition variable veloci',
        'Szybkie zmienne warunkowe',
        'Быстрые условные переменные',
        '高速条件変数'),
    'Reaproveitar buffers de draws': (
        'Reuse draw buffers',
        'Reutilizar buffers de draws',
        'Réutiliser les tampons de draws',
        'Draw-Puffer wiederverwenden',
        'Riutilizza i buffer dei draw',
        'Ponowne użycie buforów draws',
        'Повторное использование буферов draw',
        'ドローバッファの再利用'),
    'Economias por draw': (
        'Per-draw savings',
        'Ahorros por draw',
        'Économies par draw',
        'Einsparungen pro Draw',
        'Risparmi per draw',
        'Oszczędności na draw',
        'Экономия на draw',
        'ドローごとの節約'),
    'Escritas ao lado de dados da GPU sem espera': (
        'Writes beside GPU data without waiting',
        'Escrituras junto a datos de la GPU sin espera',
        'Écritures à côté des données GPU sans attente',
        'Schreiben neben GPU-Daten ohne Warten',
        'Scritture accanto ai dati GPU senza attesa',
        'Zapisy obok danych GPU bez czekania',
        'Запись рядом с данными GPU без ожидания',
        'GPU データ隣接の書き込みを待機なしで'),
    'Resultados de DMA no fim do pipe': (
        'DMA results at end of pipe',
        'Resultados de DMA al final del pipe',
        'Résultats DMA en fin de pipe',
        'DMA-Ergebnisse am Pipe-Ende',
        'Risultati DMA a fine pipe',
        'Wyniki DMA na końcu potoku',
        'Результаты DMA в конце конвейера',
        'パイプ終端での DMA 結果'),
    'Cópia de imagens minúsculas (foco da mira)': (
        'Tiny image copy (aim focus)',
        'Copia de imágenes diminutas (enfoque de la mira)',
        'Copie des toutes petites images (mise au point de la visée)',
        'Kopie winziger Bilder (Zielfokus)',
        'Copia di immagini minuscole (messa a fuoco della mira)',
        'Kopia maleńkich obrazów (ostrość celownika)',
        'Копия крошечных изображений (фокус прицела)',
        '極小画像のコピー(照準フォーカス)'),
    'Leitura das cópias da GPU (NPCs pulando)': (
        'GPU copy readback (NPCs jumping)',
        'Lectura de las copias de la GPU (NPC saltando)',
        'Relecture des copies GPU (PNJ qui sautent)',
        'Rücklesen der GPU-Kopien (springende NPCs)',
        'Rilettura delle copie GPU (NPC che saltano)',
        'Odczyt kopii z GPU (skaczący NPC)',
        'Обратное чтение копий GPU (прыгающие NPC)',
        'GPU コピーの読み戻し (NPC が跳ねる)'),
    'Juntas do tecido em todos os tamanhos': (
        'Cloth joints for every cloth size',
        'Articulaciones de la tela en todos los tamaños',
        'Articulations du tissu pour toutes les tailles',
        'Stoffgelenke für alle Stoffgrößen',
        'Giunti del tessuto per tutte le dimensioni',
        'Stawy tkaniny we wszystkich rozmiarach',
        'Суставы ткани для всех размеров',
        'すべてのサイズの布のジョイント'),
    'Acesso direto à memória': (
        'Direct memory access',
        'Acceso directo a la memoria',
        'Accès direct à la mémoire',
        'Direkter Speicherzugriff',
        'Accesso diretto alla memoria',
        'Bezpośredni dostęp do pamięci',
        'Прямой доступ к памяти',
        'ダイレクトメモリアクセス'),
    'TSC nativo': (
        'Native TSC',
        'TSC nativo',
        'TSC natif',
        'Natives TSC',
        'TSC nativo',
        'Natywny TSC',
        'Нативный TSC',
        'ネイティブ TSC'),
    'Buffer grande de staging reservado': (
        'Large reserved staging buffer',
        'Búfer grande de staging reservado',
        'Grand tampon de staging réservé',
        'Großer reservierter Staging-Puffer',
        'Buffer di staging grande riservato',
        'Duży zarezerwowany bufor staging',
        'Большой зарезервированный staging-буфер',
        '予約済みの大きなステージングバッファ'),
    'Cache de pipelines': (
        'Pipeline cache',
        'Caché de pipelines',
        'Cache des pipelines',
        'Pipeline-Cache',
        'Cache delle pipeline',
        'Pamięć podręczna potoków',
        'Кэш конвейеров',
        'パイプラインキャッシュ'),
    'Binários de pipeline': (
        'Pipeline binaries',
        'Binarios de pipeline',
        'Binaires de pipeline',
        'Pipeline-Binärdateien',
        'Binari delle pipeline',
        'Pliki binarne potoków',
        'Двоичные файлы конвейеров',
        'パイプラインバイナリ'),
    'Prefetch de pipelines': (
        'Pipeline prefetch',
        'Prefetch de pipelines',
        'Prefetch des pipelines',
        'Pipeline-Prefetch',
        'Prefetch delle pipeline',
        'Prefetch potoków',
        'Предвыборка конвейеров',
        'パイプラインのプリフェッチ'),
    'Proteção da red zone': (
        'Red zone protection',
        'Protección de la red zone',
        'Protection de la red zone',
        'Red-Zone-Schutz',
        'Protezione della red zone',
        'Ochrona red zone',
        'Защита red zone',
        'レッドゾーン保護'),
    'Correção da chuva branca': (
        'White rain fix',
        'Corrección de la lluvia blanca',
        'Correction de la pluie blanche',
        'Korrektur des weißen Regens',
        'Correzione della pioggia bianca',
        'Poprawka białego deszczu',
        'Исправление белого дождя',
        '白い雨の修正'),
    'Contorno do tecido na GTX 900/10 (evita crash)': (
        'Cloth workaround on GTX 900/10 (avoids a crash)',
        'Solución de la tela en GTX 900/10 (evita un cierre)',
        'Contournement du tissu sur GTX 900/10 (évite un plantage)',
        'Stoff-Umgehung auf GTX 900/10 (verhindert Absturz)',
        'Aggiramento del tessuto su GTX 900/10 (evita un crash)',
        'Obejście tkanin na GTX 900/10 (zapobiega awarii)',
        'Обход ткани на GTX 900/10 (предотвращает вылет)',
        'GTX 900/10 の布の回避策 (クラッシュ防止)'),
    'Tela': (
        'Screen',
        'Pantalla',
        'Écran',
        'Bildschirm',
        'Schermo',
        'Ekran',
        'Экран',
        '画面'),
    'Camadas do HUD até as bordas (ultrawide)': (
        'HUD layers reach the screen edges (ultrawide)',
        'Capas del HUD hasta los bordes (ultrawide)',
        "Calques de l'interface jusqu'aux bords (ultrawide)",
        'HUD-Ebenen bis zum Bildrand (Ultrawide)',
        "Livelli dell'HUD fino ai bordi (ultrawide)",
        'Warstwy HUD do krawędzi ekranu (ultrawide)',
        'Слои HUD до краёв экрана (ultrawide)',
        'HUDのレイヤーを画面端まで表示 (ウルトラワイド)'),
}

# Controls window: the keyboard and mouse maps the game reads from tlou.ini (key_map, mouse_map:
# "<action>=<SDL key or mouse name>,...", gow3_pc/src/runtime_pad.c apply_button_map), written
# complete like the in-game menu's Controls tab. (group, [(action, label, keys only)]).
CONTROLS = [
    ("Movimento", [("ls_up", "Andar para frente", True), ("ls_down", "Andar para trás", True),
                   ("ls_left", "Andar para a esquerda", True), ("ls_right", "Andar para a direita", True),
                   ("walk", "Andar devagar", True)]),
    ("Câmera", [("rs_up", "Olhar para cima", True), ("rs_down", "Olhar para baixo", True),
                ("rs_left", "Olhar para a esquerda", True), ("rs_right", "Olhar para a direita", True)]),
    ("Botões", [("cross", "Xis (×)", False), ("circle", "Círculo (○)", False), ("square", "Quadrado (□)", False),
                ("triangle", "Triângulo (△)", False), ("l1", "L1", False), ("r1", "R1", False),
                ("l2", "L2 (mirar)", False), ("r2", "R2 (atirar)", False), ("l3", "L3", False), ("r3", "R3", False),
                ("options", "Options (pausa)", False), ("touchpad_left", "Touchpad, lado esquerdo", False),
                ("touchpad_right", "Touchpad, lado direito", False), ("shake", "Agitar o controle (lanterna)", False)]),
    ("Direcional", [("up", "Cima", False), ("down", "Baixo", False), ("left", "Esquerda", False),
                    ("right", "Direita", False)]),
]
# runtime_pad.c's defaults: [(kind, name)], kind "key" (SDL scancode name) or "mouse".
DEFAULT_CONTROLS = {
    "ls_up": [("key", "W"), ("key", "Up")], "ls_down": [("key", "S"), ("key", "Down")],
    "ls_left": [("key", "A"), ("key", "Left")], "ls_right": [("key", "D"), ("key", "Right")],
    "walk": [("key", "Left Ctrl")],
    "rs_up": [], "rs_down": [], "rs_left": [], "rs_right": [],
    "cross": [("key", "Space"), ("key", "Return")], "circle": [("key", "C")], "square": [("key", "F")],
    "triangle": [("key", "E")], "l1": [("key", "Left Shift")], "r1": [("key", "Tab")], "l2": [("mouse", "right")],
    "r2": [("mouse", "left")], "l3": [("key", "V")], "r3": [("key", "T")],
    "options": [("key", "Escape"), ("key", "Backspace")], "touchpad_left": [("key", "CapsLock")],
    "touchpad_right": [], "shake": [("key", "G")],
    "up": [("key", "3")], "down": [("key", "4")], "left": [("key", "1"), ("mouse", "wheel_down")],
    "right": [("key", "2"), ("mouse", "wheel_up")],
}
CONTROL_SLOTS = 2   # sources per action, as in the in-game menu
MOUSE_NAMES = {"left": "Botão esquerdo do mouse", "right": "Botão direito do mouse", "middle": "Botão do meio do mouse",
               "x1": "Botão lateral 1 do mouse", "x2": "Botão lateral 2 do mouse",
               "wheel_up": "Roda do mouse para cima", "wheel_down": "Roda do mouse para baixo"}
MOUSE_BUTTONS = {1: "left", 2: "middle", 3: "right", 4: "x1", 5: "x2", 8: "x1", 9: "x2"}   # Tk's numbers
# Keys by Windows scan code (set 1), as SDL names them; "," cannot be bound (it separates the entries).
SCAN_NAMES = {0x01: "Escape", 0x0C: "-", 0x0D: "=", 0x0E: "Backspace", 0x0F: "Tab", 0x1A: "[", 0x1B: "]",
              0x1C: "Return", 0x1D: "Left Ctrl", 0x27: ";", 0x28: "'", 0x29: "`", 0x2A: "Left Shift", 0x2B: "\\",
              0x34: ".", 0x35: "/", 0x36: "Right Shift", 0x37: "Keypad *", 0x38: "Left Alt", 0x39: "Space",
              0x3A: "CapsLock", 0x45: "Numlock", 0x46: "ScrollLock", 0x4A: "Keypad -", 0x4E: "Keypad +",
              0x53: "Keypad .", 0x57: "F11", 0x58: "F12"}
SCAN_NAMES.update({0x02 + i: str((i + 1) % 10) for i in range(10)})
SCAN_NAMES.update({0x10 + i: c for i, c in enumerate("QWERTYUIOP")})
SCAN_NAMES.update({0x1E + i: c for i, c in enumerate("ASDFGHJKL")})
SCAN_NAMES.update({0x2C + i: c for i, c in enumerate("ZXCVBNM")})
SCAN_NAMES.update({0x3B + i: f"F{i + 1}" for i in range(10)})
SCAN_NAMES.update({scan: f"Keypad {n}" for scan, n in zip((0x47, 0x48, 0x49, 0x4B, 0x4C, 0x4D, 0x4F, 0x50, 0x51, 0x52),
                                                          "7894561230")})
# Keys Tk tells apart by name (sides, the extended keys); the rest go by scan code.
KEYSYM_NAMES = {"Control_L": "Left Ctrl", "Control_R": "Right Ctrl", "Shift_L": "Left Shift", "Shift_R": "Right Shift",
                "Alt_L": "Left Alt", "Alt_R": "Right Alt", "Win_L": "Left GUI", "Win_R": "Right GUI", "App": "Application",
                "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right", "Insert": "Insert", "Home": "Home",
                "End": "End", "Prior": "PageUp", "Next": "PageDown", "KP_Enter": "Keypad Enter", "KP_Divide": "Keypad /",
                "Pause": "Pause", "Print": "PrintScreen"}
KEY_LABELS = {"Space": "Espaço", "Left Ctrl": "Ctrl esquerdo", "Right Ctrl": "Ctrl direito", "Left Shift": "Shift esquerdo",
              "Right Shift": "Shift direito", "Left Alt": "Alt esquerdo", "Right Alt": "Alt direito",
              "Up": "↑", "Down": "↓", "Left": "←", "Right": "→", "Return": "Enter", "Escape": "Esc"}
CONTROL_TEXT = {
    'Teclado e mouse': ('Keyboard and mouse', 'Teclado y ratón', 'Clavier et souris', 'Tastatur und Maus',
                        'Tastiera e mouse', 'Klawiatura i mysz', 'Клавиатура и мышь', 'キーボードとマウス'),
    'Mouse': ('Mouse', 'Ratón', 'Souris', 'Maus', 'Mouse', 'Mysz', 'Мышь', 'マウス'),
    'Clique numa caixa e pressione a tecla ou o botão do mouse. Esc cancela, Delete limpa.\n'
    'Movimento e câmera aceitam só teclas. Vale na próxima vez que o jogo abrir.': (
        'Click a box and press the key or mouse button. Esc cancels, Delete clears.\n'
        'Movement and camera take keys only. Takes effect the next time the game starts.',
        'Haz clic en una casilla y pulsa la tecla o el botón del ratón. Esc cancela, Supr borra.\n'
        'Movimiento y cámara solo aceptan teclas. Se aplica la próxima vez que se abra el juego.',
        'Cliquez sur une case puis appuyez sur la touche ou le bouton de la souris. Échap annule, Suppr efface.\n'
        'Déplacement et caméra : touches uniquement. S\'applique au prochain lancement du jeu.',
        'Feld anklicken, dann Taste oder Maustaste drücken. Esc bricht ab, Entf löscht.\n'
        'Bewegung und Kamera nur mit Tasten. Gilt beim nächsten Spielstart.',
        'Fai clic su una casella e premi il tasto o il pulsante del mouse. Esc annulla, Canc cancella.\n'
        'Movimento e telecamera accettano solo tasti. Vale al prossimo avvio del gioco.',
        'Kliknij pole i naciśnij klawisz lub przycisk myszy. Esc anuluje, Delete czyści.\n'
        'Ruch i kamera tylko z klawiatury. Działa od następnego uruchomienia gry.',
        'Щёлкните поле и нажмите клавишу или кнопку мыши. Esc — отмена, Delete — очистить.\n'
        'Движение и камера — только клавиши. Действует при следующем запуске игры.',
        '枠をクリックしてキーかマウスボタンを押してください。Esc で取り消し、Delete で解除。\n'
        '移動とカメラはキーのみ。次回ゲーム起動時に反映されます。'),
    'Vale na próxima vez que o jogo abrir.': (
        'Takes effect the next time the game starts.', 'Se aplica la próxima vez que se abra el juego.',
        'S\'applique au prochain lancement du jeu.', 'Gilt beim nächsten Spielstart.',
        'Vale al prossimo avvio del gioco.', 'Działa od następnego uruchomienia gry.',
        'Действует при следующем запуске игры.', '次回ゲーム起動時に反映されます。'),
    'Pressione...': ('Press...', 'Pulsa...', 'Appuyez...', 'Taste drücken...', 'Premi...', 'Naciśnij...',
                     'Нажмите...', '入力待ち...'),
    'Movimento': ('Movement', 'Movimiento', 'Déplacement', 'Bewegung', 'Movimento', 'Ruch', 'Движение', '移動'),
    'Câmera': ('Camera', 'Cámara', 'Caméra', 'Kamera', 'Telecamera', 'Kamera', 'Камера', 'カメラ'),
    'Botões': ('Buttons', 'Botones', 'Boutons', 'Tasten', 'Pulsanti', 'Przyciski', 'Кнопки', 'ボタン'),
    'Direcional': ('D-pad', 'Cruceta', 'Croix directionnelle', 'Steuerkreuz', 'Croce direzionale', 'Krzyżak',
                   'Крестовина', '方向キー'),
    'Andar para frente': ('Move forward', 'Avanzar', 'Avancer', 'Vorwärts', 'Avanti', 'Do przodu', 'Вперёд', '前進'),
    'Andar para trás': ('Move back', 'Retroceder', 'Reculer', 'Rückwärts', 'Indietro', 'Do tyłu', 'Назад', '後退'),
    'Andar para a esquerda': ('Move left', 'Moverse a la izquierda', 'Aller à gauche', 'Nach links', 'Sinistra',
                              'W lewo', 'Влево', '左へ移動'),
    'Andar para a direita': ('Move right', 'Moverse a la derecha', 'Aller à droite', 'Nach rechts', 'Destra',
                             'W prawo', 'Вправо', '右へ移動'),
    'Andar devagar': ('Walk', 'Caminar', 'Marcher', 'Gehen', 'Camminare', 'Chód', 'Шагом', '歩く'),
    'Tecla de andar': ('Walk key', 'Tecla de caminar', 'Touche de marche', 'Geh-Taste', 'Tasto per camminare',
                       'Klawisz chodu', 'Клавиша ходьбы', '歩きキー'),
    'Segurar a tecla': ('Hold the key', 'Mantener la tecla', 'Maintenir la touche', 'Taste halten',
                        'Tieni premuto il tasto', 'Przytrzymaj klawisz', 'Удерживать клавишу', 'キーを押し続ける'),
    'Alternar ao apertar': ('Toggle on press', 'Alternar al pulsar', 'Basculer à chaque appui', 'Bei Druck umschalten',
                            'Alterna alla pressione', 'Przełącz naciśnięciem', 'Переключать нажатием',
                            '押すたびに切り替え'),
    'Velocidade ao andar (quanto o analógico esquerdo vai)': (
        'Walk speed (how far the left stick goes)', 'Velocidad al caminar (cuánto se mueve el stick izquierdo)',
        'Vitesse de marche (course du stick gauche)', 'Gehtempo (wie weit der linke Stick geht)',
        'Velocità camminando (quanto va lo stick sinistro)', 'Prędkość chodu (wychylenie lewej gałki)',
        'Скорость шага (отклонение левого стика)', '歩く速さ (左スティックの倒し具合)'),
    'Olhar para cima': ('Look up', 'Mirar arriba', 'Regarder en haut', 'Nach oben schauen', 'Guarda su',
                        'Patrz w górę', 'Смотреть вверх', '上を見る'),
    'Olhar para baixo': ('Look down', 'Mirar abajo', 'Regarder en bas', 'Nach unten schauen', 'Guarda giù',
                         'Patrz w dół', 'Смотреть вниз', '下を見る'),
    'Olhar para a esquerda': ('Look left', 'Mirar a la izquierda', 'Regarder à gauche', 'Nach links schauen',
                              'Guarda a sinistra', 'Patrz w lewo', 'Смотреть влево', '左を見る'),
    'Olhar para a direita': ('Look right', 'Mirar a la derecha', 'Regarder à droite', 'Nach rechts schauen',
                             'Guarda a destra', 'Patrz w prawo', 'Смотреть вправо', '右を見る'),
    'Xis (×)': ('Cross (×)', 'Equis (×)', 'Croix (×)', 'Kreuz (×)', 'Croce (×)', 'Krzyżyk (×)', 'Крест (×)', 'バツ (×)'),
    'Círculo (○)': ('Circle (○)', 'Círculo (○)', 'Rond (○)', 'Kreis (○)', 'Cerchio (○)', 'Kółko (○)', 'Круг (○)',
                    'マル (○)'),
    'Quadrado (□)': ('Square (□)', 'Cuadrado (□)', 'Carré (□)', 'Quadrat (□)', 'Quadrato (□)', 'Kwadrat (□)',
                     'Квадрат (□)', 'シカク (□)'),
    'Triângulo (△)': ('Triangle (△)', 'Triángulo (△)', 'Triangle (△)', 'Dreieck (△)', 'Triangolo (△)',
                      'Trójkąt (△)', 'Треугольник (△)', 'サンカク (△)'),
    'L2 (mirar)': ('L2 (aim)', 'L2 (apuntar)', 'L2 (viser)', 'L2 (zielen)', 'L2 (mirare)', 'L2 (celowanie)',
                   'L2 (прицел)', 'L2 (構える)'),
    'R2 (atirar)': ('R2 (fire)', 'R2 (disparar)', 'R2 (tirer)', 'R2 (schießen)', 'R2 (sparare)', 'R2 (strzał)',
                    'R2 (выстрел)', 'R2 (撃つ)'),
    'Options (pausa)': ('Options (pause)', 'Options (pausa)', 'Options (pause)', 'Options (Pause)', 'Options (pausa)',
                        'Options (pauza)', 'Options (пауза)', 'OPTIONS (ポーズ)'),
    'Touchpad, lado esquerdo': ('Touchpad, left side', 'Panel táctil, lado izquierdo', 'Pavé tactile, côté gauche',
                                'Touchpad, linke Seite', 'Touchpad, lato sinistro', 'Touchpad, lewa strona',
                                'Сенсорная панель, слева', 'タッチパッド左'),
    'Touchpad, lado direito': ('Touchpad, right side', 'Panel táctil, lado derecho', 'Pavé tactile, côté droit',
                               'Touchpad, rechte Seite', 'Touchpad, lato destro', 'Touchpad, prawa strona',
                               'Сенсорная панель, справа', 'タッチパッド右'),
    'Agitar o controle (lanterna)': ('Shake the controller (flashlight)', 'Agitar el mando (linterna)',
                                     'Secouer la manette (lampe)', 'Controller schütteln (Taschenlampe)',
                                     'Scuoti il controller (torcia)', 'Potrząśnij padem (latarka)',
                                     'Встряхнуть геймпад (фонарик)', 'コントローラーを振る (ライト)'),
    'Cima': ('Up', 'Arriba', 'Haut', 'Oben', 'Su', 'Góra', 'Вверх', '上'),
    'Baixo': ('Down', 'Abajo', 'Bas', 'Unten', 'Giù', 'Dół', 'Вниз', '下'),
    'Esquerda': ('Left', 'Izquierda', 'Gauche', 'Links', 'Sinistra', 'Lewo', 'Влево', '左'),
    'Direita': ('Right', 'Derecha', 'Droite', 'Rechts', 'Destra', 'Prawo', 'Вправо', '右'),
    'Usar o mouse': ('Use the mouse', 'Usar el ratón', 'Utiliser la souris', 'Maus verwenden', 'Usa il mouse',
                     'Używaj myszy', 'Использовать мышь', 'マウスを使う'),
    'Sensibilidade do mouse': ('Mouse sensitivity', 'Sensibilidad del ratón', 'Sensibilité de la souris',
                               'Mausempfindlichkeit', 'Sensibilità del mouse', 'Czułość myszy',
                               'Чувствительность мыши', 'マウス感度'),
    'Sensibilidade ao mirar': ('Aim sensitivity', 'Sensibilidad al apuntar', 'Sensibilité en visée',
                               'Empfindlichkeit beim Zielen', 'Sensibilità in mira', 'Czułość podczas celowania',
                               'Чувствительность при прицеливании', 'エイム時の感度'),
    'Inverter eixo vertical': ('Invert vertical axis', 'Invertir eje vertical', "Inverser l'axe vertical",
                               'Vertikale Achse umkehren', 'Inverti asse verticale', 'Odwróć oś pionową',
                               'Инвертировать вертикальную ось', '上下を反転'),
    'Inverter eixo horizontal': ('Invert horizontal axis', 'Invertir eje horizontal', "Inverser l'axe horizontal",
                                 'Horizontale Achse umkehren', 'Inverti asse orizzontale', 'Odwróć oś poziomą',
                                 'Инвертировать горизонтальную ось', '左右を反転'),
    'Espaço': ('Space', 'Espacio', 'Espace', 'Leertaste', 'Spazio', 'Spacja', 'Пробел', 'スペース'),
    'Ctrl esquerdo': ('Left Ctrl', 'Ctrl izquierdo', 'Ctrl gauche', 'Strg links', 'Ctrl sinistro', 'Lewy Ctrl',
                      'Левый Ctrl', '左Ctrl'),
    'Ctrl direito': ('Right Ctrl', 'Ctrl derecho', 'Ctrl droit', 'Strg rechts', 'Ctrl destro', 'Prawy Ctrl',
                     'Правый Ctrl', '右Ctrl'),
    'Shift esquerdo': ('Left Shift', 'Mayús izquierda', 'Maj gauche', 'Umschalt links', 'Maiusc sinistro',
                       'Lewy Shift', 'Левый Shift', '左Shift'),
    'Shift direito': ('Right Shift', 'Mayús derecha', 'Maj droite', 'Umschalt rechts', 'Maiusc destro', 'Prawy Shift',
                      'Правый Shift', '右Shift'),
    'Alt esquerdo': ('Left Alt', 'Alt izquierdo', 'Alt gauche', 'Alt links', 'Alt sinistro', 'Lewy Alt', 'Левый Alt',
                     '左Alt'),
    'Alt direito': ('Right Alt', 'Alt derecho', 'Alt droit', 'Alt rechts', 'Alt destro', 'Prawy Alt', 'Правый Alt',
                    '右Alt'),
    'Botão esquerdo do mouse': ('Left mouse button', 'Botón izquierdo del ratón', 'Bouton gauche de la souris',
                                'Linke Maustaste', 'Tasto sinistro del mouse', 'Lewy przycisk myszy',
                                'Левая кнопка мыши', 'マウス左ボタン'),
    'Botão direito do mouse': ('Right mouse button', 'Botón derecho del ratón', 'Bouton droit de la souris',
                               'Rechte Maustaste', 'Tasto destro del mouse', 'Prawy przycisk myszy',
                               'Правая кнопка мыши', 'マウス右ボタン'),
    'Botão do meio do mouse': ('Middle mouse button', 'Botón central del ratón', 'Bouton du milieu de la souris',
                               'Mittlere Maustaste', 'Tasto centrale del mouse', 'Środkowy przycisk myszy',
                               'Средняя кнопка мыши', 'マウス中ボタン'),
    'Botão lateral 1 do mouse': ('Mouse side button 1', 'Botón lateral 1 del ratón', 'Bouton latéral 1 de la souris',
                                 'Maus-Seitentaste 1', 'Tasto laterale 1 del mouse', 'Boczny przycisk myszy 1',
                                 'Боковая кнопка мыши 1', 'マウスサイドボタン1'),
    'Botão lateral 2 do mouse': ('Mouse side button 2', 'Botón lateral 2 del ratón', 'Bouton latéral 2 de la souris',
                                 'Maus-Seitentaste 2', 'Tasto laterale 2 del mouse', 'Boczny przycisk myszy 2',
                                 'Боковая кнопка мыши 2', 'マウスサイドボタン2'),
    'Roda do mouse para cima': ('Mouse wheel up', 'Rueda del ratón arriba', 'Molette vers le haut', 'Mausrad hoch',
                                'Rotella su', 'Kółko myszy w górę', 'Колесо мыши вверх', 'ホイール上'),
    'Roda do mouse para baixo': ('Mouse wheel down', 'Rueda del ratón abajo', 'Molette vers le bas', 'Mausrad runter',
                                 'Rotella giù', 'Kółko myszy w dół', 'Колесо мыши вниз', 'ホイール下'),
    'Controle': ('Controller', 'Mando', 'Manette', 'Controller', 'Controller', 'Pad', 'Геймпад', 'コントローラー'),
    'Mexa no controle ou aperte as teclas e os botões do mouse configurados: o desenho mostra cada comando.': (
        'Move the controller or press the bound keys and mouse buttons: the drawing shows each input.',
        'Mueve el mando o pulsa las teclas y botones del ratón asignados: el dibujo muestra cada entrada.',
        'Bougez la manette ou appuyez sur les touches et boutons de souris assignés : le dessin montre chaque commande.',
        'Bewege den Controller oder drücke die belegten Tasten und Maustasten: die Zeichnung zeigt jede Eingabe.',
        'Muovi il controller o premi i tasti e i pulsanti del mouse assegnati: il disegno mostra ogni comando.',
        'Porusz padem lub naciśnij przypisane klawisze i przyciski myszy: rysunek pokazuje każde wejście.',
        'Подвигайте геймпад или нажмите назначенные клавиши и кнопки мыши: рисунок показывает каждое нажатие.',
        'コントローラーを動かすか、割り当てたキーやマウスボタンを押すと、図に入力が表示されます。'),
    'Controle conectado: {}': ('Controller connected: {}', 'Mando conectado: {}', 'Manette connectée : {}',
                               'Controller verbunden: {}', 'Controller collegato: {}', 'Pad podłączony: {}',
                               'Геймпад подключён: {}', 'コントローラー接続中: {}'),
    'Nenhum controle conectado.': ('No controller connected.', 'Ningún mando conectado.', 'Aucune manette connectée.',
                                   'Kein Controller verbunden.', 'Nessun controller collegato.',
                                   'Brak podłączonego pada.', 'Геймпад не подключён.',
                                   'コントローラーが接続されていません。'),
    'Leitura do controle indisponível (SDL3.dll não encontrado).': (
        'Controller reading unavailable (SDL3.dll not found).',
        'Lectura del mando no disponible (no se encontró SDL3.dll).',
        'Lecture de la manette indisponible (SDL3.dll introuvable).',
        'Controller kann nicht gelesen werden (SDL3.dll nicht gefunden).',
        'Lettura del controller non disponibile (SDL3.dll non trovato).',
        'Odczyt pada niedostępny (nie znaleziono SDL3.dll).',
        'Чтение геймпада недоступно (SDL3.dll не найден).',
        'コントローラーを読み取れません (SDL3.dll が見つかりません)。'),
    'Vibração do controle': ('Controller vibration', 'Vibración del mando', 'Vibration de la manette',
                             'Controller-Vibration', 'Vibrazione del controller', 'Wibracje pada',
                             'Вибрация геймпада', 'コントローラーの振動'),
    'Vibração': ('Vibration', 'Vibración', 'Vibration', 'Vibration', 'Vibrazione', 'Wibracje', 'Вибрация', '振動'),
    'Testar vibração': ('Test vibration', 'Probar vibración', 'Tester la vibration', 'Vibration testen',
                        'Prova vibrazione', 'Testuj wibracje', 'Проверить вибрацию', '振動をテスト'),
    'Mira': ('Aiming', 'Puntería', 'Visée', 'Zielen', 'Mira', 'Celowanie', 'Прицеливание', '照準'),
    'Assistência de mira': ('Aim assist', 'Asistencia de apuntado', 'Aide à la visée', 'Zielhilfe',
                            'Assistenza alla mira', 'Wspomaganie celowania', 'Помощь в прицеливании',
                            'エイムアシスト'),
    'Desligada: o jogo não ajuda a mira a seguir os inimigos. Vale na próxima vez que o jogo abrir.': (
        'Off: the game does not help the crosshair follow enemies. Applies the next time the game starts.',
        'Desactivada: el juego no ayuda a la mira a seguir a los enemigos. Se aplica la próxima vez que se abra '
        'el juego.',
        "Désactivée : le jeu n'aide pas le réticule à suivre les ennemis. S'applique au prochain lancement du jeu.",
        'Aus: Das Spiel hilft dem Fadenkreuz nicht, Gegnern zu folgen. Gilt ab dem nächsten Spielstart.',
        'Disattivata: il gioco non aiuta il mirino a seguire i nemici. Vale dal prossimo avvio del gioco.',
        'Wyłączone: gra nie pomaga celownikowi podążać za wrogami. Działa od następnego uruchomienia gry.',
        'Выкл.: игра не помогает прицелу следовать за врагами. Действует со следующего запуска игры.',
        'オフ: 照準が敵を追うのをゲームが補助しません。次回のゲーム起動時から有効です。'),
}

# The art is the player's own dump's (sce_sys), read at run time: no game file in the repository.
ART = ["sce_sys/pic1.png", "sce_sys/pic0.png"]
# Folders the player can change; empty is the default. {key: (text key, default)}.
FOLDERS = {"game": ("game_folder", GAME), "saves": ("saves_folder", USER), "cache": ("cache_folder", USER / "cache")}
UI_LANGUAGES = {"pt": "Português", "en": "English", "es": "Español", "fr": "Français", "de": "Deutsch",
                "it": "Italiano", "pl": "Polski", "ru": "Русский", "ja": "日本語"}
KEYS = ("subtitle video game language ui_language display resolution fps upscaler quality vsync show_fps "
        "motion_blur job_sleep profile play running starting ready debug log no_game no_exe borderless fullscreen "
        "windowed off q_native q_quality q_balanced q_performance q_ultra warn tab_live tab_startup note_live "
        "note_startup reset folders game_folder saves_folder cache_folder browse default_folder folders_note "
        "no_eboot background_shaders controls pro_shadows").split()
TEXT_TABLE = {
    "pt": ["Port para PC", "Vídeo", "Jogo", "Idioma", "Idioma do launcher", "Modo de tela", "Resolução",
           "Limite de FPS", "Upscaler", "Qualidade do upscaler", "VSync", "Mostrar FPS no jogo", "Motion blur",
           "Reduzir uso de CPU ocioso", "Gravar perfil de desempenho (cerca de 10% menos FPS)", "JOGAR", "EM EXECUÇÃO",
           "INICIANDO...", "Pronto", "Debug", "Abrir log", "Jogo não encontrado em {}",
           "Executável não encontrado: rode gow3_pc/build.sh", "Sem borda", "Tela cheia", "Janela", "Desligado", "Nativo",
           "Qualidade", "Equilibrado", "Desempenho", "Ultra desempenho",
           "Para diagnóstico. Algumas opções podem travar o jogo ou causar bugs gráficos.", "Ao vivo", "Ao iniciar",
           "Valem na hora, com o jogo aberto, e duram só esta execução.", "Valem a partir da próxima vez que o jogo abrir.",
           "Restaurar padrão",
           "Pastas",
           "Arquivos do jogo",
           "Saves",
           "Cache de shaders",
           "Procurar",
           "Padrão",
           "A atualização é procurada ao lado da pasta do jogo, com '-patch' no nome. Mudar a pasta de saves ou de cache não move os arquivos: o jogo começa sem saves, ou recompila os shaders.",
           "Esta pasta não tem eboot.bin",
           "Compilar shaders em segundo plano", "Controles", "Sombras do PS4 Pro (maior resolução)"],
    "en": ["PC port", "Video", "Game", "Language", "Launcher language", "Display mode", "Resolution",
           "FPS limit", "Upscaler", "Upscaler quality", "VSync", "Show FPS in game", "Motion blur",
           "Reduce idle CPU use", "Record a performance profile (about 10% fewer FPS)", "PLAY", "RUNNING", "STARTING...",
           "Ready", "Debug", "Open log", "Game not found in {}", "Executable not found: run gow3_pc/build.sh", "Borderless",
           "Fullscreen", "Windowed", "Off", "Native", "Quality", "Balanced", "Performance", "Ultra performance",
           "For diagnostics. Some options may hang the game or cause graphics bugs.", "Live", "At start",
           "Apply at once, with the game open, for this run only.", "Apply the next time the game opens.",
           "Restore defaults",
           "Folders",
           "Game files",
           "Saves",
           "Shader cache",
           "Browse",
           "Default",
           "The update is looked for beside the game folder, with '-patch' in its name. Changing the saves or cache folder does not move the files: the game starts with no saves, or compiles its shaders again.",
           "This folder has no eboot.bin",
           "Compile shaders in the background", "Controls", "PS4 Pro shadows (higher resolution)"],
    "es": ["Port para PC", "Vídeo", "Juego", "Idioma", "Idioma del launcher", "Modo de pantalla",
           "Resolución", "Límite de FPS", "Upscaler", "Calidad del upscaler", "VSync", "Mostrar FPS en el juego",
           "Desenfoque de movimiento", "Reducir el uso de CPU inactiva",
           "Grabar perfil de rendimiento (cerca de 10% menos FPS)", "JUGAR", "EN EJECUCIÓN", "INICIANDO...", "Listo", "Debug",
           "Abrir registro", "Juego no encontrado en {}", "Ejecutable no encontrado: ejecuta gow3_pc/build.sh", "Sin bordes",
           "Pantalla completa", "Ventana", "Desactivado", "Nativo", "Calidad", "Equilibrado", "Rendimiento",
           "Ultra rendimiento", "Para diagnóstico. Algunas opciones pueden bloquear el juego o causar errores gráficos.",
           "En vivo", "Al iniciar", "Se aplican al momento, con el juego abierto, solo en esta ejecución.",
           "Se aplican la próxima vez que se abra el juego.", "Restaurar valores",
           "Carpetas",
           "Archivos del juego",
           "Partidas guardadas",
           "Caché de shaders",
           "Examinar",
           "Predeterminada",
           "La actualización se busca junto a la carpeta del juego, con '-patch' en el nombre. Cambiar la carpeta de partidas o de caché no mueve los archivos: el juego empieza sin partidas o vuelve a compilar los shaders.",
           "Esta carpeta no tiene eboot.bin",
           "Compilar shaders en segundo plano", "Controles", "Sombras de PS4 Pro (mayor resolución)"],
    "fr": ["Portage PC", "Vidéo", "Jeu", "Langue", "Langue du launcher", "Mode d'affichage", "Résolution",
           "Limite d'IPS", "Upscaler", "Qualité de l'upscaler", "VSync", "Afficher les IPS en jeu", "Flou de mouvement",
           "Réduire l'usage du CPU au repos", "Enregistrer un profil de performance (environ 10 % d'IPS en moins)", "JOUER",
           "EN COURS", "DÉMARRAGE...", "Prêt", "Debug", "Ouvrir le journal", "Jeu introuvable dans {}",
           "Exécutable introuvable : lancez gow3_pc/build.sh", "Sans bordure", "Plein écran", "Fenêtré", "Désactivé", "Natif",
           "Qualité", "Équilibré", "Performance", "Ultra performance",
           "Pour le diagnostic. Certaines options peuvent bloquer le jeu ou causer des bugs graphiques.", "En direct",
           "Au démarrage", "S'appliquent tout de suite, jeu ouvert, pour cette session seulement.",
           "S'appliquent à la prochaine ouverture du jeu.", "Rétablir les valeurs",
           "Dossiers",
           "Fichiers du jeu",
           "Sauvegardes",
           "Cache des shaders",
           "Parcourir",
           "Par défaut",
           "La mise à jour est cherchée à côté du dossier du jeu, avec '-patch' dans son nom. Changer le dossier des sauvegardes ou du cache ne déplace pas les fichiers : le jeu démarre sans sauvegardes ou recompile ses shaders.",
           "Ce dossier ne contient pas eboot.bin",
           "Compiler les shaders en arrière-plan", "Commandes", "Ombres de la PS4 Pro (meilleure résolution)"],
    "de": ["PC-Port", "Video", "Spiel", "Sprache", "Launcher-Sprache", "Anzeigemodus", "Auflösung", "FPS-Limit",
           "Upscaler", "Upscaler-Qualität", "VSync", "FPS im Spiel anzeigen", "Bewegungsunschärfe",
           "CPU-Last im Leerlauf senken", "Leistungsprofil aufzeichnen (etwa 10 % weniger FPS)", "SPIELEN", "LÄUFT",
           "STARTET...", "Bereit", "Debug", "Log öffnen", "Spiel nicht gefunden in {}",
           "Programmdatei nicht gefunden: gow3_pc/build.sh ausführen", "Randlos", "Vollbild", "Fenster", "Aus", "Nativ",
           "Qualität", "Ausgewogen", "Leistung", "Ultra-Leistung",
           "Zur Diagnose. Manche Optionen können das Spiel einfrieren oder Grafikfehler verursachen.", "Live", "Beim Start",
           "Gelten sofort bei laufendem Spiel, nur für diesen Start.", "Gelten beim nächsten Spielstart.",
           "Standard wiederherstellen",
           "Ordner",
           "Spieldateien",
           "Spielstände",
           "Shader-Cache",
           "Durchsuchen",
           "Standard",
           "Das Update wird neben dem Spielordner gesucht, mit '-patch' im Namen. Ein anderer Ordner für Spielstände oder Cache verschiebt die Dateien nicht: Das Spiel startet ohne Spielstände oder kompiliert die Shader neu.",
           "Dieser Ordner enthält keine eboot.bin",
           "Shader im Hintergrund kompilieren", "Steuerung", "PS4-Pro-Schatten (höhere Auflösung)"],
    "it": ["Port per PC", "Video", "Gioco", "Lingua", "Lingua del launcher", "Modalità schermo",
           "Risoluzione", "Limite FPS", "Upscaler", "Qualità dell'upscaler", "VSync", "Mostra FPS in gioco",
           "Sfocatura di movimento", "Riduci l'uso della CPU inattiva",
           "Registra profilo prestazioni (circa 10% di FPS in meno)", "GIOCA", "IN ESECUZIONE", "AVVIO...", "Pronto", "Debug",
           "Apri log", "Gioco non trovato in {}", "Eseguibile non trovato: esegui gow3_pc/build.sh", "Senza bordi",
           "Schermo intero", "Finestra", "Disattivato", "Nativo", "Qualità", "Bilanciato", "Prestazioni", "Ultra prestazioni",
           "Per la diagnostica. Alcune opzioni possono bloccare il gioco o causare errori grafici.", "Dal vivo", "All'avvio",
           "Valgono subito, a gioco aperto, solo per questa esecuzione.", "Valgono al prossimo avvio del gioco.",
           "Ripristina predefiniti",
           "Cartelle",
           "File del gioco",
           "Salvataggi",
           "Cache degli shader",
           "Sfoglia",
           "Predefinita",
           "L'aggiornamento viene cercato accanto alla cartella del gioco, con '-patch' nel nome. Cambiare la cartella dei salvataggi o della cache non sposta i file: il gioco parte senza salvataggi o ricompila gli shader.",
           "Questa cartella non contiene eboot.bin",
           "Compila gli shader in background", "Comandi", "Ombre di PS4 Pro (risoluzione più alta)"],
    "pl": ["Port na PC", "Obraz", "Gra", "Język", "Język launchera", "Tryb ekranu", "Rozdzielczość", "Limit FPS",
           "Upscaler", "Jakość upscalera", "VSync", "Pokazuj FPS w grze", "Rozmycie ruchu", "Mniejsze użycie CPU w bezczynności",
           "Nagrywaj profil wydajności (około 10% mniej FPS)", "GRAJ", "URUCHOMIONO", "URUCHAMIANIE...", "Gotowe", "Debug",
           "Otwórz log", "Nie znaleziono gry w {}", "Nie znaleziono pliku wykonywalnego: uruchom gow3_pc/build.sh",
           "Bez ramek", "Pełny ekran", "Okno", "Wyłączone", "Natywna", "Jakość", "Zrównoważona", "Wydajność", "Ultra wydajność",
           "Do diagnostyki. Niektóre opcje mogą zawiesić grę lub powodować błędy grafiki.", "Na żywo", "Przy starcie",
           "Działają od razu, przy otwartej grze, tylko w tym uruchomieniu.", "Działają od następnego uruchomienia gry.",
           "Przywróć domyślne",
           "Foldery",
           "Pliki gry",
           "Zapisy",
           "Pamięć podręczna shaderów",
           "Przeglądaj",
           "Domyślny",
           "Aktualizacja jest szukana obok folderu gry, z '-patch' w nazwie. Zmiana folderu zapisów lub pamięci podręcznej nie przenosi plików: gra zacznie bez zapisów albo ponownie skompiluje shadery.",
           "Ten folder nie zawiera eboot.bin",
           "Kompiluj shadery w tle", "Sterowanie", "Cienie z PS4 Pro (wyższa rozdzielczość)"],
    "ru": ["Порт для ПК", "Видео", "Игра", "Язык", "Язык лаунчера", "Режим экрана", "Разрешение", "Лимит FPS",
           "Апскейлер", "Качество апскейлера", "VSync", "Показывать FPS в игре", "Размытие в движении",
           "Снизить нагрузку на ЦП в простое", "Записывать профиль производительности (примерно на 10% меньше FPS)", "ИГРАТЬ",
           "ЗАПУЩЕНО", "ЗАПУСК...", "Готово", "Отладка", "Открыть журнал", "Игра не найдена в {}",
           "Исполняемый файл не найден: запустите gow3_pc/build.sh", "Без рамки", "Полный экран", "В окне", "Выключено",
           "Родное", "Качество", "Баланс", "Производительность", "Ультрапроизводительность",
           "Для диагностики. Некоторые параметры могут привести к зависанию или графическим ошибкам.", "На лету", "При запуске",
           "Применяются сразу, при открытой игре, только на этот запуск.", "Применяются при следующем запуске игры.",
           "Восстановить значения",
           "Папки",
           "Файлы игры",
           "Сохранения",
           "Кэш шейдеров",
           "Обзор",
           "По умолчанию",
           "Обновление ищется рядом с папкой игры, с '-patch' в имени. Смена папки сохранений или кэша не переносит файлы: игра начнётся без сохранений или заново скомпилирует шейдеры.",
           "В этой папке нет eboot.bin",
           "Компилировать шейдеры в фоне", "Управление", "Тени PS4 Pro (более высокое разрешение)"],
    "ja": ["PC移植版", "映像", "ゲーム", "言語", "ランチャーの言語", "表示モード", "解像度", "FPS上限", "アップスケーラー",
           "アップスケーラーの品質", "VSync", "ゲーム内にFPSを表示", "モーションブラー", "待機時のCPU使用を減らす",
           "パフォーマンスプロファイルを記録(FPSが約10%低下)", "プレイ", "実行中", "起動中...", "準備完了", "デバッグ", "ログを開く",
           "ゲームが見つかりません: {}", "実行ファイルが見つかりません: gow3_pc/build.sh を実行してください", "ボーダーレス", "フルスクリーン",
           "ウィンドウ", "オフ", "ネイティブ", "品質", "バランス", "パフォーマンス", "ウルトラパフォーマンス",
           "診断用です。一部のオプションはゲームの停止や表示の不具合を起こす場合があります。", "ライブ", "起動時",
           "ゲーム実行中にすぐ反映され、今回の実行のみ有効です。", "次回ゲームを開いたときに反映されます。", "初期設定に戻す",
           "フォルダー",
           "ゲームファイル",
           "セーブデータ",
           "シェーダーキャッシュ",
           "参照",
           "既定",
           "アップデートはゲームフォルダーの隣にある名前に「-patch」が付いたフォルダーから探します。セーブやキャッシュのフォルダーを変更してもファイルは移動しません。セーブなしで始まるか、シェーダーが再コンパイルされます。",
           "このフォルダーには eboot.bin がありません",
           "シェーダーをバックグラウンドでコンパイル", "操作設定", "PS4 Proの影(高解像度)"],
}
TEXTS = {code: dict(zip(KEYS, values)) for code, values in TEXT_TABLE.items()}
# The shader cache's "Clear" button (Folders window).
CLEAR_KEYS = "clear_cache clear_confirm clear_done clear_empty clear_running clear_failed".split()
CLEAR_TABLE = {
    "pt": ["Limpar", "Mover o cache de shaders para a Lixeira? O jogo compila os shaders de novo e terá pausas curtas até terminar.",
           "Cache movido para a Lixeira.", "O cache já está vazio.", "Feche o jogo antes de limpar o cache.",
           "Não foi possível mover o cache para a Lixeira."],
    "en": ["Clear", "Move the shader cache to the Recycle Bin? The game compiles its shaders again, with short pauses until it is done.",
           "Cache moved to the Recycle Bin.", "The cache is already empty.", "Close the game before clearing the cache.",
           "The cache could not be moved to the Recycle Bin."],
    "es": ["Borrar", "¿Mover la caché de shaders a la Papelera? El juego vuelve a compilar los shaders, con pausas cortas hasta terminar.",
           "Caché movida a la Papelera.", "La caché ya está vacía.", "Cierra el juego antes de borrar la caché.",
           "No se pudo mover la caché a la Papelera."],
    "fr": ["Vider", "Mettre le cache des shaders à la Corbeille ? Le jeu recompile ses shaders, avec de courtes pauses jusqu'à la fin.",
           "Cache mis à la Corbeille.", "Le cache est déjà vide.", "Fermez le jeu avant de vider le cache.",
           "Le cache n'a pas pu être mis à la Corbeille."],
    "de": ["Leeren", "Den Shader-Cache in den Papierkorb verschieben? Das Spiel kompiliert die Shader neu, mit kurzen Pausen bis es fertig ist.",
           "Cache in den Papierkorb verschoben.", "Der Cache ist bereits leer.", "Schließe das Spiel, bevor du den Cache leerst.",
           "Der Cache konnte nicht in den Papierkorb verschoben werden."],
    "it": ["Svuota", "Spostare la cache degli shader nel Cestino? Il gioco ricompila gli shader, con brevi pause finché non ha finito.",
           "Cache spostata nel Cestino.", "La cache è già vuota.", "Chiudi il gioco prima di svuotare la cache.",
           "Impossibile spostare la cache nel Cestino."],
    "pl": ["Wyczyść", "Przenieść pamięć podręczną shaderów do Kosza? Gra skompiluje shadery od nowa, z krótkimi przerwami do końca.",
           "Pamięć podręczna przeniesiona do Kosza.", "Pamięć podręczna jest już pusta.", "Zamknij grę przed czyszczeniem pamięci podręcznej.",
           "Nie udało się przenieść pamięci podręcznej do Kosza."],
    "ru": ["Очистить", "Переместить кэш шейдеров в Корзину? Игра заново скомпилирует шейдеры, с короткими паузами, пока не закончит.",
           "Кэш перемещён в Корзину.", "Кэш уже пуст.", "Закройте игру перед очисткой кэша.",
           "Не удалось переместить кэш в Корзину."],
    "ja": ["消去", "シェーダーキャッシュをごみ箱に移動しますか?ゲームはシェーダーを再コンパイルし、終わるまで短い停止が起きます。",
           "キャッシュをごみ箱に移動しました。", "キャッシュはすでに空です。", "キャッシュを消去する前にゲームを閉じてください。",
           "キャッシュをごみ箱に移動できませんでした。"],
}
for _code, _values in CLEAR_TABLE.items():
    TEXTS[_code].update(zip(CLEAR_KEYS, _values))
# The ambient shadows' size (Video section; tools/probe_tlou.py AMBIENT_SHADOWS_HALF, #66).
AMBIENT_TEXT = {
    "pt": "Sombras de contato em meia resolução (menos contorno)",
    "en": "Contact shadows at half resolution (less outline)",
    "es": "Sombras de contacto a media resolución (menos contorno)",
    "fr": "Ombres de contact en demi-résolution (moins de contour)",
    "de": "Kontaktschatten in halber Auflösung (weniger Umriss)",
    "it": "Ombre di contatto a metà risoluzione (meno contorno)",
    "pl": "Cienie kontaktowe w połowie rozdzielczości (mniej obwódek)",
    "ru": "Контактные тени в половинном разрешении (меньше ореола)",
    "ja": "接地影を半分の解像度で描画（輪郭を軽減）",
}
for _code, _value in AMBIENT_TEXT.items():
    TEXTS[_code]["ambient_half"] = _value
# The quarter-size depth those shadows are blended with: the pixel they were computed at instead of
# the farthest of each 4x4 block (GOW3_QUARTER_DEPTH, quarter_depth_pass.cpp, #5/#66).
DEPTH_TEXT = {
    "pt": "Alinhar sombras de contato com a borda dos objetos",
    "en": "Align contact shadows with object edges",
    "es": "Alinear sombras de contacto con el borde de los objetos",
    "fr": "Aligner les ombres de contact sur le bord des objets",
    "de": "Kontaktschatten an Objektkanten ausrichten",
    "it": "Allinea le ombre di contatto ai bordi degli oggetti",
    "pl": "Wyrównaj cienie kontaktowe z krawędziami obiektów",
    "ru": "Выравнивать контактные тени по краям объектов",
    "ja": "接地影を物体の輪郭に合わせる",
}
for _code, _value in DEPTH_TEXT.items():
    TEXTS[_code]["quarter_depth"] = _value
# Video memory (LOW_VRAM): the label and its three choices.
VRAM_TEXT = {
    "pt": ("Memória de vídeo", "Automático", "2-3 GB (modo econômico)", "4 GB ou mais"),
    "en": ("Video memory", "Automatic", "2-3 GB (saving mode)", "4 GB or more"),
    "es": ("Memoria de vídeo", "Automático", "2-3 GB (modo de ahorro)", "4 GB o más"),
    "fr": ("Mémoire vidéo", "Automatique", "2-3 Go (mode économe)", "4 Go ou plus"),
    "de": ("Videospeicher", "Automatisch", "2-3 GB (Sparmodus)", "4 GB oder mehr"),
    "it": ("Memoria video", "Automatico", "2-3 GB (modalità risparmio)", "4 GB o più"),
    "pl": ("Pamięć wideo", "Automatycznie", "2-3 GB (tryb oszczędny)", "4 GB lub więcej"),
    "ru": ("Видеопамять", "Автоматически", "2-3 ГБ (экономный режим)", "4 ГБ или больше"),
    "ja": ("ビデオメモリ", "自動", "2〜3 GB（節約モード）", "4 GB以上"),
}
for _code, (_label, _auto, _low, _normal) in VRAM_TEXT.items():
    TEXTS[_code].update({"vram": _label, "vram_auto": _auto, "vram_low": _low, "vram_normal": _normal})
# What each option of the main screen does, shown by its (i) icon (English for a missing text).
INFO_TEXT = {
    "pt": {
        "upscaler": "Desenha a cena menor e amplia para a tela (FSR 3, DLSS em placas NVIDIA RTX, ou TAA). Desligado: tamanho nativo.",
        "quality": "Quanto o upscaler reduz a cena antes de ampliar: Qualidade reduz pouco, Ultra desempenho reduz mais (mais FPS, menos nitidez).",
        "vsync": "Sincroniza com a taxa do monitor: sem cortes na imagem, mas com um pouco mais de atraso nos comandos.",
        "ambient_half": "Desenha as sombras suaves de contato (personagens, veículos, objetos) em meia resolução em vez de um quarto: o contorno claro onde pés e rodas tocam o chão fica com metade da largura. Custa um pouco de desempenho.",
        "quarter_depth": "Mistura essas sombras usando a profundidade do mesmo pixel em que foram calculadas, em vez da do pixel mais distante: menos contorno claro ou escuro na borda dos personagens. Sem custo perceptível.",
        "motion_blur": "Desfoque de movimento do jogo ao girar a câmera e em objetos rápidos.",
        "job_sleep": "As threads de trabalho do jogo dormem quando não há tarefa, em vez de ficarem girando: menos uso de CPU e calor, mesmo desempenho.",
        "profile": "Grava um perfil de desempenho para diagnóstico de problemas (cerca de 10% menos FPS). Deixe desligado para jogar.",
        "background_shaders": "Compila shaders novos em segundo plano: menos travadas, mas algum objeto pode aparecer um instante depois. Desligado: o jogo espera cada shader.",
        "pro_shadows": "Sombras do sol e das luzes em resolução maior, como no PS4 Pro: mais nítidas e um pouco mais pesadas.",
        "vram": "Para placas de vídeo de 2 ou 3 GB. Usa menos memória de vídeo: reservas menores, texturas antigas descartadas mais cedo e, quando a placa enche, dados vão para a memória do sistema em vez de fechar o jogo. Automático liga sozinho em placas de até 3 GB. Escolher 2-3 GB também desmarca as sombras do PS4 Pro. Combine com 1280x720 ou 1600x900 (ou um upscaler).",
    },
    "en": {
        "upscaler": "Draws the scene smaller and scales it to the screen (FSR 3, DLSS on NVIDIA RTX cards, or TAA). Off: native size.",
        "quality": "How much the upscaler shrinks the scene first: Quality a little, Ultra performance the most (more FPS, less sharpness).",
        "vsync": "Syncs to the monitor's refresh rate: no tearing, slightly more input delay.",
        "ambient_half": "Draws the soft contact shadows (characters, vehicles, props) at half resolution instead of a quarter: the light outline where feet and wheels touch the ground is half as wide. Costs some performance.",
        "quarter_depth": "Blends those shadows with the depth of the pixel they were computed at instead of the farthest one: less light or dark outline along characters' edges. No noticeable cost.",
        "motion_blur": "The game's motion blur when turning the camera and on fast objects.",
        "job_sleep": "The game's worker threads sleep when they have nothing to do instead of spinning: less CPU use and heat, same performance.",
        "profile": "Records a performance profile to diagnose problems (about 10% fewer FPS). Keep it off to play.",
        "background_shaders": "Compiles new shaders in the background: fewer stutters, but an object may appear a moment late. Off: the game waits for each shader.",
        "pro_shadows": "Sun and light shadows at a higher resolution, as on a PS4 Pro: sharper and a little heavier.",
        "vram": "For 2 or 3 GB graphics cards. Uses less video memory: smaller reserves, old textures dropped sooner and, when the card is full, data goes to system memory instead of closing the game. Automatic turns it on by itself on cards of up to 3 GB. Choosing 2-3 GB also unticks PS4 Pro shadows. Pair it with 1280x720 or 1600x900 (or an upscaler).",
    },
    "es": {
        "upscaler": "Dibuja la escena más pequeña y la amplía a la pantalla (FSR 3, DLSS en tarjetas NVIDIA RTX, o TAA). Desactivado: tamaño nativo.",
        "quality": "Cuánto reduce el escalador la escena antes de ampliarla: Calidad un poco, Ultra rendimiento lo máximo (más FPS, menos nitidez).",
        "vsync": "Se sincroniza con la frecuencia del monitor: sin cortes en la imagen, con algo más de retraso en los controles.",
        "ambient_half": "Dibuja las sombras suaves de contacto (personajes, vehículos, objetos) a media resolución en lugar de un cuarto: el contorno claro donde los pies y las ruedas tocan el suelo es la mitad de ancho. Cuesta algo de rendimiento.",
        "quarter_depth": "Mezcla esas sombras con la profundidad del píxel en que se calcularon en lugar de la del más lejano: menos contorno claro u oscuro en el borde de los personajes. Sin coste apreciable.",
        "motion_blur": "El desenfoque de movimiento del juego al girar la cámara y en objetos rápidos.",
        "job_sleep": "Los hilos de trabajo del juego duermen cuando no tienen tareas en lugar de girar en vacío: menos uso de CPU y calor, mismo rendimiento.",
        "profile": "Graba un perfil de rendimiento para diagnosticar problemas (alrededor de un 10% menos de FPS). Déjalo desactivado para jugar.",
        "background_shaders": "Compila los shaders nuevos en segundo plano: menos tirones, pero un objeto puede aparecer un instante después. Desactivado: el juego espera cada shader.",
        "pro_shadows": "Sombras del sol y de las luces a mayor resolución, como en PS4 Pro: más nítidas y algo más exigentes.",
        "vram": "Para tarjetas gráficas de 2 o 3 GB. Usa menos memoria de vídeo: reservas más pequeñas, texturas viejas descartadas antes y, cuando la tarjeta se llena, los datos van a la memoria del sistema en lugar de cerrar el juego. Automático lo activa solo en tarjetas de hasta 3 GB. Elegir 2-3 GB también desmarca las sombras de PS4 Pro. Combínalo con 1280x720 o 1600x900 (o un upscaler).",
    },
    "fr": {
        "upscaler": "Dessine la scène plus petite puis l'agrandit à l'écran (FSR 3, DLSS sur les cartes NVIDIA RTX, ou TAA). Désactivé : taille native.",
        "quality": "À quel point l'upscaler réduit la scène avant de l'agrandir : Qualité un peu, Ultra performance le plus (plus de FPS, moins de netteté).",
        "vsync": "Se synchronise sur la fréquence de l'écran : pas de déchirure de l'image, un peu plus de latence des commandes.",
        "ambient_half": "Dessine les ombres de contact douces (personnages, véhicules, objets) en demi-résolution au lieu d'un quart : le contour clair là où les pieds et les roues touchent le sol est deux fois moins large. Coûte un peu de performances.",
        "quarter_depth": "Mélange ces ombres avec la profondeur du pixel où elles ont été calculées au lieu du plus éloigné : moins de contour clair ou sombre au bord des personnages. Sans coût notable.",
        "motion_blur": "Le flou de mouvement du jeu en tournant la caméra et sur les objets rapides.",
        "job_sleep": "Les threads de travail du jeu dorment quand ils n'ont rien à faire au lieu de tourner à vide : moins de CPU et de chaleur, mêmes performances.",
        "profile": "Enregistre un profil de performances pour diagnostiquer un problème (environ 10 % de FPS en moins). Laissez-le désactivé pour jouer.",
        "background_shaders": "Compile les nouveaux shaders en arrière-plan : moins de saccades, mais un objet peut apparaître un instant plus tard. Désactivé : le jeu attend chaque shader.",
        "pro_shadows": "Ombres du soleil et des lumières en plus haute résolution, comme sur PS4 Pro : plus nettes et un peu plus gourmandes.",
        "vram": "Pour les cartes graphiques de 2 ou 3 Go. Utilise moins de mémoire vidéo : réserves plus petites, anciennes textures libérées plus tôt et, quand la carte est pleine, les données vont en mémoire système au lieu de fermer le jeu. Automatique l'active tout seul sur les cartes jusqu'à 3 Go. Choisir 2-3 Go décoche aussi les ombres PS4 Pro. À combiner avec 1280x720 ou 1600x900 (ou un upscaler).",
    },
    "de": {
        "upscaler": "Zeichnet die Szene kleiner und skaliert sie auf den Bildschirm (FSR 3, DLSS auf NVIDIA-RTX-Karten oder TAA). Aus: native Größe.",
        "quality": "Wie stark der Upscaler die Szene vorher verkleinert: Qualität wenig, Ultra-Leistung am meisten (mehr FPS, weniger Schärfe).",
        "vsync": "Synchronisiert mit der Bildwiederholrate des Monitors: kein Tearing, etwas mehr Eingabeverzögerung.",
        "ambient_half": "Zeichnet die weichen Kontaktschatten (Figuren, Fahrzeuge, Objekte) in halber statt viertel Auflösung: Der helle Rand, wo Füße und Räder den Boden berühren, ist halb so breit. Kostet etwas Leistung.",
        "quarter_depth": "Mischt diese Schatten mit der Tiefe des Pixels, an dem sie berechnet wurden, statt mit der des entferntesten: weniger heller oder dunkler Rand an den Kanten der Figuren. Ohne spürbare Kosten.",
        "motion_blur": "Die Bewegungsunschärfe des Spiels beim Drehen der Kamera und bei schnellen Objekten.",
        "job_sleep": "Die Arbeits-Threads des Spiels schlafen, wenn sie nichts zu tun haben, statt zu kreisen: weniger CPU-Last und Wärme, gleiche Leistung.",
        "profile": "Zeichnet ein Leistungsprofil zur Fehlersuche auf (etwa 10 % weniger FPS). Zum Spielen ausgeschaltet lassen.",
        "background_shaders": "Kompiliert neue Shader im Hintergrund: weniger Ruckler, aber ein Objekt kann einen Moment später erscheinen. Aus: Das Spiel wartet auf jeden Shader.",
        "pro_shadows": "Sonnen- und Lichtschatten in höherer Auflösung wie auf der PS4 Pro: schärfer und etwas anspruchsvoller.",
        "vram": "Für Grafikkarten mit 2 oder 3 GB. Verbraucht weniger Videospeicher: kleinere Reserven, alte Texturen werden früher verworfen, und wenn die Karte voll ist, landen Daten im Systemspeicher, statt das Spiel zu beenden. Automatisch schaltet ihn bei Karten bis 3 GB selbst ein. 2-3 GB entfernt auch das Häkchen bei den PS4-Pro-Schatten. Am besten mit 1280x720 oder 1600x900 (oder einem Upscaler).",
    },
    "it": {
        "upscaler": "Disegna la scena più piccola e la ingrandisce sullo schermo (FSR 3, DLSS sulle schede NVIDIA RTX, o TAA). Disattivato: dimensione nativa.",
        "quality": "Quanto l'upscaler riduce la scena prima di ingrandirla: Qualità poco, Prestazioni ultra al massimo (più FPS, meno nitidezza).",
        "vsync": "Si sincronizza con la frequenza del monitor: niente tearing, un po' più di ritardo nei comandi.",
        "ambient_half": "Disegna le ombre di contatto morbide (personaggi, veicoli, oggetti) a metà risoluzione invece che a un quarto: il contorno chiaro dove piedi e ruote toccano il suolo è largo la metà. Costa un po' di prestazioni.",
        "quarter_depth": "Fonde queste ombre con la profondità del pixel in cui sono state calcolate invece che con quella del più lontano: meno contorno chiaro o scuro ai bordi dei personaggi. Nessun costo percepibile.",
        "motion_blur": "Il motion blur del gioco quando si ruota la telecamera e sugli oggetti veloci.",
        "job_sleep": "I thread di lavoro del gioco dormono quando non hanno nulla da fare invece di girare a vuoto: meno uso della CPU e calore, stesse prestazioni.",
        "profile": "Registra un profilo delle prestazioni per diagnosticare problemi (circa il 10% di FPS in meno). Lascialo disattivato per giocare.",
        "background_shaders": "Compila i nuovi shader in background: meno scatti, ma un oggetto può comparire un istante dopo. Disattivato: il gioco attende ogni shader.",
        "pro_shadows": "Ombre del sole e delle luci a risoluzione più alta, come su PS4 Pro: più nitide e un po' più pesanti.",
        "vram": "Per schede video da 2 o 3 GB. Usa meno memoria video: riserve più piccole, texture vecchie scartate prima e, quando la scheda è piena, i dati vanno nella memoria di sistema invece di chiudere il gioco. Automatico la attiva da solo sulle schede fino a 3 GB. Scegliere 2-3 GB toglie anche la spunta alle ombre PS4 Pro. Abbinala a 1280x720 o 1600x900 (o a un upscaler).",
    },
    "pl": {
        "upscaler": "Rysuje scenę w mniejszym rozmiarze i skaluje ją do ekranu (FSR 3, DLSS na kartach NVIDIA RTX lub TAA). Wyłączone: rozmiar natywny.",
        "quality": "Jak bardzo upscaler najpierw zmniejsza scenę: Jakość trochę, Ultra wydajność najbardziej (więcej FPS, mniej ostrości).",
        "vsync": "Synchronizuje z odświeżaniem monitora: bez rozrywania obrazu, nieco większe opóźnienie sterowania.",
        "ambient_half": "Rysuje miękkie cienie kontaktowe (postacie, pojazdy, obiekty) w połowie rozdzielczości zamiast w ćwierci: jasna obwódka tam, gdzie stopy i koła dotykają podłoża, jest o połowę węższa. Kosztuje trochę wydajności.",
        "quarter_depth": "Miesza te cienie z głębią piksela, w którym zostały obliczone, zamiast najdalszego: mniej jasnych lub ciemnych obwódek na krawędziach postaci. Bez zauważalnego kosztu.",
        "motion_blur": "Rozmycie ruchu w grze przy obracaniu kamery i przy szybkich obiektach.",
        "job_sleep": "Wątki robocze gry usypiają, gdy nie mają zadań, zamiast kręcić się w pętli: mniejsze użycie CPU i mniej ciepła, ta sama wydajność.",
        "profile": "Zapisuje profil wydajności do diagnozowania problemów (około 10% mniej FPS). Do gry zostaw wyłączone.",
        "background_shaders": "Kompiluje nowe shadery w tle: mniej przycięć, ale obiekt może pojawić się chwilę później. Wyłączone: gra czeka na każdy shader.",
        "pro_shadows": "Cienie od słońca i świateł w wyższej rozdzielczości, jak na PS4 Pro: ostrzejsze i nieco bardziej wymagające.",
        "vram": "Dla kart graficznych z 2 lub 3 GB. Zużywa mniej pamięci wideo: mniejsze rezerwy, stare tekstury zwalniane wcześniej, a gdy karta jest pełna, dane trafiają do pamięci systemowej zamiast zamykać grę. Automatycznie włącza się sam na kartach do 3 GB. Wybór 2-3 GB odznacza też cienie z PS4 Pro. Łącz z 1280x720 lub 1600x900 (albo upscalerem).",
    },
    "ru": {
        "upscaler": "Рисует сцену в меньшем размере и масштабирует её до экрана (FSR 3, DLSS на картах NVIDIA RTX или TAA). Выкл.: родной размер.",
        "quality": "Насколько апскейлер сначала уменьшает сцену: Качество — немного, Ультра производительность — сильнее всего (больше FPS, меньше чёткости).",
        "vsync": "Синхронизация с частотой монитора: без разрывов изображения, но с чуть большей задержкой управления.",
        "ambient_half": "Рисует мягкие контактные тени (персонажи, транспорт, предметы) в половинном разрешении вместо четвертного: светлый ореол там, где ноги и колёса касаются земли, становится вдвое уже. Немного снижает производительность.",
        "quarter_depth": "Смешивает эти тени по глубине того пикселя, в котором они вычислены, а не самого дальнего: меньше светлых и тёмных ореолов по краям персонажей. Без заметной нагрузки.",
        "motion_blur": "Размытие движения в игре при повороте камеры и на быстрых объектах.",
        "job_sleep": "Рабочие потоки игры засыпают, когда им нечего делать, вместо холостого вращения: меньше нагрузка на CPU и нагрев, та же производительность.",
        "profile": "Записывает профиль производительности для поиска проблем (примерно на 10% меньше FPS). Для игры оставьте выключенным.",
        "background_shaders": "Компилирует новые шейдеры в фоне: меньше подтормаживаний, но объект может появиться с небольшой задержкой. Выкл.: игра ждёт каждый шейдер.",
        "pro_shadows": "Тени от солнца и источников света в более высоком разрешении, как на PS4 Pro: чётче и немного тяжелее.",
        "vram": "Для видеокарт с 2 или 3 ГБ. Расходует меньше видеопамяти: меньшие резервы, старые текстуры выгружаются раньше, а когда память карты заполнена, данные уходят в системную память вместо закрытия игры. «Автоматически» включает режим сам на картах до 3 ГБ. Выбор 2-3 ГБ также снимает флажок теней PS4 Pro. Сочетайте с 1280x720 или 1600x900 (или с апскейлером).",
    },
    "ja": {
        "upscaler": "シーンを小さく描画して画面サイズに拡大します（FSR 3、NVIDIA RTXカードのDLSS、またはTAA）。オフ：ネイティブサイズ。",
        "quality": "アップスケーラーが拡大前にシーンをどれだけ縮小するか：クオリティは少し、ウルトラパフォーマンスは最大（FPSは上がり、鮮明さは下がります）。",
        "vsync": "モニターのリフレッシュレートに同期します：ティアリングはなくなりますが、操作の遅延が少し増えます。",
        "ambient_half": "やわらかい接地影（キャラクター、車両、オブジェクト）を4分の1ではなく半分の解像度で描画します：足や車輪が地面に触れる部分の明るい輪郭の幅が半分になります。少し負荷が増えます。",
        "quarter_depth": "これらの影を、最も遠いピクセルではなく計算に使ったピクセルの深度で合成します：キャラクターの縁の明るい・暗い輪郭が減ります。負荷はほとんどありません。",
        "motion_blur": "カメラを回したときや速い物体に付くゲームのモーションブラー。",
        "job_sleep": "ゲームのワーカースレッドが仕事のないときに空回りせず休止します：CPU使用率と発熱が減り、性能は変わりません。",
        "profile": "問題の診断用にパフォーマンスプロファイルを記録します（FPSが約10%低下）。プレイするときはオフにしてください。",
        "background_shaders": "新しいシェーダーをバックグラウンドでコンパイルします：カクつきは減りますが、オブジェクトの表示が少し遅れることがあります。オフ：ゲームが各シェーダーを待ちます。",
        "pro_shadows": "太陽や照明の影をPS4 Proと同じく高い解像度で描画します：よりシャープになり、少し重くなります。",
        "vram": "2〜3 GBのグラフィックカード向けです。ビデオメモリの使用量を減らします：予約領域を小さくし、古いテクスチャを早めに解放し、カードがいっぱいになったらゲームを終了せずにデータをシステムメモリに置きます。「自動」は3 GB以下のカードで自動的にオンになります。2〜3 GBを選ぶとPS4 Proの影のチェックも外れます。1280x720か1600x900（またはアップスケーラー）と組み合わせてください。",
    },
}
# The PS4 system language the game sees (system_language in tlou.ini; tools/probe_tlou.py passes it
# as GOW3_LANGUAGE). The game starts in it and offers Korean and Chinese only with it (issue #156).
# {id: name}, in the game's own language names; English (US) is the default.
SYSTEM_LANGUAGES = {"1": "English", "2": "Français", "3": "Español (España)", "20": "Español (Latinoamérica)",
                    "4": "Deutsch", "5": "Italiano", "6": "Nederlands", "7": "Português (Portugal)",
                    "17": "Português (Brasil)", "8": "Русский", "16": "Polski", "0": "日本語", "9": "한국어",
                    "10": "中文 (繁體)", "12": "Suomi", "13": "Svenska", "14": "Dansk", "15": "Norsk",
                    "19": "Türkçe", "23": "Čeština", "25": "Ελληνικά"}
SYSTEM_LANGUAGE_TEXT = {
    "pt": ("Idioma do jogo", "O idioma em que o jogo começa; depois ele pode ser trocado no menu do jogo. Coreano e chinês só aparecem no menu com eles escolhidos aqui. Dublagens além de inglês, francês, espanhol latino e português do Brasil precisam do arquivo <idioma>-sp.psarc da versão de outra região em build/ps4/main/sound1/streams/speech; sem ele essa dublagem não aparece em Diálogo."),
    "en": ("Game language", "The language the game starts in; it can then be changed in the game's menu. Korean and Chinese appear in the menu only when chosen here. Dubs other than English, French, Latin American Spanish and Brazilian Portuguese need the <language>-sp.psarc file of another region's release in build/ps4/main/sound1/streams/speech; without it that dub is not offered under Dialogue."),
    "es": ("Idioma del juego", "El idioma en que empieza el juego; después se puede cambiar en el menú del juego. El coreano y el chino solo aparecen en el menú si se eligen aquí. Los doblajes distintos de inglés, francés, español latino y portugués de Brasil necesitan el archivo <idioma>-sp.psarc de la versión de otra región en build/ps4/main/sound1/streams/speech; sin él ese doblaje no aparece en Diálogo."),
    "fr": ("Langue du jeu", "La langue dans laquelle le jeu démarre ; elle peut ensuite être changée dans le menu du jeu. Le coréen et le chinois n'apparaissent dans le menu que s'ils sont choisis ici. Les doublages autres que l'anglais, le français, l'espagnol d'Amérique latine et le portugais du Brésil demandent le fichier <langue>-sp.psarc de la version d'une autre région dans build/ps4/main/sound1/streams/speech ; sans lui ce doublage n'est pas proposé dans Dialogue."),
    "de": ("Spielsprache", "Die Sprache, in der das Spiel startet; danach lässt sie sich im Spielmenü ändern. Koreanisch und Chinesisch erscheinen im Menü nur, wenn sie hier gewählt sind. Synchronisationen außer Englisch, Französisch, lateinamerikanischem Spanisch und brasilianischem Portugiesisch brauchen die Datei <sprache>-sp.psarc der Version einer anderen Region in build/ps4/main/sound1/streams/speech; ohne sie wird diese Synchronisation unter Dialog nicht angeboten."),
    "it": ("Lingua del gioco", "La lingua in cui parte il gioco; poi si può cambiare nel menu del gioco. Coreano e cinese compaiono nel menu solo se scelti qui. I doppiaggi diversi da inglese, francese, spagnolo latinoamericano e portoghese brasiliano richiedono il file <lingua>-sp.psarc della versione di un'altra regione in build/ps4/main/sound1/streams/speech; senza, quel doppiaggio non compare in Dialogo."),
    "pl": ("Język gry", "Język, w którym gra się uruchamia; potem można go zmienić w menu gry. Koreański i chiński pojawiają się w menu tylko wtedy, gdy wybrano je tutaj. Dubbingi inne niż angielski, francuski, hiszpański latynoamerykański i portugalski brazylijski wymagają pliku <język>-sp.psarc z wersji z innego regionu w build/ps4/main/sound1/streams/speech; bez niego ten dubbing nie pojawia się w Dialogach."),
    "ru": ("Язык игры", "Язык, на котором запускается игра; потом его можно сменить в меню игры. Корейский и китайский появляются в меню, только если выбраны здесь. Озвучки, кроме английской, французской, латиноамериканской испанской и бразильской португальской, требуют файла <язык>-sp.psarc из версии другого региона в build/ps4/main/sound1/streams/speech; без него эта озвучка не появляется в пункте «Диалоги»."),
    "ja": ("ゲームの言語", "ゲームが起動する言語です。あとでゲームのメニューから変更できます。韓国語と中国語は、ここで選んだときだけメニューに表示されます。英語、フランス語、中南米スペイン語、ブラジルポルトガル語以外の音声には、他の地域版の <言語>-sp.psarc を build/ps4/main/sound1/streams/speech に置く必要があります。ない場合、その音声は「会話」に表示されません。"),
}
for _code, (_label, _info) in SYSTEM_LANGUAGE_TEXT.items():
    TEXTS[_code]["system_language"] = _label
    INFO_TEXT.setdefault(_code, {})["system_language"] = _info
# Updates from the GitHub releases (launcher/updater.py): {} is the version, then the size in MB.
UPDATE_KEYS = "update_button update_available update_confirm update_downloading update_installing update_failed update_close_game".split()
UPDATE_TABLE = {
    "pt": ["ATUALIZAR PARA {}", "Versão {} disponível. Saves e configurações são mantidos.",
           "Instalar a versão {} (download de {} MB)?\n\nSaves, configurações e o cache de shaders são mantidos. "
           "O launcher fecha, instala a atualização e abre de novo.",
           "Baixando a atualização... {}", "Instalando a atualização...",
           "Não foi possível atualizar. A versão instalada continua a mesma.\n\n{}", "Feche o jogo antes de atualizar."],
    "en": ["UPDATE TO {}", "Version {} is available. Saves and settings are kept.",
           "Install version {} ({} MB download)?\n\nSaves, settings and the shader cache are kept. "
           "The launcher closes, installs the update and opens again.",
           "Downloading the update... {}", "Installing the update...",
           "The update could not be installed. The installed version is unchanged.\n\n{}", "Close the game before updating."],
    "es": ["ACTUALIZAR A {}", "Versión {} disponible. Se conservan partidas y ajustes.",
           "¿Instalar la versión {} (descarga de {} MB)?\n\nSe conservan las partidas, los ajustes y la caché de shaders. "
           "El launcher se cierra, instala la actualización y se abre de nuevo.",
           "Descargando la actualización... {}", "Instalando la actualización...",
           "No se pudo actualizar. La versión instalada no cambia.\n\n{}", "Cierra el juego antes de actualizar."],
    "fr": ["METTRE À JOUR EN {}", "Version {} disponible. Sauvegardes et réglages conservés.",
           "Installer la version {} (téléchargement de {} Mo) ?\n\nLes sauvegardes, les réglages et le cache des shaders "
           "sont conservés. Le launcher se ferme, installe la mise à jour et se rouvre.",
           "Téléchargement de la mise à jour... {}", "Installation de la mise à jour...",
           "La mise à jour n'a pas pu être installée. La version installée reste inchangée.\n\n{}",
           "Fermez le jeu avant la mise à jour."],
    "de": ["AUF {} AKTUALISIEREN", "Version {} ist verfügbar. Spielstände und Einstellungen bleiben erhalten.",
           "Version {} installieren ({} MB Download)?\n\nSpielstände, Einstellungen und der Shader-Cache bleiben erhalten. "
           "Der Launcher schließt sich, installiert das Update und startet neu.",
           "Update wird heruntergeladen... {}", "Update wird installiert...",
           "Das Update konnte nicht installiert werden. Die installierte Version bleibt unverändert.\n\n{}",
           "Schließe das Spiel vor dem Update."],
    "it": ["AGGIORNA A {}", "Versione {} disponibile. Salvataggi e impostazioni vengono mantenuti.",
           "Installare la versione {} (download di {} MB)?\n\nSalvataggi, impostazioni e cache degli shader vengono "
           "mantenuti. Il launcher si chiude, installa l'aggiornamento e si riapre.",
           "Download dell'aggiornamento... {}", "Installazione dell'aggiornamento...",
           "Impossibile installare l'aggiornamento. La versione installata non cambia.\n\n{}",
           "Chiudi il gioco prima di aggiornare."],
    "pl": ["AKTUALIZUJ DO {}", "Dostępna wersja {}. Zapisy i ustawienia zostaną zachowane.",
           "Zainstalować wersję {} (pobieranie {} MB)?\n\nZapisy, ustawienia i pamięć podręczna shaderów zostaną "
           "zachowane. Launcher zamknie się, zainstaluje aktualizację i uruchomi ponownie.",
           "Pobieranie aktualizacji... {}", "Instalowanie aktualizacji...",
           "Nie udało się zainstalować aktualizacji. Zainstalowana wersja się nie zmienia.\n\n{}",
           "Zamknij grę przed aktualizacją."],
    "ru": ["ОБНОВИТЬ ДО {}", "Доступна версия {}. Сохранения и настройки останутся.",
           "Установить версию {} (загрузка {} МБ)?\n\nСохранения, настройки и кэш шейдеров останутся. "
           "Лаунчер закроется, установит обновление и откроется снова.",
           "Загрузка обновления... {}", "Установка обновления...",
           "Не удалось установить обновление. Установленная версия не изменилась.\n\n{}",
           "Закройте игру перед обновлением."],
    "ja": ["{} に更新", "バージョン {} が利用可能です。セーブと設定は保持されます。",
           "バージョン {} をインストールしますか?({} MB のダウンロード)\n\nセーブ、設定、シェーダーキャッシュは保持されます。"
           "ランチャーは閉じてアップデートをインストールし、再び開きます。",
           "アップデートをダウンロード中... {}", "アップデートをインストール中...",
           "アップデートをインストールできませんでした。インストール済みのバージョンは変わりません。\n\n{}",
           "アップデートの前にゲームを閉じてください。"],
}
for _code, _values in UPDATE_TABLE.items():
    TEXTS[_code].update(zip(UPDATE_KEYS, _values))
# Shown on PLAY (which then stops) when the update was merged into the game folder but its eboot.bin is still 1.00's
# (tools/probe_tlou.py merged_with_base_executable); {} is the game folder's name.
MERGED_TEXT = {
    "pt": "Os arquivos da atualização 1.11 foram mesclados nesta pasta, mas o eboot.bin ainda é o da versão 1.00, "
          "e o port só roda o executável da 1.11.\n\nCopie eboot.bin, sce_module e sce_sys da atualização por cima "
          "dos do jogo (substituindo os arquivos), ou deixe a atualização numa pasta própria ao lado do jogo ({}-patch).",
    "en": "Update 1.11's files were merged into this folder, but its eboot.bin is still version 1.00's, and the port "
          "only runs 1.11's executable.\n\nCopy the update's eboot.bin, sce_module and sce_sys over the game's "
          "(replacing the files), or keep the update in its own folder next to the game ({}-patch).",
    "es": "Los archivos de la actualización 1.11 se combinaron en esta carpeta, pero su eboot.bin sigue siendo el de "
          "la versión 1.00, y el port solo ejecuta el de la 1.11.\n\nCopia eboot.bin, sce_module y sce_sys de la "
          "actualización sobre los del juego (reemplazando los archivos), o deja la actualización en su propia "
          "carpeta junto al juego ({}-patch).",
    "fr": "Les fichiers de la mise à jour 1.11 ont été fusionnés dans ce dossier, mais son eboot.bin est toujours "
          "celui de la version 1.00, et le port n'exécute que celui de la 1.11.\n\nCopiez eboot.bin, sce_module et "
          "sce_sys de la mise à jour par-dessus ceux du jeu (en remplaçant les fichiers), ou gardez la mise à jour "
          "dans son propre dossier à côté du jeu ({}-patch).",
    "de": "Die Dateien von Update 1.11 wurden in diesen Ordner kopiert, aber die eboot.bin ist noch die von Version "
          "1.00, und der Port startet nur die von 1.11.\n\nKopiere eboot.bin, sce_module und sce_sys des Updates "
          "über die des Spiels (Dateien ersetzen) oder lege das Update in einen eigenen Ordner neben dem Spiel "
          "({}-patch).",
    "it": "I file dell'aggiornamento 1.11 sono stati uniti in questa cartella, ma il suo eboot.bin è ancora quello "
          "della versione 1.00, e il port avvia solo quello della 1.11.\n\nCopia eboot.bin, sce_module e sce_sys "
          "dell'aggiornamento sopra quelli del gioco (sostituendo i file), oppure tieni l'aggiornamento in una "
          "cartella a parte accanto al gioco ({}-patch).",
    "pl": "Pliki aktualizacji 1.11 zostały scalone z tym folderem, ale jego eboot.bin to nadal wersja 1.00, a port "
          "uruchamia tylko plik z 1.11.\n\nSkopiuj eboot.bin, sce_module i sce_sys z aktualizacji na pliki gry "
          "(zastępując je) albo trzymaj aktualizację w osobnym folderze obok gry ({}-patch).",
    "ru": "Файлы обновления 1.11 объединены с этой папкой, но eboot.bin в ней всё ещё от версии 1.00, а порт "
          "запускает только файл версии 1.11.\n\nСкопируйте eboot.bin, sce_module и sce_sys из обновления поверх "
          "файлов игры (с заменой) или держите обновление в отдельной папке рядом с игрой ({}-patch).",
    "ja": "このフォルダーにはアップデート 1.11 のファイルが統合されていますが、eboot.bin はまだバージョン 1.00 のもので、"
          "移植版は 1.11 の実行ファイルでしか動きません。\n\nアップデートの eboot.bin、sce_module、sce_sys をゲームの"
          "ものに上書きコピーするか、アップデートをゲームの隣の専用フォルダー({}-patch)に置いてください。",
}
for _code, _value in MERGED_TEXT.items():
    TEXTS[_code]["merged_old_exe"] = _value


def merged_with_base_executable(game):
    """tools/probe_tlou.py's check; False when it can't be loaded."""
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import probe_tlou
        return probe_tlou.merged_with_base_executable(Path(game))
    except Exception:
        return False
    finally:
        sys.path.pop(0)

BG, PANEL, TEXT, MUTED, ACCENT, ACCENT_HOT = "#090909", "#141414", "#ece9e2", "#8f8f88", "#c8862a", "#e4a23f"
DISABLED, ERROR = "#2b2b2b", "#e06c5a"


def read_ini(path=None):
    """{key: value} of the settings file; {} when it does not exist yet."""
    path = path or INI
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def write_ini(changes, path=None):
    """Sets these keys in place; every other line, comments included, stays as it is."""
    path = path or INI
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
    left = dict(changes)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in left:
            lines[i] = f"{key}={left.pop(key)}"
    lines += [f"{key}={value}" for key, value in left.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def read_controls(ini):
    """{action: [(kind, name) or None] * CONTROL_SLOTS}: the defaults with the file's maps applied as
    runtime_pad.c applies them (an action's first entry in a map replaces its sources of that kind)."""
    bindings = {action: list(sources) for action, sources in DEFAULT_CONTROLS.items()}
    keys_only = {action for _, rows in CONTROLS for action, _, only in rows if only}
    for kind, spec in (("key", ini.get("key_map", "")), ("mouse", ini.get("mouse_map", ""))):
        replaced = set()
        for entry in spec.split(","):
            if "=" not in entry:
                continue
            action, name = (part.strip() for part in entry.split("=", 1))
            action = "touchpad_left" if action.lower() == "touchpad" else action.lower()
            if kind == "mouse":
                name = name.lower()
            if action not in bindings or (kind == "mouse" and (action in keys_only or name and name not in MOUSE_NAMES)):
                continue
            if action not in replaced:
                replaced.add(action)
                bindings[action] = [s for s in bindings[action] if s[0] != kind]
            if name:
                bindings[action].append((kind, name))
    return {action: (sources + [None] * CONTROL_SLOTS)[:CONTROL_SLOTS] for action, sources in bindings.items()}


def control_maps(bindings):
    """The key_map and mouse_map values of these bindings: complete, every action listed ("id=" for
    none), like the in-game menu writes them."""
    keys_only = {action for _, rows in CONTROLS for action, _, only in rows if only}
    keys, mouse = [], []
    for action, sources in bindings.items():
        names = {kind: [s[1] for s in sources if s and s[0] == kind] for kind in ("key", "mouse")}
        keys += [f"{action}={name}" for name in names["key"]] or [f"{action}="]
        if action not in keys_only:
            mouse += [f"{action}={name}" for name in names["mouse"]] or [f"{action}="]
    return ",".join(keys), ",".join(mouse)


def key_from_event(keysym, keycode):
    """The SDL name of a key Tk reported (keycode: the Windows virtual key); None if unknown."""
    if keysym in KEYSYM_NAMES:
        return KEYSYM_NAMES[keysym]
    try:
        scan = ctypes.windll.user32.MapVirtualKeyW(keycode, 0)   # MAPVK_VK_TO_VSC
    except (AttributeError, OSError):
        return None
    return SCAN_NAMES.get(scan)


def key_label(name):
    """A key's text in Portuguese (translated by the caller): the character the player's keyboard
    layout has at that place for letters, digits and signs, else the SDL name."""
    if name in KEY_LABELS:
        return KEY_LABELS[name]
    scan = next((s for s, n in SCAN_NAMES.items() if n.lower() == name.lower()), None)
    if scan is not None and len(SCAN_NAMES[scan]) == 1:
        try:
            user32 = ctypes.windll.user32
            char = user32.MapVirtualKeyW(user32.MapVirtualKeyW(scan, 1), 2) & 0x7FFF   # VSC_TO_VK, VK_TO_CHAR
            if char > 32:
                return chr(char).upper()
        except (AttributeError, OSError):
            pass
    return name.upper() if len(name) == 1 else name


def read_switches(path=None):
    path = path or SWITCHES
    values = dict(DEFAULT_SWITCHES)
    try:
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            if k == "low_vram":
                values[k] = v if v in LOW_VRAM.values() else "auto"
            elif k in DEFAULT_SWITCHES:
                values[k] = bool(v)
    except (OSError, ValueError):
        pass
    return values


def read_overrides(path=None):
    """The debug menu's startup options that differ from their defaults: {variable: value}."""
    known = {option[1]: option[2:4] for _, options in STARTUP for option in options}
    try:
        saved = json.loads((path or SWITCHES).read_text(encoding="utf-8")).get("overrides", {})
    except (OSError, ValueError, AttributeError):
        return {}
    return {k: v for k, v in saved.items() if k in known and v in known[k]}


def read_ui_language(path=None):
    try:
        saved = json.loads((path or SWITCHES).read_text(encoding="utf-8")).get("ui_language")
    except (OSError, ValueError, AttributeError):
        return DEFAULT_LANGUAGE
    return saved if saved in TEXTS else DEFAULT_LANGUAGE


def read_folders(path=None):
    """{key: folder} chosen in the launcher; a missing or empty one is the default."""
    try:
        saved = json.loads((path or SWITCHES).read_text(encoding="utf-8")).get("folders", {})
    except (OSError, ValueError, AttributeError):
        return {}
    return {k: v for k, v in saved.items() if k in FOLDERS and isinstance(v, str) and v} if isinstance(saved, dict) else {}


def game_folder(folders):
    return Path(folders.get("game") or GAME)


def game_command(python, folders, seconds=0):
    """The probe's command line; a default folder is not passed (--saves also changes what it records)."""
    command = [python, "tools/probe_tlou.py", "--boot", "--seconds", str(seconds)]
    if folders.get("game"):
        command += ["--game", folders["game"]]
    if folders.get("saves"):
        command += ["--saves", folders["saves"]]
    return command


def load_art(width, height, game=None):
    """The dump's key art, covering width x height and kept to its right edge; None without it."""
    folder = Path(game or GAME)
    path = next((folder / p for p in ART if (folder / p).is_file()), None)
    if path is None:
        return None
    try:
        from PIL import Image, ImageTk
        image = Image.open(path).convert("RGB")
        scale = max(width / image.width, height / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        left, top = image.width - width, (image.height - height) // 2
        return ImageTk.PhotoImage(image.crop((left, top, left + width, top + height)))
    except ImportError:
        return tk.PhotoImage(file=str(path)).zoom(2).subsample(3)   # 1920x1080 -> 1280x720
    except (OSError, ValueError):
        return None


def starter_exe():
    """TLOU Launcher.exe: at the top of a package, beside this file in the repository."""
    return next((p for p in (ROOT / "TLOU Launcher.exe", HERE / "TLOU Launcher.exe") if p.is_file()), None)

def game_exe():
    """TLOU Game.exe: at the top of a package, beside this file in the repository."""
    return next((p for p in (ROOT / "TLOU Game.exe", HERE / "TLOU Game.exe") if p.is_file()), None)

def make_game_ico(game, destination):
    """The dump's icon as an .ico for the shortcut; False without the picture or Pillow."""
    picture = Path(game) / GAME_ICON
    if not picture.is_file():
        return False
    try:
        from PIL import Image
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.open(picture).convert("RGBA").save(destination, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except (ImportError, OSError, ValueError):
        return False
    return True


def make_shortcut(link, target, icon):
    """A Windows shortcut through the shell's own COM object: the paths go by environment."""
    command = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:TLOU_LINK); $s.TargetPath = $env:TLOU_TARGET; "
               "$s.WorkingDirectory = $env:TLOU_DIR; $s.IconLocation = $env:TLOU_ICON; $s.Save()")
    env = dict(os.environ, TLOU_LINK=str(link), TLOU_TARGET=str(target), TLOU_DIR=str(Path(target).parent), TLOU_ICON=str(icon))
    try:
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], env=env, timeout=30,
                              capture_output=True, creationflags=0x08000000).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def game_shortcut(game):
    """In a package, whose exe has no icon: a shortcut beside it with the icon of the player's game."""
    launcher_exe = starter_exe()
    g_exe = game_exe()

    if (launcher_exe is None or ICON.is_file()) or (g_exe is None or ICON.is_file()):
        return False
    ico = USER / "game-icon.ico"
    return make_game_ico(game, ico) and make_shortcut(launcher_exe.with_name("The Last of Us Remastered Launcher.lnk"), launcher_exe, ico) and make_shortcut(g_exe.with_name("The Last of Us Remastered.lnk"), g_exe, ico)


def toggle_mask(disabled):
    return sum(set(disabled))


def write_toggles(mask, path=None):
    """Writes the live toggles for the running game (read every 250 ms)."""
    path = path or TOGGLE_FILE
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(mask), encoding="ascii")
    os.replace(temporary, path)


def game_env(switches, base=None, overrides=None, folders=None):
    """The environment of a run with these switches."""
    env = dict(os.environ if base is None else base)
    overrides = overrides or {}
    env.update(overrides)
    # This host's game build runs at the wrong speed with the monotonic fallback;
    # preserve the calibrated native clock unless Startup explicitly disables it.
    env["GOW3_NATIVE_TSC"] = overrides.get("GOW3_NATIVE_TSC", "1")
    env["GOW3_GUEST_PRIORITY"] = overrides.get("GOW3_GUEST_PRIORITY", "1")
    if folders and folders.get("cache"):
        env["GOW3_GPU_CACHE_DIR"] = folders["cache"]
    env["GOW3_NO_MOTION_BLUR"] = "0" if switches["motion_blur"] else "1"
    env["TLOU_JOB_SLEEP"] = "1" if switches["job_sleep"] else "0"
    env["TLOU_PRO_SHADOWS"] = "1" if switches["pro_shadows"] else "0"
    env["TLOU_AMBIENT_SHADOWS"] = "half" if switches.get("ambient_half", True) else "quarter"
    env["GOW3_QUARTER_DEPTH"] = "1" if switches.get("quarter_depth", True) else "0"
    env["GOW3_LOW_VRAM"] = {"on": "1", "off": "0"}.get(switches.get("low_vram", "auto"), "auto")
    if switches["background_shaders"]:
        # Draws are skipped while their pipeline compiles: it pauses less. It flashed white and
        # crashed the game on 22 Oct 2026 and this switch turns it off if that comes back.
        env["GOW3_ASYNC_SHADERS"] = "1"
        env["GOW3_ASYNC_SHADERS_LOCKED"] = "0"
    if switches["profile"]:
        env.update(PROFILE_ENV)
    return env


# The game's shader and pipeline caches in the cache folder: <serial> (shaders and pipeline
# binaries), <serial>.gpu-<n> (another GPU's, kept aside) and <serial>.zip (archived mode). Only
# these are cleared, never the rest of a folder the player chose.
CACHE_SERIAL = "CUSA00552"


def cache_entries(folder):
    try:
        return sorted(p for p in Path(folder).iterdir()
                      if p.name == CACHE_SERIAL or p.name.startswith(CACHE_SERIAL + "."))
    except OSError:
        return []


def recycle(paths):
    """Moves the paths to the Recycle Bin (SHFileOperationW, FO_DELETE with FOF_ALLOWUNDO): never a
    permanent delete. True when it worked."""
    if not paths:
        return True
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    fo_delete, fof_silent, fof_noconfirmation, fof_allowundo, fof_noerrorui = 3, 0x4, 0x10, 0x40, 0x400
    files = "".join(str(Path(p).resolve()) + "\0" for p in paths) + "\0"
    op = SHFILEOPSTRUCTW(None, fo_delete, files, None,
                         fof_silent | fof_noconfirmation | fof_allowundo | fof_noerrorui, False, None, None)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return result == 0 and not op.fAnyOperationsAborted and not any(Path(p).exists() for p in paths)


def python_exe():
    """python.exe beside this interpreter (pythonw has no output for the scripts it runs)."""
    candidate = Path(sys.executable).with_name("python.exe")
    return str(candidate if candidate.is_file() else sys.executable)


def game_running():
    no_window = 0x08000000
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq gow3-probe.exe", "/NH"], capture_output=True,
                             text=True, errors="replace", creationflags=no_window).stdout
    except OSError:
        return False
    return "gow3-probe" in out


def key_for(options, value, fallback):
    return next((label for label, v in options.items() if v == value), fallback)


# The Controller tab reads the controller with the game's own SDL3 (gow3_pc/out in a package,
# MSYS2's CLANG64 in a source tree), and only while the tab is shown.
SDL_INIT_GAMEPAD = 0x2000
# SDL_GamepadButton 0..14 (south east west north back guide start, sticks, shoulders, d-pad) and 20.
SDL_BUTTONS = ("cross", "circle", "square", "triangle", "share", "ps", "options", "l3", "r3", "l1", "r1",
               "up", "down", "left", "right")
SDL_TOUCHPAD, SDL_MISC1 = 20, 15   # MISC1: Share/Capture on Xbox Series pads, the game's shake


def sdl_candidates():
    yield ROOT / "gow3_pc/out/SDL3.dll"
    yield Path(os.environ.get("MSYS2_ROOT", "C:/msys64")) / "clang64/bin/SDL3.dll"


class PadReader:
    """The first connected controller through SDL3. ready is False without SDL3.dll."""
    sdl = None

    @classmethod
    def library(cls):
        if cls.sdl is None:
            cls.sdl = False
            for path in sdl_candidates():
                if not path.is_file():
                    continue
                try:
                    sdl = ctypes.CDLL(str(path))
                except OSError:
                    continue
                c = ctypes
                for name, result, arguments in (
                        ("SDL_SetHint", c.c_bool, [c.c_char_p, c.c_char_p]),
                        ("SDL_InitSubSystem", c.c_bool, [c.c_uint32]), ("SDL_QuitSubSystem", None, [c.c_uint32]),
                        ("SDL_GetGamepads", c.POINTER(c.c_uint32), [c.POINTER(c.c_int)]), ("SDL_free", None, [c.c_void_p]),
                        ("SDL_OpenGamepad", c.c_void_p, [c.c_uint32]), ("SDL_CloseGamepad", None, [c.c_void_p]),
                        ("SDL_GamepadConnected", c.c_bool, [c.c_void_p]), ("SDL_GetGamepadName", c.c_char_p, [c.c_void_p]),
                        ("SDL_UpdateGamepads", None, []), ("SDL_FlushEvents", None, [c.c_uint32, c.c_uint32]),
                        ("SDL_GetGamepadButton", c.c_bool, [c.c_void_p, c.c_int]),
                        ("SDL_GetGamepadAxis", c.c_int16, [c.c_void_p, c.c_int]),
                        ("SDL_RumbleGamepad", c.c_bool, [c.c_void_p, c.c_uint16, c.c_uint16, c.c_uint32])):
                    function = getattr(sdl, name)
                    function.restype, function.argtypes = result, arguments
                cls.sdl = sdl
                break
        return cls.sdl or None

    def __init__(self):
        self.pad = None
        sdl = self.library()
        self.ready = False
        if sdl:
            # No SDL window here: without this a controller is read only while one has focus.
            sdl.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
            self.ready = bool(sdl.SDL_InitSubSystem(SDL_INIT_GAMEPAD))

    def state(self):
        """(name, {pressed buttons}, (lx, ly, rx, ry) in -1..1, (l2, r2) in 0..1), or None without a controller."""
        if not self.ready:
            return None
        sdl = self.sdl
        sdl.SDL_UpdateGamepads()            # also finds controllers connected since
        sdl.SDL_FlushEvents(0, 0xFFFF)      # nobody reads the event queue here
        if self.pad and not sdl.SDL_GamepadConnected(self.pad):
            sdl.SDL_CloseGamepad(self.pad)
            self.pad = None
        if not self.pad:
            count = ctypes.c_int(0)
            ids = sdl.SDL_GetGamepads(ctypes.byref(count))
            if ids:
                if count.value > 0:
                    self.pad = sdl.SDL_OpenGamepad(ids[0])
                sdl.SDL_free(ids)
            if not self.pad:
                return None
        pad = self.pad
        pressed = {name for i, name in enumerate(SDL_BUTTONS) if sdl.SDL_GetGamepadButton(pad, i)}
        if sdl.SDL_GetGamepadButton(pad, SDL_TOUCHPAD):
            pressed.add("touchpad")
        if sdl.SDL_GetGamepadButton(pad, SDL_MISC1):
            pressed.add("shake")
        axes = tuple(max(-1.0, sdl.SDL_GetGamepadAxis(pad, i) / 32767) for i in range(4))
        triggers = tuple(max(0.0, sdl.SDL_GetGamepadAxis(pad, i) / 32767) for i in (4, 5))
        name = (sdl.SDL_GetGamepadName(pad) or b"").decode("utf-8", "replace")
        return name, pressed, axes, triggers

    def rumble(self, seconds):
        if self.ready and self.pad:
            self.sdl.SDL_RumbleGamepad(self.pad, 0xC000, 0xC000, int(seconds * 1000))

    def close(self):
        if self.ready:
            if self.pad:
                self.sdl.SDL_RumbleGamepad(self.pad, 0, 0, 0)
                self.sdl.SDL_CloseGamepad(self.pad)
            self.sdl.SDL_QuitSubSystem(SDL_INIT_GAMEPAD)
        self.pad, self.ready = None, False


def rounded(canvas, x0, y0, x1, y1, r, **options):
    """A rounded rectangle (a smoothed polygon)."""
    points = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1,
              x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(points, smooth=True, **options)


class PadView:
    """The Controller tab's drawing: a controller whose buttons light up, sticks move and triggers
    fill. show() takes the combined state of the controller and the bound keys."""
    W, H = 720, 360
    FACE = {"triangle": "#4fd1a5", "circle": "#ff6b6b", "cross": "#7fa8ff", "square": "#e88bd7"}
    LIT_DIM = "#4a3517"   # the accent, dimmed: lit areas that carry text or other marks
    STICK_TRAVEL = 16

    def __init__(self, parent):
        c = self.canvas = tk.Canvas(parent, width=self.W, height=self.H, bg=PANEL, highlightthickness=0,
                                    takefocus=1)
        self.lights = {}    # name: [(item, idle options, lit options)]
        self.lit = {}
        self.offset = [0, 0]
        self.shake_until = 0.0
        self.sticks, self.triggers = {}, {}
        self.build()

    def light(self, name, item, idle, lit):
        self.canvas.itemconfigure(item, **idle)
        self.lights.setdefault(name, []).append((item, idle, lit))

    def build(self):
        c, W = self.canvas, self.W
        left = [(360, 92), (250, 88), (178, 82), (140, 98), (116, 150), (100, 240), (98, 300), (116, 340),
                (156, 346), (186, 326), (222, 282), (268, 266), (360, 264)]
        outline = left + [(W - x, y) for x, y in reversed(left[1:-1])]
        flat = [v for point in outline for v in point]
        c.create_polygon([v + (6 if i % 2 == 0 else 8) for i, v in enumerate(flat)], smooth=True, splinesteps=24,
                         fill="#050505")
        c.create_polygon(flat, smooth=True, splinesteps=24, fill="#1c1c1c", outline="#303030", width=2)
        small = ("Segoe UI Semibold", 8)
        tiny = ("Segoe UI", 7)

        # Triggers (a gauge that fills) and shoulder buttons.
        for side, x0 in (("l", 150), ("r", W - 250)):
            rounded(c, x0, 18, x0 + 100, 46, 10, fill="#121212", outline="#303030", width=2)
            bar = c.create_rectangle(x0 + 4, 22, x0 + 4, 42, fill=ACCENT, width=0)
            self.triggers[side + "2"] = (bar, x0 + 4, x0 + 96)
            label = c.create_text(x0 + 50, 32, text=side.upper() + "2", font=small)
            self.light(side + "2", label, {"fill": MUTED}, {"fill": BG})
            bumper = rounded(c, x0, 56, x0 + 100, 78, 10, width=2)
            self.light(side + "1", bumper, {"fill": "#252525", "outline": "#333333"}, {"fill": ACCENT, "outline": ACCENT_HOT})
            label = c.create_text(x0 + 50, 67, text=side.upper() + "1", font=small)
            self.light(side + "1", label, {"fill": MUTED}, {"fill": BG})

        # D-pad: four arms pointing at its centre.
        cx, cy = 196, 160
        arms = {"up": [(-12, -36), (12, -36), (12, -14), (0, -4), (-12, -14)],
                "down": [(-12, 36), (12, 36), (12, 14), (0, 4), (-12, 14)],
                "left": [(-36, -12), (-36, 12), (-14, 12), (-4, 0), (-14, -12)],
                "right": [(36, -12), (36, 12), (14, 12), (4, 0), (14, -12)]}
        c.create_oval(cx - 46, cy - 46, cx + 46, cy + 46, fill="#181818", outline="")
        for name, points in arms.items():
            arm = c.create_polygon([v for x, y in points for v in (cx + x, cy + y)], width=2)
            self.light(name, arm, {"fill": "#2a2a2a", "outline": "#363636"}, {"fill": ACCENT, "outline": ACCENT_HOT})

        # Face buttons with their symbols.
        cx, cy, d, r = W - 196, 160, 34, 16
        c.create_oval(cx - 56, cy - 56, cx + 56, cy + 56, fill="#181818", outline="")
        for name, (dx, dy) in (("triangle", (0, -d)), ("circle", (d, 0)), ("cross", (0, d)), ("square", (-d, 0))):
            x, y, color = cx + dx, cy + dy, self.FACE[name]
            base = c.create_oval(x - r, y - r, x + r, y + r, width=2)
            self.light(name, base, {"fill": "#252525", "outline": "#363636"}, {"fill": color, "outline": color})
            if name == "triangle":
                marks = [c.create_polygon(x, y - 8, x + 8, y + 6, x - 8, y + 6, fill="", width=2)]
                key = "outline"
            elif name == "circle":
                marks = [c.create_oval(x - 7, y - 7, x + 7, y + 7, width=2)]
                key = "outline"
            elif name == "square":
                marks = [c.create_rectangle(x - 7, y - 7, x + 7, y + 7, width=2)]
                key = "outline"
            else:
                marks = [c.create_line(x - 7, y - 7, x + 7, y + 7, width=2, capstyle="round"),
                         c.create_line(x + 7, y - 7, x - 7, y + 7, width=2, capstyle="round")]
                key = "fill"
            for mark in marks:
                self.light(name, mark, {key: color}, {key: BG})

        # Touchpad (two halves, as the game sees its clicks), light bar, Share, Options, PS.
        rounded(c, 290, 96, 430, 176, 14, fill="#202020", outline="#333333", width=2)
        self.light_bar = rounded(c, 314, 89, 406, 93, 2, fill="#2a2a2a", outline="")
        for name, x0, x1 in (("touchpad_left", 296, 359), ("touchpad_right", 361, 424)):
            half = c.create_rectangle(x0, 102, x1, 170, width=0)
            self.light(name, half, {"fill": ""}, {"fill": self.LIT_DIM})
        c.create_line(360, 104, 360, 168, fill="#2c2c2c", dash=(2, 3))
        c.create_text(360, 136, text="TOUCHPAD", font=tiny, fill="#4a4a4a")
        for name, x, text in (("share", 266, "SHARE"), ("options", W - 266, "OPTIONS")):
            pill = rounded(c, x - 5, 98, x + 5, 120, 5, width=1)
            self.light(name, pill, {"fill": "#2a2a2a", "outline": "#3a3a3a"}, {"fill": ACCENT, "outline": ACCENT_HOT})
            c.create_text(x, 130, text=text, font=tiny, fill="#5a5a5a")
        ps = c.create_oval(349, 225, 371, 247, width=2)
        self.light("ps", ps, {"fill": "#222222", "outline": "#363636"}, {"fill": ACCENT, "outline": ACCENT_HOT})
        label = c.create_text(360, 236, text="PS", font=tiny)
        self.light("ps", label, {"fill": "#6a6a6a"}, {"fill": BG})

        # Sticks: a well, a ring for the walk key's reach, the cap and its dot.
        for side, x in (("l", 282), ("r", W - 282)):
            y = 228
            c.create_oval(x - 32, y - 32, x + 32, y + 32, fill="#111111", outline="#2e2e2e", width=2)
            ring = c.create_oval(x, y, x, y, outline=ACCENT, dash=(3, 3), state="hidden")
            cap = c.create_oval(x - 20, y - 20, x + 20, y + 20, width=2)
            self.light(side + "3", cap, {"fill": "#2b2b2b", "outline": "#3c3c3c"},
                       {"fill": self.LIT_DIM, "outline": ACCENT_HOT})
            dot = c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=MUTED, outline="")
            self.sticks[side] = (x, y, ring, cap, dot)

    def set_lit(self, name, on):
        if self.lit.get(name) == on:
            return
        self.lit[name] = on
        for item, idle, lit in self.lights.get(name, ()):
            self.canvas.itemconfigure(item, **(lit if on else idle))

    def shake(self, seconds):
        self.shake_until = max(self.shake_until, time.monotonic() + seconds)

    def show(self, pressed, sticks, triggers, walk=None, connected=False):
        """pressed: lit names; sticks: {"l"/"r": (x, y)} in -1..1; triggers: {"l2"/"r2": 0..1};
        walk: the left stick's reach while the walk key holds it back (0..1), or None."""
        c = self.canvas
        shaking = time.monotonic() < self.shake_until
        offset = [random.randint(-3, 3), random.randint(-2, 2)] if shaking else [0, 0]
        if offset != self.offset:
            c.move("all", offset[0] - self.offset[0], offset[1] - self.offset[1])
            self.offset = offset
        ox, oy = offset
        for name in self.lights:
            if name not in self.triggers:
                self.set_lit(name, name in pressed)
        for name, (bar, x0, x1) in self.triggers.items():
            value = triggers.get(name, 0.0)
            c.coords(bar, x0 + ox, 22 + oy, x0 + ox + (x1 - x0) * value, 42 + oy)
            c.itemconfigure(bar, state="normal" if value > 0.01 else "hidden")
            self.set_lit(name, value > 0.5)
        for side, (x, y, ring, cap, dot) in self.sticks.items():
            sx, sy = sticks.get(side, (0.0, 0.0))
            px, py = x + ox + sx * self.STICK_TRAVEL, y + oy + sy * self.STICK_TRAVEL
            c.coords(cap, px - 20, py - 20, px + 20, py + 20)
            c.coords(dot, px - 3, py - 3, px + 3, py + 3)
            c.itemconfigure(dot, fill=ACCENT if sx * sx + sy * sy > 0.04 else MUTED)
            if side == "l" and walk is not None:
                reach = self.STICK_TRAVEL * walk + 20
                c.coords(ring, x + ox - reach, y + oy - reach, x + ox + reach, y + oy + reach)
                c.itemconfigure(ring, state="normal")
            else:
                c.itemconfigure(ring, state="hidden")
        c.itemconfigure(self.light_bar, fill="#4b7bec" if connected else "#2a2a2a")


class Launcher(tk.Tk):
    WIDTH, HEIGHT = 1180, 690   # 690: room for Video memory (14 Nov 2026)

    def __init__(self, seconds=0):
        # Its own taskbar entry and icon, not those of pythonw.exe.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TLOU.Remastered.Launcher")
        super().__init__()
        self.seconds = seconds
        self.process = None
        self.overrides = read_overrides()
        self.disabled = set()   # live toggles switched off in the debug menu
        self.was_running = False
        self.debug_window = None
        self.folders_window = None
        self.controls_window = None
        self.language = read_ui_language()
        self.folders = read_folders()
        self.title("The Last of Us Remastered")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.art = load_art(self.WIDTH, self.HEIGHT, game_folder(self.folders))
        self.game_icon = None
        self.set_icon()
        self.style()
        self.ready_message = None
        # Updates: a newer release (looked for at every start) shows the update button beside PLAY.
        self.update_release = None
        self.update_busy = False
        self.update_note = None   # (text key, values) of the status line: an update found or on its way
        self.build()
        self.refresh()
        self.after(3000, lambda: threading.Thread(target=updater.cleanup, args=(ROOT,), daemon=True).start())
        self.after(1000, self.check_updates)

    def set_icon(self):
        """The window's icon, also for the windows opened from this one."""
        if ICON.is_file():
            self.iconbitmap(default=str(ICON))
            return
        game = game_folder(self.folders)
        picture = game / GAME_ICON
        if picture.is_file():
            try:
                self.game_icon = tk.PhotoImage(file=str(picture))
                self.iconphoto(True, self.game_icon)
            except tk.TclError:
                pass
            threading.Thread(target=game_shortcut, args=(game,), daemon=True).start()

    def t(self, key):
        return TEXTS[self.language][key]

    def labels(self, options):
        """{text on screen: value} of a table keyed by text keys (other keys are shown as they are)."""
        return {self.t(k) if k in KEYS else k: v for k, v in options.items()}

    def build(self):
        """The whole window: called again when the launcher's language changes."""
        for child in self.winfo_children():
            if child not in (self.debug_window, self.folders_window, self.controls_window):
                child.destroy()
        ini, switches = read_ini(), read_switches()
        canvas = self.canvas = tk.Canvas(self, width=self.WIDTH, height=self.HEIGHT, bg=BG, highlightthickness=0)
        canvas.place(x=0, y=0)
        if self.art:
            canvas.create_image(self.WIDTH, 0, image=self.art, anchor="ne")
        x = 44
        canvas.create_text(x, 30, text="THE LAST OF US", anchor="nw", fill=TEXT, font=("Segoe UI Light", 32))
        canvas.create_text(x + 2, 84, text=f"REMASTERED   ·   {self.t('subtitle')}", anchor="nw", fill=ACCENT,
                           font=("Segoe UI Semibold", 10))

        panel = tk.Frame(self, bg=BG)
        panel.place(x=x, y=124, width=460)
        panel.columnconfigure(1, weight=1)
        row = self.section(panel, 0, self.t("language"))
        self.ui_labels = {name: code for code, name in UI_LANGUAGES.items()}
        self.ui_language = self.choice(panel, row, self.t("ui_language"), list(self.ui_labels),
                                       UI_LANGUAGES[self.language], self.change_language)
        self.system_labels = {name: code for code, name in SYSTEM_LANGUAGES.items()}
        system = ini.get("system_language", "1")
        self.system_language = self.choice(panel, row + 1, self.t("system_language"), list(self.system_labels),
                                           SYSTEM_LANGUAGES.get(system, SYSTEM_LANGUAGES["1"]))
        self.info(panel, row + 1, "system_language")
        row = self.section(panel, row + 2, self.t("video"))
        self.display_labels = self.labels(DISPLAY)
        self.display = self.choice(panel, row, self.t("display"), list(self.display_labels),
                                   key_for(self.display_labels, ini.get("display_mode"), self.t("borderless")))
        res = ini.get("output_res", "1920x1080")
        self.resolution = self.choice(panel, row + 1, self.t("resolution"),
                                      RESOLUTIONS + ([res] if res not in RESOLUTIONS else []), res)
        fps = ini.get("fps_limit", "60")
        self.fps = self.choice(panel, row + 2, self.t("fps"), FPS if fps in FPS else FPS + [fps], fps)
        self.upscaler_labels = self.labels(UPSCALERS)
        self.upscaler = self.choice(panel, row + 3, self.t("upscaler"), list(self.upscaler_labels),
                                    key_for(self.upscaler_labels, ini.get("upscaler"), self.t("off")))
        self.preset_labels = self.labels(PRESETS)
        self.preset = self.choice(panel, row + 4, self.t("quality"), list(self.preset_labels),
                                  key_for(self.preset_labels, ini.get("preset"), self.t("q_native")))
        self.vsync = self.check(panel, row + 5, self.t("vsync"), ini.get("vsync", "0") == "1")
        self.show_fps = self.check(panel, row + 6, self.t("show_fps"), ini.get("show_fps", "1") == "1")
        self.ambient_half = self.check(panel, row + 7, self.t("ambient_half"), switches["ambient_half"])
        self.quarter_depth = self.check(panel, row + 8, self.t("quarter_depth"), switches["quarter_depth"])
        self.vram_labels = {self.t(key): value for key, value in LOW_VRAM.items()}
        self.low_vram = self.choice(panel, row + 9, self.t("vram"), list(self.vram_labels),
                                    key_for(self.vram_labels, switches["low_vram"], self.t("vram_auto")),
                                    self.change_vram)
        self.info(panel, row + 9, "vram")
        self.info(panel, row + 3, "upscaler")
        self.info(panel, row + 4, "quality")
        self.info(panel, row + 5, "vsync")
        self.info(panel, row + 7, "ambient_half")
        self.info(panel, row + 8, "quarter_depth")
        row = self.section(panel, row + 10, self.t("game"))
        self.motion_blur = self.check(panel, row, self.t("motion_blur"), switches["motion_blur"])
        self.job_sleep = self.check(panel, row + 1, self.t("job_sleep"), switches["job_sleep"])
        self.profile = self.check(panel, row + 2, self.t("profile"), switches["profile"])
        self.background_shaders = self.check(panel, row + 3, self.t("background_shaders"), switches["background_shaders"])
        self.pro_shadows = self.check(panel, row + 4, self.t("pro_shadows"), switches["pro_shadows"])
        self.info(panel, row, "motion_blur")
        self.info(panel, row + 1, "job_sleep")
        self.info(panel, row + 2, "profile")
        self.info(panel, row + 3, "background_shaders")
        self.info(panel, row + 4, "pro_shadows")

        self.play_button = tk.Button(self, text=self.t("play"), command=self.play, bg=ACCENT, fg=BG,
                                     activebackground=ACCENT_HOT, activeforeground=BG, bd=0, relief="flat",
                                     font=("Segoe UI Semibold", 15), cursor="hand2", disabledforeground="#77756f")
        self.play_button.place(x=self.WIDTH - 300, y=self.HEIGHT - 88, width=256, height=52)
        # Placed beside PLAY only once a newer release is found (sync_update_button).
        self.update_button = tk.Button(self, command=self.install_update, bg=PANEL, fg=ACCENT, activebackground="#222",
                                       activeforeground=ACCENT_HOT, bd=0, relief="flat", font=("Segoe UI Semibold", 10),
                                       cursor="hand2", disabledforeground="#77756f", wraplength=170)
        self.sync_update_button()
        # Above the button, wrapped to its width: the line below it holds the links.
        self.status = canvas.create_text(self.WIDTH - 300, self.HEIGHT - 96, anchor="sw", width=256, fill=MUTED,
                                         font=("Segoe UI", 9))
        x = self.WIDTH - 44
        for text, command in ((self.t("debug"), self.open_debug), (self.t("log"), self.open_log),
                              (self.t("folders"), self.open_folders), (self.t("controls"), self.open_controls)):
            x = canvas.bbox(self.link(canvas, x, text, command))[0] - 18
        self.ready_message = self.check_install()

    def style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT, arrowcolor=ACCENT,
                        bordercolor="#2c2c2c", lightcolor=PANEL, darkcolor=PANEL, selectbackground=PANEL,
                        selectforeground=TEXT, padding=3)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", TEXT)],
                  bordercolor=[("focus", ACCENT), ("hover", "#555")])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(12, 5), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", "#222")], foreground=[("selected", TEXT)])
        style.configure("Vertical.TScrollbar", background="#2c2c2c", troughcolor=PANEL, arrowcolor=MUTED, bordercolor=PANEL)
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)

    def section(self, parent, row, title):
        tk.Label(parent, text=title.upper(), bg=BG, fg=ACCENT, font=("Segoe UI Semibold", 9)).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 3))
        return row + 1

    def choice(self, parent, row, text, values, current, command=None):
        tk.Label(parent, text=text, bg=BG, fg=TEXT, font=("Segoe UI", 10)).grid(row=row, column=0, sticky="w", pady=2)
        variable = tk.StringVar(value=current)
        box = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=24)
        box.grid(row=row, column=1, sticky="e", pady=2)
        box.bind("<<ComboboxSelected>>", lambda _: (box.selection_clear(), (command or self.save)()))
        return variable

    def check(self, parent, row, text, value):
        variable = tk.BooleanVar(value=value)
        tk.Checkbutton(parent, text=text, variable=variable, command=self.save, bg=BG, fg=TEXT, selectcolor=PANEL,
                       activebackground=BG, activeforeground=TEXT, font=("Segoe UI", 10), bd=0,
                       highlightthickness=0, anchor="w").grid(row=row, column=0, columnspan=2, sticky="w")
        return variable

    def info(self, parent, row, key):
        """A small (i) beside a main-screen option; hovering it shows what the option does."""
        texts = INFO_TEXT.get(self.language, INFO_TEXT["en"])
        text = texts.get(key) or INFO_TEXT["en"].get(key)
        if not text:
            return
        icon = tk.Label(parent, text="ⓘ", bg=BG, fg=MUTED, font=("Segoe UI", 10), cursor="question_arrow")
        icon.grid(row=row, column=2, sticky="e", padx=(6, 0))
        tip = {}

        def show(_):
            window = tk.Toplevel(self)
            window.wm_overrideredirect(True)
            window.attributes("-topmost", True)
            tk.Label(window, text=text, bg=PANEL, fg=TEXT, font=("Segoe UI", 9), justify="left",
                     wraplength=320, padx=10, pady=7, bd=1, relief="solid").pack()
            window.update_idletasks()
            x = icon.winfo_rootx() + icon.winfo_width() + 8
            y = icon.winfo_rooty() - 4
            window.wm_geometry(f"+{x}+{y}")
            icon.configure(fg=ACCENT)
            tip["window"] = window

        def hide(_):
            if tip.get("window"):
                tip.pop("window").destroy()
            icon.configure(fg=MUTED)

        icon.bind("<Enter>", show)
        icon.bind("<Leave>", hide)

    def link(self, canvas, x, text, command):
        item = canvas.create_text(x, self.HEIGHT - 26, text=text, anchor="e", fill=ACCENT, font=("Segoe UI", 9, "underline"))
        canvas.tag_bind(item, "<Button-1>", lambda _: command())
        canvas.tag_bind(item, "<Enter>", lambda _: canvas.configure(cursor="hand2"))
        canvas.tag_bind(item, "<Leave>", lambda _: canvas.configure(cursor=""))
        return item

    def change_language(self):
        self.language = self.ui_labels[self.ui_language.get()]
        self.save()
        window = self.debug_window
        if window is not None and window.winfo_exists():
            window.destroy()
        for window in (self.folders_window, self.controls_window):
            if window is not None and window.winfo_exists():
                window.destroy()
        self.build()

    def check_install(self):
        game = game_folder(self.folders)
        if not (game / "eboot.bin").is_file():
            return self.t("no_game").format(game)
        if not (ROOT / "gow3_pc/out/gow3-probe.exe").is_file():
            return self.t("no_exe")
        return ""

    def switches(self):
        return {"motion_blur": self.motion_blur.get(), "job_sleep": self.job_sleep.get(), "profile": self.profile.get(),
                "background_shaders": self.background_shaders.get(), "pro_shadows": self.pro_shadows.get(),
                "ambient_half": self.ambient_half.get(), "quarter_depth": self.quarter_depth.get(),
                "low_vram": self.vram_labels.get(self.low_vram.get(), "auto")}

    def change_vram(self):
        """Video memory: 2-3 GB also unticks the PS4 Pro's larger shadow maps."""
        if self.vram_labels.get(self.low_vram.get()) == "on":
            self.pro_shadows.set(False)
        self.save()

    def save(self):
        write_ini({"display_mode": self.display_labels[self.display.get()], "output_res": self.resolution.get(),
                   "fps_limit": self.fps.get(), "upscaler": self.upscaler_labels[self.upscaler.get()],
                   "preset": self.preset_labels[self.preset.get()], "vsync": int(self.vsync.get()),
                   "show_fps": int(self.show_fps.get()),
                   "system_language": self.system_labels[self.system_language.get()]})
        SWITCHES.write_text(json.dumps({**self.switches(), "ui_language": self.language, "folders": self.folders,
                                        "overrides": self.overrides},
                                       indent=2) + "\n", encoding="utf-8")

    def refresh(self):
        """Keeps JOGAR and the status line current, also for a game started some other way."""
        def check():
            running = self.process is not None and self.process.poll() is None or game_running()
            self.after(0, lambda: self.show_state(running))
        threading.Thread(target=check, daemon=True).start()
        self.after(2000, self.refresh)

    def show_state(self, running):
        # probe_tlou.py deleted the file at the start: the menu's choices go back once it runs.
        if running and not self.was_running and self.disabled:
            self.after(3000, lambda: write_toggles(toggle_mask(self.disabled)))
        self.was_running = running
        if self.ready_message:
            self.canvas.itemconfigure(self.status, text=self.ready_message, fill=ERROR)
            self.play_button.config(state="disabled", text=self.t("play"), bg=DISABLED)
        elif self.update_busy and self.update_note:
            self.canvas.itemconfigure(self.status, text=self.note(), fill=MUTED)
            self.play_button.config(state="disabled", text=self.t("play"), bg=DISABLED)
        elif running:
            self.canvas.itemconfigure(self.status, text=self.t("running").capitalize(), fill=MUTED)
            self.play_button.config(state="disabled", text=self.t("running"), bg=DISABLED)
        elif self.update_note:
            self.canvas.itemconfigure(self.status, text=self.note(), fill=ACCENT)
            self.play_button.config(state="normal", text=self.t("play"), bg=ACCENT)
        else:
            self.canvas.itemconfigure(self.status, text=self.t("ready"), fill=MUTED)
            self.play_button.config(state="normal", text=self.t("play"), bg=ACCENT)

    def play(self):
        self.save()
        game = game_folder(self.folders)
        if merged_with_base_executable(game):
            messagebox.showerror(self.title(), self.t("merged_old_exe").format(game.name), parent=self)
            return
        self.play_button.config(state="disabled", text=self.t("starting"), bg=DISABLED)
        env = game_env(self.switches(), overrides=self.overrides, folders=self.folders)
        if self.profile.get():
            Path(PROFILE_ENV["GOW3_DUMP_DIR"]).mkdir(parents=True, exist_ok=True)
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("w", encoding="utf-8") as output:
            self.process = subprocess.Popen(
                game_command(python_exe(), self.folders, self.seconds), cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, creationflags=0x08000000)

    def note(self):
        key, *values = self.update_note
        return self.t(key).format(*values)

    def check_updates(self):
        """Looks for a newer release in the background; a copy run from the source tree (no VERSION)
        is never updated. Offline or GitHub's rate limit: nothing is shown."""
        current = updater.installed_version(ROOT)
        if current is None or self.update_busy:
            return
        result = []

        def work():
            try:
                result.append(updater.latest_release(current))
            except Exception:   # offline, rate limit, a bad answer: try again next start
                result.append(None)

        def done():
            if not result:
                self.after(250, done)
                return
            if result[0] and not self.update_busy:
                self.update_release = result[0]
                self.update_note = ("update_available", updater.version_text(result[0]["version"]))
                self.sync_update_button()
                self.show_state(self.process is not None and self.process.poll() is None)
        threading.Thread(target=work, daemon=True).start()
        self.after(250, done)

    def sync_update_button(self):
        """Shows the update button beside PLAY once a newer release was found (also after build())."""
        if not self.update_release:
            self.update_button.place_forget()
            return
        self.update_button.config(text=self.t("update_button").format(updater.version_text(self.update_release["version"])),
                                  state="disabled" if self.update_busy else "normal")
        self.update_button.place(x=self.WIDTH - 300 - 192, y=self.HEIGHT - 88, width=180, height=52)

    def install_update(self):
        release = self.update_release
        if not release or self.update_busy:
            return
        if self.process is not None and self.process.poll() is None or game_running():
            messagebox.showinfo(self.title(), self.t("update_close_game"), parent=self)
            return
        version = updater.version_text(release["version"])
        if not messagebox.askyesno(self.title(), self.t("update_confirm").format(version, round(release["size"] / 1e6))
                                   + f"\n\n{release['page']}", parent=self):
            return
        self.save()   # settings live in analysis/, which the update never touches
        self.update_busy = True
        self.update_note = ("update_downloading", "0%")
        self.sync_update_button()
        self.show_state(False)
        state = {"progress": (0, release["size"]), "error": None, "staged": None}

        def progress(done, total):
            state["progress"] = (done, total)

        def work():
            try:
                archive = updater.download(release, ROOT, progress)
                state["staged"] = updater.stage(archive, ROOT, release["version"])
            except Exception as error:
                state["error"] = str(error) or error.__class__.__name__

        def tick():
            if state["error"]:
                self.update_busy = False
                self.update_note = ("update_available", version)
                self.sync_update_button()
                self.show_state(False)
                messagebox.showerror(self.title(), self.t("update_failed").format(state["error"]), parent=self)
                return
            if state["staged"]:
                self.update_note = ("update_installing",)
                self.show_state(False)
                try:
                    updater.start_apply(ROOT, state["staged"])
                except OSError as error:
                    state["error"] = str(error)
                    tick()
                    return
                self.destroy()   # apply_update.py waits for this process to exit
                return
            done, total = state["progress"]
            self.update_note = ("update_downloading", f"{100 * done / total:.0f}%" if total else f"{done / 1e6:.0f} MB")
            self.show_state(False)
            self.after(200, tick)

        threading.Thread(target=work, daemon=True).start()
        self.after(200, tick)

    def open_debug(self):
        if self.debug_window is not None and self.debug_window.winfo_exists():
            self.debug_window.lift()
            return
        self.debug_window = window = tk.Toplevel(self, bg=BG)
        window.title(self.t("debug"))
        window.geometry(f"640x640+{self.winfo_rootx() + 40}+{self.winfo_rooty() + 20}")
        tk.Label(window, text=self.t("warn"), bg=BG, fg=ERROR, font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(12, 8))
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=14)
        self.live_tab(tabs)
        self.startup_tab(tabs)

    def scroll_area(self, tabs, title, note):
        outer = tk.Frame(tabs, bg=PANEL)
        tabs.add(outer, text=title)
        tk.Label(outer, text=note, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), justify="left").pack(anchor="w", padx=10, pady=(8, 4))
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=PANEL)
        body.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.bind("<Enter>", lambda _: canvas.bind_all(
            "<MouseWheel>", lambda e: canvas.winfo_exists() and canvas.yview_scroll(-e.delta // 120, "units")))
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 8))
        return outer, body

    def debug_text(self, text):
        """A debug or controls window text in the launcher's language."""
        table = DEBUG_TEXT if text in DEBUG_TEXT else CONTROL_TEXT
        if self.language in DEBUG_LANGUAGES and text in table:
            return table[text][DEBUG_LANGUAGES.index(self.language)]
        return text

    def group(self, body, title):
        tk.Label(body, text=self.debug_text(title), bg=PANEL, fg=ACCENT, font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(8, 2))

    def option(self, body, text, variable, command):
        tk.Checkbutton(body, text=self.debug_text(text), variable=variable, command=command, bg=PANEL, fg=TEXT, selectcolor=BG,
                       activebackground=PANEL, activeforeground=TEXT, font=("Segoe UI", 10), bd=0,
                       highlightthickness=0, anchor="w").pack(anchor="w")

    def reset_link(self, outer, command):
        button = tk.Button(outer, text=self.t("reset"), command=command, bg=BG, fg=TEXT, bd=0, cursor="hand2",
                           activebackground=BG, activeforeground=ACCENT, font=("Segoe UI", 9))
        button.place(relx=1.0, x=-30, y=6, anchor="ne")

    def live_tab(self, tabs):
        outer, body = self.scroll_area(tabs, self.t("tab_live"), self.t("note_live"))
        variables = {}

        def apply(bit):
            (self.disabled.discard if variables[bit].get() else self.disabled.add)(bit)
            write_toggles(toggle_mask(self.disabled))

        for title, items in TOGGLES:
            self.group(body, title)
            for text, bit in items:
                variables[bit] = tk.BooleanVar(value=bit not in self.disabled)
                self.option(body, text, variables[bit], lambda b=bit: apply(b))

        def reset():
            self.disabled.clear()
            write_toggles(0)
            for variable in variables.values():
                variable.set(True)

        self.reset_link(outer, reset)

    def startup_tab(self, tabs):
        outer, body = self.scroll_area(tabs, self.t("tab_startup"), self.t("note_startup"))
        variables = {}

        def apply(option):
            name, on_value, off_value, default = option[1:]
            wanted = variables[name].get()
            if wanted == default:
                self.overrides.pop(name, None)
            else:
                self.overrides[name] = on_value if wanted else off_value
            self.save()

        for title, items in STARTUP:
            self.group(body, title)
            for option in items:
                text, name, on_value, _, default = option
                current = self.overrides.get(name)
                variables[name] = tk.BooleanVar(value=default if current is None else current == on_value)
                self.option(body, text, variables[name], lambda o=option: apply(o))

        def reset():
            self.overrides.clear()
            self.save()
            for _, items in STARTUP:
                for _, name, _, _, default in items:
                    variables[name].set(default)

        self.reset_link(outer, reset)

    def open_controls(self):
        if self.controls_window is not None and self.controls_window.winfo_exists():
            self.controls_window.lift()
            return
        self.controls_window = window = tk.Toplevel(self, bg=BG)
        window.title(self.t("controls"))
        window.geometry(f"780x720+{self.winfo_rootx() + 40}+{self.winfo_rooty() + 20}")
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=14, pady=14)
        # Aim assist (aim_assist in tlou.ini; tools/probe_tlou.py patches the game's aim settings
        # at start): one switch, shown in the Controller and Mouse tabs.
        self.aim_assist = tk.BooleanVar(value=read_ini().get("aim_assist", "1") != "0")
        self.pad_tab(window, tabs)
        self.keys_tab(window, tabs)
        self.mouse_tab(tabs)

    def save_aim_assist(self):
        write_ini({"aim_assist": int(self.aim_assist.get())})

    def pad_tab(self, window, tabs):
        """A drawn controller that follows the controller (SDL3, while the tab is shown) and the bound keys
        and mouse buttons, and the controller's vibration (vibration in tlou.ini, runtime_pad.c)."""
        outer = tk.Frame(tabs, bg=PANEL)
        tabs.add(outer, text=self.debug_text("Controle"))
        tk.Label(outer, text=self.debug_text("Mexa no controle ou aperte as teclas e os botões do mouse configurados: "
                                             "o desenho mostra cada comando."),
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=560, justify="left").pack(anchor="w", padx=10, pady=(8, 2))
        view = PadView(outer)
        view.canvas.pack(padx=10, pady=(6, 0))
        status = tk.Label(outer, bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        status.pack(pady=(2, 0))

        card = tk.Frame(outer, bg=BG, highlightthickness=1, highlightbackground="#262626")
        card.pack(fill="x", padx=10, pady=(16, 0))
        tk.Label(card, text=self.debug_text("Vibração").upper(), bg=BG, fg=ACCENT,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=14, pady=(10, 2))
        row = tk.Frame(card, bg=BG)
        row.pack(anchor="w", fill="x", padx=12, pady=(0, 12))
        vibration = tk.BooleanVar(value=read_ini().get("vibration", "1") != "0")
        state = {"reader": None, "held": set(), "flash": {}, "lookup": {}, "walk_toggle": False, "walk_speed": 0.75,
                 "walking": False, "walk_was": False, "job": None}

        def test():
            view.shake(0.6)
            if state["reader"]:
                state["reader"].rumble(0.6)

        def save():
            write_ini({"vibration": int(vibration.get())})
            tester.config(state="normal" if vibration.get() else "disabled")

        tk.Checkbutton(row, text=self.debug_text("Vibração do controle"), variable=vibration, command=save, bg=BG,
                       fg=TEXT, selectcolor=PANEL, activebackground=BG, activeforeground=TEXT, font=("Segoe UI", 10),
                       bd=0, highlightthickness=0, takefocus=0).pack(side="left")
        tester = tk.Button(row, text=self.debug_text("Testar vibração"), command=test, bg=PANEL, fg=TEXT, bd=0,
                           cursor="hand2", activebackground="#222", activeforeground=ACCENT, font=("Segoe UI", 9),
                           padx=12, pady=3, takefocus=0, disabledforeground="#555")
        tester.pack(side="left", padx=(14, 0))
        tk.Label(row, text=self.debug_text("Vale na próxima vez que o jogo abrir."), bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(14, 0))
        tester.config(state="normal" if vibration.get() else "disabled")

        aim_card = tk.Frame(outer, bg=BG, highlightthickness=1, highlightbackground="#262626")
        aim_card.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(aim_card, text=self.debug_text("Mira").upper(), bg=BG, fg=ACCENT,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Checkbutton(aim_card, text=self.debug_text("Assistência de mira"), variable=self.aim_assist,
                       command=self.save_aim_assist, bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                       activeforeground=TEXT, font=("Segoe UI", 10), bd=0, highlightthickness=0,
                       takefocus=0).pack(anchor="w", padx=12)
        tk.Label(aim_card, text=self.debug_text("Desligada: o jogo não ajuda a mira a seguir os inimigos. "
                                                "Vale na próxima vez que o jogo abrir."),
                 bg=BG, fg=MUTED, font=("Segoe UI", 9), wraplength=560, justify="left").pack(anchor="w", padx=14,
                                                                                              pady=(0, 10))

        def shown():
            return outer.winfo_exists() and tabs.select() == str(outer)

        def source_key(kind, name):
            return kind, name.lower()

        def load_bindings():
            ini = read_ini()
            state["lookup"] = {source_key(*source): action for action, sources in read_controls(ini).items()
                               for source in sources if source}
            state["walk_toggle"] = ini.get("walk_toggle", "0") == "1"
            try:
                state["walk_speed"] = min(max(float(ini.get("walk_speed", "0.75")), 0.1), 1.0)
            except ValueError:
                state["walk_speed"] = 0.75

        def key_press(event):
            name = key_from_event(event.keysym, event.keycode)
            if name:
                state["held"].add(source_key("key", name))
            return "break"    # no focus moves, no Space on a button, no Alt menu while trying keys

        def key_release(event):
            name = key_from_event(event.keysym, event.keycode)
            if name:
                state["held"].discard(source_key("key", name))
            return "break"

        def mouse(event, down):
            name = MOUSE_BUTTONS.get(event.num)
            if name:
                (state["held"].add if down else state["held"].discard)(source_key("mouse", name))
            view.canvas.focus_set()

        def wheel(event):
            state["flash"][source_key("mouse", "wheel_up" if event.delta > 0 else "wheel_down")] = time.monotonic() + 0.15

        canvas = view.canvas
        canvas.bind("<KeyPress>", key_press)
        canvas.bind("<KeyRelease>", key_release)
        canvas.bind("<ButtonPress>", lambda e: mouse(e, True))
        canvas.bind("<ButtonRelease>", lambda e: mouse(e, False))
        canvas.bind("<MouseWheel>", wheel)
        canvas.bind("<FocusOut>", lambda _: state["held"].clear())

        def axis(actions, negative, positive):
            return (positive in actions) - (negative in actions)

        def tick():
            state["job"] = None
            if not shown():
                return
            now = time.monotonic()
            state["flash"] = {s: until for s, until in state["flash"].items() if until > now}
            actions = {state["lookup"][s] for s in state["held"] | set(state["flash"]) if s in state["lookup"]}
            pad = state["reader"].state() if state["reader"] else None
            pressed, (lx, ly, rx, ry), (l2, r2) = (pad[1], pad[2], pad[3]) if pad else (set(), (0, 0, 0, 0), (0, 0))
            pressed = set(pressed) | (actions & {"cross", "circle", "square", "triangle", "l1", "r1", "l3", "r3",
                                                 "options", "touchpad_left", "touchpad_right", "up", "down", "left",
                                                 "right"})
            if "touchpad" in pressed:
                pressed |= {"touchpad_left", "touchpad_right"}
            if "share" in pressed:      # Back/View/Share: the right side of the touchpad (runtime_pad.c)
                pressed.add("touchpad_right")
            l2, r2 = max(l2, 1.0 if "l2" in actions else 0.0), max(r2, 1.0 if "r2" in actions else 0.0)
            lx, ly = lx + axis(actions, "ls_left", "ls_right"), ly + axis(actions, "ls_up", "ls_down")
            rx, ry = rx + axis(actions, "rs_left", "rs_right"), ry + axis(actions, "rs_up", "rs_down")
            walk_held = "walk" in actions
            if state["walk_toggle"]:
                if walk_held and not state["walk_was"]:
                    state["walking"] = not state["walking"]
            else:
                state["walking"] = walk_held
            state["walk_was"] = walk_held
            sticks = {}
            for side, x, y, limit in (("l", lx, ly, state["walk_speed"] if state["walking"] else 1.0), ("r", rx, ry, 1.0)):
                length = (x * x + y * y) ** 0.5
                scale = limit / length if length > limit else 1.0
                sticks[side] = (x * scale, y * scale)
            if "shake" in actions or "shake" in pressed:
                view.shake(0.1)
            view.show(pressed, sticks, {"l2": l2, "r2": r2}, state["walk_speed"] if state["walking"] else None,
                      connected=bool(pad))
            if state["reader"] is None or not state["reader"].ready:
                text = self.debug_text("Leitura do controle indisponível (SDL3.dll não encontrado).")
            elif pad:
                text = self.debug_text("Controle conectado: {}").format(pad[0] or "?")
            else:
                text = self.debug_text("Nenhum controle conectado.")
            if status.cget("text") != text:
                status.config(text=text)
            state["job"] = outer.after(16, tick)

        def stop():
            if state["job"]:
                outer.after_cancel(state["job"])
                state["job"] = None
            if state["reader"]:
                state["reader"].close()
                state["reader"] = None
            state["held"].clear()

        def changed(_=None):
            if shown():
                load_bindings()
                if state["reader"] is None:
                    state["reader"] = PadReader()
                canvas.after_idle(canvas.focus_set)
                if state["job"] is None:
                    tick()
            else:
                stop()

        tabs.bind("<<NotebookTabChanged>>", changed, add="+")
        outer.after_idle(changed)
        outer.bind("<Destroy>", lambda e: stop() if e.widget is outer else None)

        def reset():
            vibration.set(True)
            save()
            self.aim_assist.set(True)
            self.save_aim_assist()

        self.reset_link(outer, reset)
        window.pad_view, window.pad_state = view, state   # (tests)

    def keys_tab(self, window, tabs):
        """Two keys or mouse buttons per action: click a box, then press it (Esc cancels, Delete clears)."""
        outer, body = self.scroll_area(tabs, self.debug_text("Teclado e mouse"), self.debug_text(
            "Clique numa caixa e pressione a tecla ou o botão do mouse. Esc cancela, Delete limpa.\n"
            "Movimento e câmera aceitam só teclas. Vale na próxima vez que o jogo abrir."))
        bindings = read_controls(read_ini())
        keys_only = {action for _, rows in CONTROLS for action, _, only in rows if only}
        boxes = {}
        capture = []   # [action, slot] while a box waits for its key or button

        def show():
            for (action, slot), box in boxes.items():
                source = bindings[action][slot]
                if source is None:
                    box.config(text="—", fg=MUTED)
                else:
                    kind, name = source
                    box.config(text=self.debug_text(MOUSE_NAMES[name] if kind == "mouse" else key_label(name)), fg=TEXT)

        def assign(source):
            action, slot = capture
            capture.clear()
            if source is not None:   # a key or button does one thing: it leaves the action that had it
                same = (source[0], source[1].lower())
                for sources in bindings.values():
                    sources[:] = [None if s and (s[0], s[1].lower()) == same else s for s in sources]
            bindings[action][slot] = source
            write_ini(dict(zip(("key_map", "mouse_map"), control_maps(bindings))))
            show()

        def press_box(event, action, slot):
            if capture:
                return press_mouse(MOUSE_BUTTONS.get(event.num))
            if event.num == 1:
                capture[:] = [action, slot]
                boxes[action, slot].config(text=self.debug_text("Pressione..."), fg=ACCENT)
                boxes[action, slot].focus_set()
            return "break"

        def press_mouse(name):
            if not capture:
                return None
            if name is not None and capture[0] not in keys_only:
                assign(("mouse", name))
            return "break"

        def press_key(event):
            if not capture:
                return None
            if event.keysym == "Escape":
                capture.clear()
                show()
            elif event.keysym == "Delete":
                assign(None)
            else:
                name = key_from_event(event.keysym, event.keycode)
                if name:
                    assign(("key", name))
            return "break"

        for title, rows in CONTROLS:
            self.group(body, title)
            for action, text, _ in rows:
                row = tk.Frame(body, bg=PANEL)
                row.pack(anchor="w", pady=1)
                tk.Label(row, text=self.debug_text(text), width=32, anchor="w", bg=PANEL, fg=TEXT,
                         font=("Segoe UI", 10)).pack(side="left")
                for slot in range(CONTROL_SLOTS):
                    box = tk.Label(row, width=26, bg=BG, fg=TEXT, font=("Segoe UI", 9), cursor="hand2", pady=2,
                                   highlightthickness=1, highlightbackground="#2c2c2c")
                    box.pack(side="left", padx=(6, 0))
                    box.bind("<ButtonPress>", lambda e, a=action, s=slot: press_box(e, a, s))
                    boxes[action, slot] = box
        window.bind("<KeyPress>", press_key)
        window.bind("<ButtonPress>", lambda e: press_mouse(MOUSE_BUTTONS.get(e.num)))
        window.bind("<MouseWheel>", lambda e: press_mouse("wheel_up" if e.delta > 0 else "wheel_down"))
        walk_toggle, walk_speed, save_walk = self.walk_options(body)

        def reset():
            capture.clear()
            write_ini({"key_map": "", "mouse_map": ""})
            bindings.clear()
            bindings.update(read_controls({}))
            walk_toggle.set(0)
            walk_speed.set(75)
            save_walk()
            show()

        self.reset_link(outer, reset)
        show()
        window.press_key, window.press_box = press_key, press_box   # (tests)

    def walk_options(self, body):
        """The walk key (runtime_pad.c "walk"): held or toggled by each press (walk_toggle), and how far
        the left stick goes while walking (walk_speed). Returns the variables and their save."""
        ini = read_ini()
        try:
            speed = round(float(ini.get("walk_speed", "0.75")) * 100)
        except ValueError:
            speed = 75
        toggle = tk.IntVar(value=1 if ini.get("walk_toggle", "0") == "1" else 0)
        speed = tk.IntVar(value=min(max(speed, 10), 100))
        pending = []

        def save():
            write_ini({"walk_toggle": toggle.get(), "walk_speed": f"{speed.get() / 100:.2f}"})

        def save_later(_=None):   # the slider writes once it rests
            for job in pending:
                self.after_cancel(job)
            pending[:] = [self.after(300, save)]

        self.group(body, "Tecla de andar")
        for value, text in ((0, "Segurar a tecla"), (1, "Alternar ao apertar")):
            tk.Radiobutton(body, text=self.debug_text(text), variable=toggle, value=value, command=save, bg=PANEL,
                           fg=TEXT, selectcolor=BG, activebackground=PANEL, activeforeground=TEXT,
                           font=("Segoe UI", 10), bd=0, highlightthickness=0, anchor="w").pack(anchor="w")
        self.group(body, "Velocidade ao andar (quanto o analógico esquerdo vai)")
        row = tk.Frame(body, bg=PANEL)
        row.pack(anchor="w")
        tk.Scale(row, from_=10, to=100, resolution=5, orient="horizontal", length=420, variable=speed,
                 showvalue=False, command=save_later, bg=PANEL, fg=TEXT, troughcolor=BG, activebackground=ACCENT,
                 highlightthickness=0, bd=0).pack(side="left")
        tk.Label(row, textvariable=speed, width=5, anchor="e", bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(side="left")
        tk.Label(row, text="%", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")
        return toggle, speed, save

    def mouse_tab(self, tabs):
        outer, body = self.scroll_area(tabs, self.debug_text("Mouse"), self.debug_text("Vale na próxima vez que o jogo abrir."))
        ini = read_ini()

        def percent(key):
            try:
                return round(float(ini.get(key, "1")) * 100)
            except ValueError:
                return 100

        enabled = tk.BooleanVar(value=ini.get("mouse", "1") != "0")
        invert = tk.BooleanVar(value=ini.get("mouse_invert", "0") == "1")
        invert_x = tk.BooleanVar(value=ini.get("mouse_invert_x", "0") == "1")
        sens, aim = tk.IntVar(value=percent("mouse_sens")), tk.IntVar(value=percent("mouse_aim_sens"))
        pending = []

        def save():
            write_ini({"mouse": int(enabled.get()), "mouse_sens": f"{sens.get() / 100:.2f}",
                       "mouse_aim_sens": f"{aim.get() / 100:.2f}", "mouse_invert": int(invert.get()),
                       "mouse_invert_x": int(invert_x.get())})

        def save_later(_=None):   # a slider writes once it rests
            for job in pending:
                self.after_cancel(job)
            pending[:] = [self.after(300, save)]

        tk.Frame(body, bg=PANEL, height=6).pack()
        self.option(body, "Usar o mouse", enabled, save)
        for text, variable, top in (("Sensibilidade do mouse", sens, 500), ("Sensibilidade ao mirar", aim, 1000)):
            self.group(body, text)
            row = tk.Frame(body, bg=PANEL)
            row.pack(anchor="w")
            tk.Scale(row, from_=25, to=top, resolution=5, orient="horizontal", length=420, variable=variable,
                     showvalue=False, command=save_later, bg=PANEL, fg=TEXT, troughcolor=BG, activebackground=ACCENT,
                     highlightthickness=0, bd=0).pack(side="left")
            tk.Label(row, textvariable=variable, width=5, anchor="e", bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, text="%", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")
        tk.Frame(body, bg=PANEL, height=8).pack()
        self.option(body, "Inverter eixo vertical", invert, save)
        self.option(body, "Inverter eixo horizontal", invert_x, save)
        tk.Frame(body, bg=PANEL, height=8).pack()
        self.option(body, "Assistência de mira", self.aim_assist, self.save_aim_assist)
        tk.Label(body, text=self.debug_text("Desligada: o jogo não ajuda a mira a seguir os inimigos. "
                                            "Vale na próxima vez que o jogo abrir."),
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=520, justify="left").pack(anchor="w", padx=(22, 0))

        def reset():
            enabled.set(True)
            invert.set(False)
            invert_x.set(False)
            sens.set(100)
            aim.set(100)
            save()

        self.reset_link(outer, reset)

    def open_folders(self):
        if self.folders_window is not None and self.folders_window.winfo_exists():
            self.folders_window.lift()
            return
        self.folders_window = window = tk.Toplevel(self, bg=BG)
        window.title(self.t("folders"))
        window.resizable(False, False)
        window.geometry(f"+{self.winfo_rootx() + 60}+{self.winfo_rooty() + 90}")
        window.columnconfigure(0, weight=1)
        for row, (key, (text, default)) in enumerate(FOLDERS.items()):
            self.folder_row(window, row, key, text, default)
        tk.Label(window, text=self.t("folders_note"), bg=BG, fg=MUTED, font=("Segoe UI", 9), justify="left",
                 wraplength=640).grid(row=len(FOLDERS), column=0, columnspan=3, sticky="w", padx=16, pady=(4, 14))

    def folder_row(self, window, row, key, text, default):
        area = tk.Frame(window, bg=BG)
        area.grid(row=row, column=0, sticky="ew", padx=16, pady=(12, 4))
        area.columnconfigure(0, weight=1)
        tk.Label(area, text=text and self.t(text), bg=BG, fg=ACCENT, font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w")
        shown = tk.StringVar(value=self.folders.get(key) or str(default))
        tk.Entry(area, textvariable=shown, state="readonly", readonlybackground=PANEL, fg=TEXT, bd=0, width=70,
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="ew", ipady=4)
        error = tk.Label(area, bg=BG, fg=ERROR, font=("Segoe UI", 9))
        error.grid(row=2, column=0, sticky="w")

        def change(value):
            if key == "game" and value and not (Path(value) / "eboot.bin").is_file():
                error.config(text=self.t("no_eboot"), fg=ERROR)
                return
            error.config(text="", fg=ERROR)
            if value:
                self.folders[key] = value
            else:
                self.folders.pop(key, None)
            shown.set(value or str(default))
            self.save()
            if key == "game":
                self.art = load_art(self.WIDTH, self.HEIGHT, game_folder(self.folders))
                self.set_icon()
                self.build()

        def browse():
            chosen = filedialog.askdirectory(parent=window, initialdir=shown.get())
            if chosen:
                change(str(Path(chosen)))

        def clear():
            if game_running():
                error.config(text=self.t("clear_running"), fg=ERROR)
                return
            entries = cache_entries(shown.get())
            if not entries:
                error.config(text=self.t("clear_empty"), fg=MUTED)
                return
            if not messagebox.askyesno(self.t("cache_folder"), self.t("clear_confirm"), parent=window):
                return
            ok = recycle(entries)
            error.config(text=self.t("clear_done" if ok else "clear_failed"), fg=MUTED if ok else ERROR)

        buttons = [(self.t("browse"), browse), (self.t("default_folder"), lambda: change(""))]
        if key == "cache":
            buttons.append((self.t("clear_cache"), clear))
        for column, (label, command) in enumerate(buttons):
            tk.Button(area, text=label, command=command, bg=PANEL, fg=TEXT, bd=0, cursor="hand2", activebackground="#222",
                      activeforeground=ACCENT, font=("Segoe UI", 9), padx=10).grid(row=1, column=column + 1, padx=(6, 0), ipady=2)
        area.change = change    # (tests)

    def open_log(self):
        if LOG.is_file():
            os.startfile(LOG)


def main():
    seconds = 0
    if "--seconds" in sys.argv:
        seconds = int(sys.argv[sys.argv.index("--seconds") + 1])
    app = Launcher(seconds)
    if "--close-after" in sys.argv:
        app.after(int(sys.argv[sys.argv.index("--close-after") + 1]) * 1000, app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
