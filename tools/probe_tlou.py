#!/usr/bin/env python3
"""The Last of Us Remastered (CUSA00552): prepare the executable and start it in the runtime.

Without --boot it only lists the PS4 imports the runtime does not provide (--check-imports).
The update (a folder next to the game named like it plus -patch) is laid over the game: its
eboot.bin, sce_module and sce_sys are prepared, and the runtime reads its files over the game's.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPDATE_SUFFIXES = ("-patch", "-update")


def find_update(game):
    chosen = os.environ.get("TLOU_UPDATE", "")
    if chosen == "0":
        return None
    candidates = [Path(chosen)] if chosen else [game.parent / (game.name + s) for s in UPDATE_SUFFIXES]
    for candidate in candidates:
        if (candidate / "eboot.bin").is_file() and (candidate / "sce_sys/param.sfo").is_file():
            return candidate.resolve()
    return None


def base_executable(eboot):
    """True when `eboot` is version 1.00's executable: it imports sceRemoteplayGetConnectionStatus
    (NID g3PNjYKWqnQ), which 1.11's no longer does."""
    try:
        with open(eboot, "rb") as f:
            tail = b""
            while chunk := f.read(1 << 22):
                if b"g3PNjYKWqnQ" in tail + chunk:
                    return True
                tail = chunk[-10:]
    except OSError:
        pass
    return False


def merged_with_base_executable(game):
    """A game folder the update was merged into (it holds patch.psarc) whose eboot.bin is still
    1.00's: the merge kept the game's executable (12 Nov 2026 reports; 1.00 then stopped at
    sceRemoteplayGetConnectionStatus)."""
    return (game / "patch.psarc").is_file() and base_executable(game / "eboot.bin")


def app_version(folder):
    """APP_VER from the folder's sce_sys/param.sfo, or None."""
    sys.path.insert(0, str(ROOT / "gow3_pc/scripts"))
    try:
        from prepare import sfo
        return sfo((folder / "sce_sys/param.sfo").read_bytes()).get("APP_VER")
    except (OSError, ValueError, ImportError):
        return None
    finally:
        sys.path.pop(0)


def boot_view(base, update):
    """Copies of what the prepare scripts read (eboot.bin, sce_module, sce_sys), the update's
    over the game's, in analysis/views (copied again only when a source changes)."""
    chosen = {}
    for layer in (base, update) if update else (base,):
        for entry in layer.iterdir():
            if entry.is_file() and entry.name.lower() == "eboot.bin":
                chosen["eboot.bin"] = ("eboot.bin", entry)
            elif entry.is_dir() and entry.name.lower() in ("sce_module", "sce_sys"):
                for path in entry.rglob("*"):
                    if path.is_file():
                        relative = path.relative_to(layer)
                        chosen[str(relative).lower()] = (str(relative), path)
    view = ROOT / "analysis/views" / ("boot-" + (update or base).name)
    for relative, source in chosen.values():
        dest = view / relative
        wanted = source.stat()
        try:
            current = dest.stat()
            if current.st_size == wanted.st_size and abs(current.st_mtime - wanted.st_mtime) <= 2:
                continue
        except OSError:
            pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        os.utime(dest, ns=(wanted.st_atime_ns, wanted.st_mtime_ns))
    return view


def convert_frames(capture):
    """The presented frames (output_NNN_WxH.raw, RGBA) as PNG files next to them."""
    import re
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed: frames stay as .raw files")
        return
    for raw in sorted(capture.glob("output*.raw")):
        size = re.search(r"_(\d+)x(\d+)\.raw$", raw.name)
        if not size:
            continue
        w, h = int(size.group(1)), int(size.group(2))
        data = raw.read_bytes()
        if len(data) < w * h * 4:
            continue
        image = Image.frombytes("RGBA", (w, h), data[:w * h * 4]).convert("RGB")
        image.thumbnail((1280, 720))
        image.save(raw.with_suffix(".png"))
    print(f"Frames: {capture}")


# Changes to the game's executable before it starts: per change, image offsets (eboot vaddr), the
# bytes that must be there, and the bytes written instead (None: only checked).
# Cheats from the GoldHEN cheat repository, by Celogamez: json/CUSA00552_01.11.json
# (https://github.com/GoldHEN/GoldHEN_Cheat_Repository).
CHEATS = {
    # "Flashlight Does Not Turn Off": the flashlight's battery store (vmovss [r13+0x7d4], xmm2)
    # writes xmm1, the level before the drain, so the flashlight never flickers and needs no shake
    # of a DualShock 4's motion sensors. The cheat jumps to a code cave holding that instruction;
    # it is the same length, so it is written in place.
    # The countdown it holds (+0x7d4 of the player's flashlight) is the wait of the flicker
    # sequence's current step: about 360-400 s before the first flicker in the areas checked on
    # 17 Nov 2026, then short steps, and at the last step (31) a flicker every few seconds until
    # the controller is shaken. tools/flashlight_probe.py --force starts a flicker in a running game.
    "flashlight": [(0x697B98, "C4C17A1195D4070000", "C4C17A118DD4070000")],
}
# Idle job workers sleep (TLOU_JOB_SLEEP=0 keeps them spinning). The engine's 6 NdJobWorkerThreads spin on their job
# queues with `pause` unless the job manager's sleep flag (byte +0x59) is on; with it on, an idle
# worker waits on the manager's condition variable after 1000 spins (10 ms timeout) and every job
# submission wakes them. On a PS4 each worker owns a core; here they took about half of their time
# spinning (pause at 0xC87D65). The flag is treated as on in the worker loop, its re-check before
# waiting and the submit path, as in the Uncharted ports of the same engine.
JOB_SLEEP = [(0xC87D63, "7304", "EB04"), (0xC87D96, "741B", "9090"), (0xC873E1, "742D", "9090")]
# Every language in the game's menus (issue #156; TLOU_ALL_LANGUAGES=0 keeps the Americas set). The
# executable holds one product config per regional release (product-config.cpp, registered at
# 0x41340) and picks it by USER_DEFINED_PARAM_1 of param.sfo (552 here). Its language mask (+0x1b0)
# is English, French, Latin American Spanish and Brazilian Portuguese for CUSA00552; bits 0-20 are
# the game's 21 languages (languagemanager.cpp table at 0xbe8b40). The text of all of them is in
# core.psarc and patch.psarc. A dub (German, Italian, Spanish, Portuguese, Polish, Russian,
# Japanese) needs the other region's <language>-sp/-mp.psarc in sound1/streams/speech; without it
# the runtime's PlayGo reports that chunk as not downloaded and the game speaks English. Korean
# and Chinese are offered only with that system language (GOW3_LANGUAGE 9, 10 or 11).
ALL_LANGUAGES = [(0x1518720, "03000300", "FFFF1F00")]
# The dialogue languages of the menu (languagemanager, 0xbe98e4..): one `cmp byte [rdx+X], 0; je`
# per dub adds it to the list when its language is on. A dub whose speech file is in neither the
# game nor the update folder is left out (the je becomes a jmp), so the menu does not offer it: the
# game would ask to download that "language pack" and never get it. {speech name: je address}.
DUB_CHECKS = {"french": 0xBE98EB, "italian": 0xBE990F, "german": 0xBE9933, "spanish": 0xBE9957,
              "portuguese": 0xBE997B, "japanese": 0xBE999F, "russian": 0xBE99C3, "polish": 0xBE99E7,
              "sam-spanish": 0xBE9A0B, "brazilian": 0xBE9A2F}


def missing_dubs_patch(folders):
    """Leaves the dubs without a <name>-sp.psarc or <name>-mp.psarc in sound1/streams/speech of any
    of these folders out of the dialogue menu; returns (changes, names left out)."""
    speech = [Path(f) / "build/ps4/main/sound1/streams/speech" for f in folders if f]
    missing = [name for name in DUB_CHECKS
               if not any((d / f"{name}-{kind}.psarc").is_file() for d in speech for kind in ("sp", "mp"))]
    return [(DUB_CHECKS[name], "741B", "EB1B") for name in missing], missing
# The launcher's "System language" (system_language in tlou.ini): the PS4 system language the game
# starts in, by its id.
SYSTEM_LANGUAGES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                    "16", "17", "18", "19", "20", "22", "23", "25"}


def aim_assist_off_patch():
    """The launcher's Controls > Aim assist off (aim_assist=0 in tlou.ini, TLOU_AIM_ASSIST=0).

    12 Nov 2026: the game's aim settings (DC module player-aim-settings, one set per difficulty:
    a list of named settings - default, shotgun, bow, scoped, rifle-scoped, rebar-aim, ... - that
    eboot 1.11 0x4b3b10 searches by name; 0x4b3c80 copies the one found into the player's table at
    0x15e1c60) include Naughty Dog's own "no-aim-assist": the default setting with its four assist
    lists emptied (counts at +0x58, +0x68, +0x78 and +0x88: zones with three curves and a weight
    each, the rest identical). Off, every setting found has those counts set to 0 the same way,
    so each weapon keeps its own speeds, zoom and curves without the assist. The search's match
    (cmp [rax+8], ebx; je found) jumps to three chained caves in the padding after short
    functions (cmp/jne back; xor ecx, then the four counts; then the found path).
    """
    import struct
    rel = lambda after, target: struct.pack("<i", target - after).hex().upper()
    nops = lambda count: "90" * count
    site, loop, found = 0x4B3C20, 0x4B3C25, 0x4B3C2D
    cave1, cave2, cave3 = 0x4B3FE1, 0x4B3FF1, 0x4B4001
    code1 = "395808" + "0F85" + rel(cave1 + 9, loop) + "E9" + rel(cave1 + 14, cave2)
    code2 = "31C9" + "894858" + "894868" + "E9" + rel(cave2 + 13, cave3)
    code3 = "894878" + "898888000000" + "E9" + rel(cave3 + 14, found)
    return [(site, "3958087408", "E9" + rel(site + 5, cave1)),
            (cave1, nops(15), code1 + nops(15 - len(code1) // 2)),
            (cave2, nops(15), code2 + nops(15 - len(code2) // 2)),
            (cave3, nops(15), code3 + nops(15 - len(code3) // 2))]
# The PS4 Pro's shadows (TLOU_PRO_SHADOWS=0 keeps the base PS4's). On a Pro (sceKernelIsNeoMode, the
# engine's IsNeo at 0xB28670) the game sets the render options' (0x181A770) "Neo Enhancements" mode
# (+0x7FC: resolution, framerate, quality), which picks the resolution (+0xF0), checkerboard (+0x7D0)
# and "Toggle On High Detailed Mode" (+0x21D, on in framerate and quality). That flag is read only by
# the frame's shadow setup (0xFD31): the sun's cascades go to 4096x2048 instead of 2560x1280 (render
# target 0x3C instead of 0x3B), and the local lights' shadow memory, which the base PS4 always splits
# into 16 maps, is split by the number of lights casting shadows (targets 0x3D-0x41: 1 map of 4 times
# the side, 2, 4, 8 or 16). The base PS4 creates those targets too. The flag's default (the render
# options' initializer) is on; the rest of the Pro mode stays off (+0x42C, a faster depth clear, too).
PRO_SHADOWS = [(0xAEB313, "C6831D02000000", "C6831D02000001")]
# The ambient shadows (the soft contact shadows of characters, vehicles and props) at half the render
# size instead of a quarter (TLOU_AMBIENT_SHADOWS=quarter keeps the game's). The game computes them
# into render targets of a quarter of its size (the render target setup at 0xAD4210: width >> 2 and
# height >> 2 at 0xAD4DED/0xAD4DFB) and every lit material upsamples them with a depth-aware filter;
# where a foot or a wheel touches the ground both have the same depth, so the ground there takes the
# unshadowed value of the object: a light ring a quarter-size texel or two wide (#5, #66; a PS4 Pro's
# 3200x1800 makes it thinner). Three targets change: 0x18, the quarter-size depth
# (Cs_DownsampleQuarterDepth, dispatched through 0xABEB20 -> 0xABE6F0, which sizes itself from the
# target), created from the shifted registers (its call goes through the padding at 0xABE903, which
# doubles them), and 0x3F (object +0x15A8) and 0x40 (+0x1600), the shadows before and after their
# copy, which load the quarter size from [rbp-0x38C]/[rbp-0x358] and now load the half size the same
# function keeps at [rbp-0x388]/[rbp-0x3B0] (from 0xAD5F9C, for target 0x4E). The ambient shadows'
# constants (0xB26220: width, height and the full-size width over the target's as the fetch scale)
# and dispatches follow the targets' size; the materials' 1080p grid literals, which the game
# rewrites for a quarter of its render size, follow the texture they sample in the runtime
# (shader_recompiler/ir/passes/quarter_grid_pass.cpp). 4x the pixels of that pass; replays on a
# GTX 1050 Ti at 1080p showed no frame rate change (12 Nov 2026), and 2560x1440 starts and plays
# with 1280x720 targets.
AMBIENT_SHADOWS_HALF = [
    (0xAD64A8, "448B8574FCFFFF", "448B8578FCFFFF"),   # target 0x3F: mov r8d, [rbp-0x388]
    (0xAD64AF, "448B8DA8FCFFFF", "448B8D50FCFFFF"),   #              mov r9d, [rbp-0x3B0]
    (0xAD8A02, "448B8574FCFFFF", "448B8578FCFFFF"),   # target 0x40
    (0xAD8A09, "448B8DA8FCFFFF", "448B8D50FCFFFF"),
    (0xAD4E06, "E89581FEFF", "E8F89AFEFF"),           # target 0x18: call 0xABE903
    # add r8d, r8d; add r9d, r9d; jmp 0xABCFA0 (the creation call), in the padding after 0xABE8FE
    (0xABE903, "90" * 13, "4501C0" "4501C9" "E992E6FFFF" "9090"),
]
# The render resolution (the launcher's Resolution, output_res in tlou.ini; TLOU_RESOLUTION=<w>x<h>
# overrides it). The game draws its 3D scene at the size the render options' +0xF0 picks from the
# engine's table of render sizes (0xAB0600, the debug menu's "Screen Resolution"): 0 = 3840x2160,
# 1-8 = the widths at 0x116BFA0 and heights at 0x116BFC0, 3680x2070 ... 1920x1080. The options'
# initializer (0xAEB00F) sets 8 and only a PS4 Pro changes it, so the last entry, 1920x1080, is
# replaced by the chosen size.
# Above 1080p the render targets no longer fit the base PS4's memory map (the "Render Target Heap"
# assertion at boot). A PS4 Pro's map (0x151A4F0) gives ALLOCATION_GPU_MEMORY 600 MiB more for its
# 3200x1800 checkerboard: 130 instead of 42 MiB and 1171 instead of 659 MiB for the two
# ALLOCATION_RENDER_TARGET heaps. Here the base map (0x14A58C0, in the RELRO segment) grows the GPU
# memory and those heaps the same way, by RENDER_TARGET_EXTRA_MIB scaled with the pixel count; the
# direct memory total of the base config (0x2F5E44) and the size the game expects
# sceKernelGetDirectMemorySize to report (0x10B36; another size means a development kit) grow with
# it. The GPU memory is reserved in a range aligned to 4 GiB, so it stays below 4096 MiB.
RENDER_DEFAULT = (1920, 1080)
GPU_MEMORY_MIB = 2588
RENDER_TARGET_EXTRA_MIB = 500     # per 1080p frame of extra pixels (1440p: 448 MiB, 4K: 1500 MiB)


def parse_size(text):
    match = re.fullmatch(r"\s*(\d{3,4})\s*x\s*(\d{3,4})\s*", text or "")
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if not (640 <= width <= 7680 and 360 <= height <= 4320):
        return None
    # The game asserts that its display width is a multiple of 16 (1366x768 draws at 1360x768).
    return width & ~15, height & ~1


# Temporal anti-aliasing and upscaling (the launcher's Upscaler and Upscaler quality: upscaler and
# preset in tlou.ini; the in-game menu changes the upscaler live, the quality needs a restart). The
# renderer's upscaler (TemporalUpscaler, from the GOW3/Uncharted ports) needs this game's frame:
#   - the camera: the motion blur's camera velocity pass, ps dbdfaf50 (960x540 RG16F). Its constants
#     give z = c9 / (depth - c8), view = z (c0 u + c2, c1 v + c3) and rows c12..c23 take the view
#     position straight to the previous clip x, y, w (GOW3_VELOCITY_LAYOUT=view_clip);
#   - the scene depth: the frame's first pass at the render size (the depth prepass, D32S8);
#   - at the native size the anti-aliasing runs on the lit HDR scene color (RGBA16F) before the post
#     processing reads it (cs 8bfb95d1, the first one that does, after the velocity pass);
#   - the frame ends in the tonemap compute shader 15d34dcb, which writes the finished frame
#     straight into the display buffer (A2B10G10R10); the HUD is drawn over it there. No draw
#     copies the frame to the display buffer as in the other games, so frames start there
#     (GOW3_FRAME_CS). Below the output size the game draws at the reduced size (the render size
#     patch) and that frame is upscaled there into an output-size display buffer, where the HUD
#     is then drawn at the output size;
#   - the sun shadow mask (ps a1f94b7c) rebuilds positions from the scene depth: it gets the
#     scene's jitter (GOW3_JITTER_PS), or its shadows would crawl with every jitter step.
# TLOU_UPSCALER_HOOKS=0 leaves all of it out (the upscaler options then do nothing).
UPSCALER_SCALES = (1.0, 1.5, 1.7, 2.0, 3.0)     # Native AA, Quality, Balanced, Performance, Ultra
UPSCALERS = ("fsr3", "fsr4", "fsr411", "taa", "dlss", "xess")


def upscaled_render_size(output, preset):
    """The reduced size the game draws at for an upscaler quality preset (None at native)."""
    if not 0 < preset < len(UPSCALER_SCALES):
        return None
    scale = UPSCALER_SCALES[preset]
    height = max(360, int(round(output[1] / scale / 2)) * 2)
    width = max(640, int(round(height * output[0] / output[1] / 16)) * 16)
    if width >= output[0] or height >= output[1]:
        return None
    return width, height


def render_extra_mib(size):
    """GPU memory added for render targets at this size (0 up to 1080p's pixel count)."""
    pixels = size[0] * size[1] / (RENDER_DEFAULT[0] * RENDER_DEFAULT[1])
    # The memory map's sizes must be whole multiples of 2 MiB (an assertion at boot): 64 MiB steps.
    extra = -(-int(max(0.0, pixels - 1.0) * RENDER_TARGET_EXTRA_MIB) // 64) * 64
    return min(extra, 4096 - GPU_MEMORY_MIB - 8)


def ini_value(path, key):
    """A key of the settings file (tlou.ini), or None."""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key:
                return value.strip()
    except OSError:
        pass
    return None


def render_size_patch(size, extra_mib):
    q = lambda value: value.to_bytes(8, "little").hex().upper()
    d = lambda value: value.to_bytes(4, "little").hex().upper()
    changes = [(0x116BFBC, d(RENDER_DEFAULT[0]), d(size[0])),
               (0x116BFDC, d(RENDER_DEFAULT[1]), d(size[1]))]
    if extra_mib:
        extra = extra_mib << 20
        small = min(extra, 88 << 20)   # the first render target heap: 42 -> at most the Pro's 130 MiB
        total = 0x120000000 + extra
        changes += [(0x2F5E44, "48BA" + q(0x120000000), "48BA" + q(total)),
                    (0x10B36, "49BC" + q(0x120000000), "49BC" + q(total)),
                    (0x14A5908, q(0xA1C00000), q(0xA1C00000 + extra)),   # ALLOCATION_GPU_MEMORY
                    (0x14A6708, q(0x2A1C000), q(0x2A1C000 + small)),     # ALLOCATION_RENDER_TARGET
                    (0x14A6728, q(0x292B3C00), q(0x292B3C00 + extra - small))]
    return changes


# Another aspect ratio than 16:9 (ultrawide monitors, 21:9 and 32:9; also 16:10). The game assumes
# 16:9 in its camera and its 1280x720 interface canvas; at 2560x1080 alone it stretches both. These
# changes keep the vertical field of view and show more at the sides (Hor+), keep the interface 16:9
# and centered, and its markers on the right places. The sites and values follow
# github.com/SanyaGames227/tlou_r_shadps4_ultrawide (an XML patch generator for shadPS4; its
# addresses are these plus 0x400000); each one was checked against the 01.11 executable. R is the
# width's ratio to 16:9 (2560x1080: 1.333). The movies' black bars are the runtime's
# (GOW3_DISPLAY_ASPECT, gpu/shadps4/video_core/renderer_vulkan/vk_movie_aspect.h).
ASPECT_16_9 = 16 / 9


def is_16_9(size):
    return abs(size[0] / size[1] / ASPECT_16_9 - 1) < 0.003


def aspect_patch(size):
    import math
    import struct
    f = lambda value: struct.pack("<f", value).hex().upper()
    rel = lambda after, target: struct.pack("<i", target - after).hex().upper()
    nops = lambda count: "90" * count
    r = size[0] / size[1] / ASPECT_16_9
    changes = [
        # The render options' aspect ratio (+0x18, 16:9) in both of their initializers.
        (0xAB0594, f(ASPECT_16_9), f(size[0] / size[1])),
        (0xAB091B, f(ASPECT_16_9), f(size[0] / size[1])),
        # The camera's projection: cot(fov/2) / R instead of 1 / tan(fov/2) (the vertical field of
        # view stays). The 1.0 it divides (0x115DF64, shared) is read from 1/R in padding instead.
        (0xA223C7, nops(4), f(1 / r)),
        (0xA252AC, rel(0xA252B0, 0x115DF64), rel(0xA252B0, 0xA223C7)),
    ]
    # The camera's clipping bounds (the three cameras' half width, x R): each multiply moves into a
    # call to a stub in the padding after the render options' initializer, which multiplies by R.
    times_r = 0xAB0673
    changes.append((times_r, nops(13), "C5FA590501000000C3" + f(r)))    # vmulss xmm0,[R]; ret; R
    for site, original, stub, room in [(0x29234C, "C5FA5985E0FDFFFF", 0xAB05F1, 13),   # [rbp-0x220]
                                       (0x29374B, "C5FA5945F0", 0xAB0663, 10),          # [rbp-0x10]
                                       (0x2965A4, "C5FA5985A0FEFFFF", 0xAB0691, 13)]:   # [rbp-0x160]
        code = original + "E9" + rel(stub + len(original) // 2 + 5, times_r)
        changes.append((stub, nops(room), code))
        call = "E8" + rel(site + 5, stub)
        changes.append((site, original, call + "0F1F00"[:len(original) - 10]))
    # The interface: its 1280-wide canvas maps to the middle 1/R of the screen (x scale 2/1280R,
    # offset -1/R), also in the debug menus; text's canvas is 1280R wide, and the crosshair divides
    # by it (instead of its own 1280.0 at 0x10FD238); the 3D canvas projection is x R.
    ui_scale, ui_offset = f(2 / (1280 * r)), f(-1 / r)
    for at in (0x1172444, 0x116CED0, 0x116CEE0, 0x116CF20):
        changes += [(at, f(2 / 1280), ui_scale), (at + 8, f(-1.0), ui_offset)]
    changes += [(0x1186BB8, f(1280.0), f(1280 * r)), (0x1186BC4, f(1280.0), f(1280 * r)),
                (0x6250DC, rel(0x6250E0, 0x10FD238), rel(0x6250E0, 0x1186BB8)),
                (0x1186F80, f(1.0), f(r))]
    # The interaction and pickup icons: world -> canvas x = (x R + 1) * 640 instead of (x + 1) *
    # 640, through a jump to padding (vaddss xmm5,xmm6 and vmulss xmm5,[640.0] move there).
    site, back, cave = 0xAAD8C1, 0xAAD8CD, 0xC9FC61
    changes.append((site, "C5D258EEC5D2592D" + rel(back, 0x116BC3C), "E9" + rel(site + 5, cave) + nops(7)))
    changes.append((cave, nops(29), "C5D2592D" + rel(cave + 8, cave + 25) + "C5D258EE"
                    + "C5D2592D" + rel(cave + 20, 0x116BC3C) + "E9" + rel(cave + 25, back) + f(r)))
    # The throwables' target circle (GitHub #143): the aim reticle's update (0x2992xx) keeps its
    # canvas position at +0x1c34, springs it towards its world point projected by the icon
    # function above (x R), then turns it back into a world point with 0xaadb50 (canvas -> ray,
    # 2x/1280 - 1, no R), so each frame pushed the circle R times further from the middle until
    # it stuck at a screen edge. Its one call goes through a stub that first maps x to
    # 640 + (x - 640) / R (x / R + 640 (1 - 1/R)); the two constants sit in the padding after the
    # icon function.
    stub, constants, call = 0xCFA5E8, 0xAAD8E6, 0x299467
    changes.append((constants, nops(8), f(1 / r) + f(640 * (1 - 1 / r))))
    changes.append((stub, nops(21), "C5FA5905" + rel(stub + 8, constants) + "C5FA5805" + rel(stub + 16, constants + 4)
                    + "E9" + rel(stub + 21, 0xAADB50)))
    changes.append((call, "E8" + rel(call + 5, 0xAADB50), "E8" + rel(call + 5, stub)))
    # The collectibles' preview camera (pickup notice, artifacts in the backpack): its 60 degree
    # horizontal field of view widens to keep the 16:9 vertical one.
    fov = 2 * math.atan(r * math.tan(1.0470927953720093 / 2))
    changes += [(0x10C9890, "2307863F", f(fov)), (0x10C9928, "2307863F", f(fov))]
    return changes


# Fixes, on in every run.
FIXES = {
    # 23 Oct 2026 (the house, after the autosave before the basement): the renderer's frame
    # watchdog (ndgi-orbis-device.cpp WaitOnContextDone) halted the game when a frame took over 2 s
    # on the GPU. There the GPU command thread created ~150 new pipelines inside one frame (stack: a
    # DRAW_INDEX_OFFSET_2 opening a pipeline cache file on a hard disk). The compare (vucomiss
    # xmm0, [2.0f]) and its message (vmovsd xmm0, [2.0]) read a 20.0 of the read-only image
    # instead: a compile burst is a pause, a real GPU hang still stops the game.
    "frame-timeout": [(0xB3D887, "C5F82E05819F6300", "C5F82E05194D5C00"),
                      (0xB3D92F, "C5FB1005F19E6300", "C5FB1005A13D5900"),
                      (0x11025A8, "0000A041", None), (0x10D16D8, "0000000000003440", None)],
}


def image_patches(elf_path, out_path, patches, applied=None):
    """Writes the chosen changes ({name: changes}, from CHEATS and FIXES) as a G3PATCH2 file for
    the runtime's --patches (src/probe.c apply_patches) and returns its path, or None. A change
    whose original bytes are not in this executable (another version) is left out with a message.
    The names of the changes written are added to `applied`."""
    import struct
    elf = elf_path.read_bytes()
    phoff, = struct.unpack_from("<Q", elf, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x36)
    segments = []
    for i in range(phnum):
        kind, _, offset, vaddr, _, filesz, _, _ = struct.unpack_from("<IIQQQQQQ", elf, phoff + i * phentsize)
        if kind in (1, 0x61000010):  # PT_LOAD and PT_SCE_RELRO (the memory map table)
            segments.append((vaddr, offset, filesz))

    def read(vaddr, size):
        for start, offset, filesz in segments:
            if start <= vaddr and vaddr + size <= start + filesz:
                return elf[offset + vaddr - start:offset + vaddr - start + size]
        return b""

    entries = []
    for name, changes in patches.items():
        differ = [at for at, original, _ in changes if read(at, len(original) // 2) != bytes.fromhex(original)]
        if differ:
            print(f"Patches: '{name}' left out: this executable is not version 01.11's at {differ[0]:#x}")
            continue
        entries += [(at, bytes.fromhex(replacement)) for at, _, replacement in changes if replacement]
        if applied is not None:
            applied.add(name)
        print(f"Patches: {name} on")
    if not entries:
        return None
    data = b"G3PATCH2" + struct.pack("<QQ", 0, len(entries))
    for at, replacement in entries:
        data += struct.pack("<QQ", at, len(replacement)) + replacement
    out_path.write_bytes(data)
    return out_path


def game_executable():
    built = ROOT / "gow3_pc/out/gow3-probe.exe"
    if not built.is_file():
        raise SystemExit(f"Build the runtime first (gow3_pc/build.sh): {built} is missing")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=ROOT.parent / "CUSA00552")
    parser.add_argument("--boot", action="store_true", help="Execute the game after preparation")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--seconds", type=int, default=60, help="Process time limit; 0 disables it")
    parser.add_argument("--capture", type=int, default=0, metavar="SECONDS",
                        help="Save the displayed frame every SECONDS (analysis/tlou/captures)")
    parser.add_argument("--saves", type=Path, default=None,
                        help="Saves and game settings folder (default analysis/tlou-user, the "
                             "player's): a test profile leaves the player's saves alone")
    args = parser.parse_args()
    game = args.game.resolve()
    update = find_update(game)
    if update:
        print(f"Game {game} (version {app_version(game)}) with its update {update} "
              f"(version {app_version(update)}): the update's executable and files are used", flush=True)
    else:
        print(f"Game {game} (version {app_version(game)}); no update found next to it "
              f"({game.name}-patch or {game.name}-update, or TLOU_UPDATE=<folder>)", flush=True)
        if merged_with_base_executable(game):
            print("WARNING: update 1.11's files were merged into this folder, but its eboot.bin is still "
                  "version 1.00's, which the port does not run (it stops during start). "
                  "Copy the update's eboot.bin, sce_module and sce_sys over the game's (replace the "
                  f"files), or keep the update in its own folder next to the game ({game.name}-patch).",
                  flush=True)
    out = ROOT / "analysis/tlou"
    user = ROOT / "analysis/tlou-user"
    user.mkdir(parents=True, exist_ok=True)
    # Diagnostic switches (tlou_pad.py) last for one run: a new run starts with the defaults.
    # A file that cannot be removed (Windows answered "access denied" for probe-pad.txt left by a
    # replay test) must not stop the game: it is emptied instead, which turns the switch off for
    # the text files, and the run goes on with a warning.
    for name in ("disabled-optimizations.txt", "skip.txt", "readback.trigger", "capture.trigger",
                 "probe-pad.txt"):
        try:
            (user / name).unlink(missing_ok=True)
        except OSError as error:
            try:
                (user / name).write_bytes(b"")
                print(f"WARNING: could not remove {user / name} ({error}); emptied it instead",
                      flush=True)
            except OSError:
                print(f"WARNING: could not remove or empty {user / name} ({error}); "
                      "the diagnostic switch it holds stays on for this run", flush=True)
    # The diagnostic files and the shader cache stay in analysis/tlou-user either way.
    saves = args.saves.resolve() if args.saves else user
    saves.mkdir(parents=True, exist_ok=True)
    view = boot_view(game, update)
    scripts = ROOT / "gow3_pc/scripts"
    for script in ("prepare.py", "link_modules.py", "content_profile.py"):
        subprocess.run([sys.executable, str(scripts / script), str(view), "--out", str(out)], check=True)
    # CPUs older than Haswell (Sandy / Ivy Bridge) lack instructions the PS4 code uses (BMI1, LZCNT,
    # MOVBE, F16C): the red-zone table also replaces those by emulation code (gow3_pc/scripts/
    # isa_emulation.py; from vitolasanha/tlou-remastered-pc-sandy-bridge-fix).
    # TLOU_ISA_REWRITE=auto (default: only what this CPU lacks) | force (all, for tests) | off.
    sys.path.insert(0, str(scripts))
    import cpu_features
    isa_mode = os.environ.get("TLOU_ISA_REWRITE", "auto").lower()
    if isa_mode == "off":
        isa_names = []
    elif isa_mode == "force":
        isa_names = list(cpu_features.FEATURE_KINDS)
    else:
        isa_names = cpu_features.missing_emulatable() or []
    # Windows red-zone protection (gow3_pc/scripts/red_zone.py): GOW3_RED_ZONE_REWRITE=0 turns it off.
    red_zone = None
    want_red_zone = os.environ.get("GOW3_RED_ZONE_REWRITE") != "0"
    if want_red_zone or isa_names:
        command = [sys.executable, str(scripts / "red_zone_table.py"), str(out / "boot-linked.bin"), "--out", str(out)]
        if not want_red_zone:
            command.append("--isa-only")
        table = subprocess.run(command)
        if table.returncode == 0:
            red_zone = out / "red-zone.bin"
        elif isa_names:
            sys.exit("This CPU lacks " + ", ".join(isa_names) + ", which the game's code uses, and the table that "
                     "replaces those instructions could not be made (see above).")
    if isa_names:
        print("ISA emulation active for: " + ", ".join(isa_names))

    env = dict(os.environ)
    if isa_names and not cpu_features.missing_blocking():
        env["GOW3_CPU_CHECK"] = "0"           # the runtime would stop on the sets that are emulated now
    msys = Path(os.environ.get("MSYS2_ROOT", "C:/msys64"))
    env["PATH"] = str(msys / "clang64/bin") + os.pathsep + env.get("PATH", "")
    env["GOW3_GPU_USER_DIR"] = str(user)
    # The shader cache: analysis/tlou-user/cache or, when that is a link (a junction to an SSD,
    # README), the folder it points to.
    cache = user / "cache"
    if cache.exists():
        env.setdefault("GOW3_GPU_CACHE_DIR", str(cache.resolve()))
    env.setdefault("GOW3_CONFIG", str(user / "tlou.ini"))
    # A draw whose pipeline is still compiling in the background is skipped: the curtain on the
    # title screen and the characters' hair flashed white the first time they were drawn, and the
    # stale data skipped draws leave crashed the game (two device losses in the Quarantine Zone,
    # a game assertion; 22 Oct 2026). New pipelines are compiled at once (a short stall the first
    # time; the pipeline cache and the prefetch below keep it short), whatever the settings file
    # says, and the in-game menu does not offer the switch.
    env.setdefault("GOW3_ASYNC_SHADERS", "0")
    env.setdefault("GOW3_ASYNC_SHADERS_LOCKED", "1")
    # The game compares sceKernelGetDirectMemorySize with a retail console's 4608 MiB; any other
    # size makes it reserve a development kit's 768 MiB debug area.
    env.setdefault("GOW3_DMEM_REPORT_MB", "4608")
    env.setdefault("GOW3_DIRECT_MEMORY_ACCESS", "1")
    env.setdefault("GOW3_PIPELINE_CACHE", "1")
    env.setdefault("GOW3_PIPELINE_BINARIES", "1")
    # Frame rate (Sarah's bedroom, GTX 1050 Ti Mobile: 20.6 FPS with none of these, 47 with all
    # and the priority below; 48-53 walking upstairs): multithreaded rendering (draw and Vulkan
    # recording threads), 28 FPS on its own with the DMA windows; the DMA shaders' guest
    # memory synchronized in 256 KiB windows at the addresses in their user data instead of every
    # CPU-dirty range; and one frame in flight, the game building the next frame while the GPU
    # command thread still decodes the last one, instead of both waiting for the other at each
    # sceGnmSubmitDone (liverpool.h, FramesInFlight).
    # The two-stage draw pipe (draw recording thread) was off from 9 to 10 Nov 2026, blamed for a
    # lost device in Pittsburgh. The cause was the cloth solver Cs_ClothSimulate512 on NVIDIA
    # cards before Turing (SkipPascalCloth512 in vk_rasterizer.cpp); the pipe only made the loss
    # more frequent, and 9 replays of that route with it and the solver skipped kept the device.
    # The launcher's Startup tab turns it off.
    env.setdefault("GOW3_DRAW_PIPE", "1")
    env.setdefault("GOW3_VK_RECORD_THREAD", "1")
    env.setdefault("GOW3_DMA_SYNC", "pointers 256")
    env.setdefault("GOW3_FRAMES_IN_FLIGHT", "1")
    # At normal priority the GPU command thread and the draw and Vulkan recording threads shared
    # cores with the game's six busy job workers in some sessions, for the whole session: 33-34
    # instead of 46-47 FPS in 5 of 11 runs. Above normal: 47 FPS in 4 runs of 4.
    env.setdefault("GOW3_GPU_THREAD_PRIORITY", "above")
    # A game thread's write into memory the GPU is still writing waits for that GPU work itself,
    # not on the GPU command thread (which stopped drawing meanwhile): 32-36 -> 41-47 FPS in the
    # bedroom (4 alternating runs, 22 Oct 2026). Sarah's hair shakes with and without it.
    env.setdefault("GOW3_FETCH_NO_BLOCK", "1")
    # New pipelines the draw preparation workers see ahead are built on the compiler threads, in
    # parallel, before the GPU command thread reaches their draws (an empty cache: 573 -> 167 ms
    # of compiling on that thread). The first visit of an area hitches less.
    env.setdefault("GOW3_PREFETCH_PIPELINES", "1")
    # Memory waits met by a fence still queued for the draw recording thread, nested compute
    # buffers and compute memory waits do not drain the draw pipe (in the heavy part of the house
    # the GPU command thread waited for that thread 47% of its time). The player's choice (22 Oct
    # 2026) after playing it: it felt smoother there; the measured runs saw different parts of
    # the scene, so no clean number.
    env.setdefault("GOW3_FEWER_DRAINS", "1")
    # The 480 Hz vblank (frames shown when they are done, the frame rate capped at the display's):
    # at a 60 Hz vblank the game waits for the next one, so frames of the heavy areas took
    # 16.7/33.3/50 ms in turn (~30 FPS). The player's session with it (24 Oct 2026): 30-34 FPS
    # there, normal game speed. (Joel's hair still stretches with it at times: not the vblank.)
    env.setdefault("GOW3_VBLANK_HZ", "0")
    # A game thread's write beside GPU bytes still being written (the copy shader 82c46b8b's
    # constant blocks in the per-frame rings) goes ahead instead of waiting ~37 ms for the frame
    # in flight; those GPU bytes are poisoned in guest memory until their data lands. Pages of
    # the physics shaders (results the game reads) still wait. The player's sessions in the
    # Quarantine Zone (24 Oct 2026): frame time stddev ~15 -> ~10 ms, slow frames 41-72 -> 5-19
    # per 5 s, same FPS; GOW3_SHARE_CHECK=1 found no stale or poisoned bytes.
    env.setdefault("GOW3_SHARE_UNFINISHED_WRITES", "1")
    # The small GPU-side DMA copies into guest memory (counters the game reads) come back with
    # the next end-of-pipe fence, held until they land, so writes beside them need not wait
    # either. Shared without it, the NPCs walked oddly. The player's sessions (25 Oct 2026):
    # NPCs normal in 2 of 3 (rising in the first, with no difference found in its log), stddev
    # ~11 -> ~8 ms, slow frames ~12 -> ~2 per 5 s. GOW3_DMA_EOP_READBACK=0: those pages wait.
    env.setdefault("GOW3_DMA_EOP_READBACK", "1")
    env.setdefault("GOW3_NATIVE_TSC", "1")
    env.setdefault("GOW3_FRAME_STATS", "1")
    # Scripted controller input (tools/tlou_pad.py writes it).
    env.setdefault("GOW3_PAD_FILE", str(user / "probe-pad.txt"))
    # 24 Oct 2026: the player's sessions record their controller input from the first press
    # (GOW3_PAD_RECORD, flushed as it goes), one file per session, the newest 5 kept: a device loss
    # came minutes into a level only on some routes, and a recording replays the route in a test
    # profile (GOW3_PAD_REPLAY). Test runs (--saves) record nothing; TLOU_RECORD_PAD=0 turns it off.
    if not args.saves and os.environ.get("TLOU_RECORD_PAD") != "0" and "GOW3_PAD_RECORD" not in env:
        records = sorted(out.glob("pad-record-*.txt"))
        for old in records[:-4]:
            old.unlink(missing_ok=True)
        env["GOW3_PAD_RECORD"] = str(out / f"pad-record-{time.strftime('%Y%m%d-%H%M%S')}.txt")
        (user / "probe-pad.txt").write_text("record", encoding="ascii")
    # Diagnostics driven by tlou_pad.py: the frame analyzer (every pass, its shaders and textures,
    # written to <user>/captures) and shaders whose draws are skipped (hex hashes, re-read live).
    env.setdefault("GOW3_CAPTURE_TRIGGER", str(user / "capture.trigger"))
    # The game never draws into its display buffers, so the analyzer sees no frame boundary: it
    # records this many passes (a number in the trigger file overrides it).
    env.setdefault("GOW3_CAPTURE_PASSES", "400")
    env.setdefault("GOW3_SKIP_FILE", str(user / "skip.txt"))
    # "<lo> <hi>" (hex) in this file copies that GPU-written range back to guest memory once.
    env.setdefault("GOW3_READBACK_TRIGGER", str(user / "readback.trigger"))
    # Tiny GPU-written images (one row, at most 4 KiB) return to guest memory with the next fence:
    # the aim focus distance may be read from a 1x1 target; never copied back, it stays 0.
    env.setdefault("GOW3_TINY_IMAGE_READBACK", "1")
    # The GPU command thread is the limit in heavy areas (0% idle, ~29 FPS). Submissions reuse their
    # buffers instead of freeing them (~16% of that thread) and small per-draw savings: 39.9 -> 43.4 FPS.
    env.setdefault("GOW3_PREP_POOL", "1")
    env.setdefault("GOW3_CPU_TRIMS", "1")
    # The game's motion blur off (GOW3_NO_MOTION_BLUR=0 keeps it): its shaders see no movement.
    env.setdefault("GOW3_NO_MOTION_BLUR", "1")
    # Optimizations switched off while the game runs (a bit mask, src/runtime_memory.c).
    env.setdefault("GOW3_TOGGLE_FILE", str(user / "disabled-optimizations.txt"))
    if os.environ.get("TLOU_GPU_CHECKPOINTS") == "1":
        env.setdefault("GOW3_GPU_CHECKPOINTS", "1")

    command = [str(game_executable()), str(out / "boot-linked.bin"),
               "--no-game-hooks", "--low-image", "--content-profile", str(out / "content.bin"),
               "--app0", str(game), "--user", str(saves)]
    if update:
        command += ["--app0-overlay", str(update)]
    command += ["--guest-path", "/app0/eboot.bin"]
    # TLOU_CHEATS=<name>,...: the cheats above (default: flashlight, unless flashlight_fix=0 in
    # tlou.ini; "none": no cheats).
    # TLOU_FIXES=none leaves out the fixes.
    patches = dict(FIXES) if os.environ.get("TLOU_FIXES") != "none" else {}
    if os.environ.get("TLOU_JOB_SLEEP") != "0":
        patches["job-sleep"] = JOB_SLEEP
    if os.environ.get("TLOU_PRO_SHADOWS") != "0":
        patches["pro-shadows"] = PRO_SHADOWS
    if os.environ.get("TLOU_AMBIENT_SHADOWS") != "quarter":
        patches["ambient-shadows-half"] = AMBIENT_SHADOWS_HALF
    if os.environ.get("TLOU_ALL_LANGUAGES") != "0":
        patches["all-languages"] = ALL_LANGUAGES
        dubs, missing = missing_dubs_patch([game, update])
        if dubs:
            patches["dubs without files hidden"] = dubs
            print(f"Languages: dubs not in the game folder, left out of the dialogue menu: {', '.join(missing)}")
    # GOW3_LANGUAGE set outside wins over the launcher's choice.
    system_language = (ini_value(Path(env["GOW3_CONFIG"]), "system_language") or "").strip()
    if system_language in SYSTEM_LANGUAGES:
        env.setdefault("GOW3_LANGUAGE", system_language)
    elif system_language:
        print(f"Language: system_language '{system_language}' is not a PS4 system language id")
    # The launcher's Controls > Aim assist (aim_assist in tlou.ini; TLOU_AIM_ASSIST=0/1 wins).
    aim_assist = os.environ.get("TLOU_AIM_ASSIST") or ini_value(Path(env["GOW3_CONFIG"]), "aim_assist")
    if (aim_assist or "1").strip() == "0":
        patches["aim-assist-off"] = aim_assist_off_patch()
    # The launcher's Controls > DirectInput controller (dinput, dinput_buttons, dinput_axes in
    # tlou.ini; src/runtime_pad.c): GOW3_DINPUT* set outside win.
    for key, name in (("dinput", "GOW3_DINPUT"), ("dinput_buttons", "GOW3_DINPUT_BUTTONS"),
                      ("dinput_axes", "GOW3_DINPUT_AXES")):
        value = (ini_value(Path(env["GOW3_CONFIG"]), key) or "").strip()
        if value:
            env.setdefault(name, value)
    chosen = os.environ.get("TLOU_RESOLUTION") or ini_value(Path(env["GOW3_CONFIG"]), "output_res")
    size = parse_size(chosen) or RENDER_DEFAULT
    if chosen and not parse_size(chosen):
        print(f"Resolution: '{chosen}' is not <width>x<height>; the game's own 1920x1080")
    output = size
    upscaler_hooks = os.environ.get("TLOU_UPSCALER_HOOKS", "1") != "0"
    upscaler = (os.environ.get("GOW3_UPSCALER") or ini_value(Path(env["GOW3_CONFIG"]), "upscaler")
                or "off").strip()
    try:
        preset = int((os.environ.get("TLOU_UPSCALER_PRESET")
                      or ini_value(Path(env["GOW3_CONFIG"]), "preset") or "0").strip())
    except ValueError:
        preset = 0
    upscaled = upscaled_render_size(output, preset) if upscaler_hooks and upscaler in UPSCALERS else None
    if upscaled:
        size = upscaled
    resolution_patch = f"resolution {size[0]}x{size[1]}"
    extra = render_extra_mib(size)
    if size != RENDER_DEFAULT:
        patches[resolution_patch] = render_size_patch(size, extra)
    # TLOU_ASPECT=0 leaves another aspect ratio than 16:9 stretched. The output size's aspect: a
    # reduced render size is rounded to whole blocks of 16 pixels.
    aspect_name = f"aspect ratio {output[0]}:{output[1]}"
    if not is_16_9(output) and os.environ.get("TLOU_ASPECT") != "0":
        patches[aspect_name] = aspect_patch(output)
    # The launcher's Controls > Flashlight (flashlight_fix in tlou.ini, default on) chooses the
    # flashlight cheat; TLOU_CHEATS set outside wins.
    cheats = os.environ.get("TLOU_CHEATS")
    if cheats is None:
        flashlight = (ini_value(Path(env["GOW3_CONFIG"]), "flashlight_fix") or "1").strip()
        cheats = "none" if flashlight == "0" else "flashlight"
    for name in [c.strip() for c in cheats.split(",") if c.strip()]:
        if name in CHEATS:
            patches[name] = CHEATS[name]
        elif name != "none":
            print(f"Cheats: unknown '{name}' (known: {', '.join(CHEATS)})")
    applied = set()
    if patches:
        patch_file = image_patches(out / "eboot.elf", out / "patches.bin", patches, applied)
        if patch_file:
            command += ["--patches", str(patch_file)]
    if resolution_patch not in applied:
        size, extra = RENDER_DEFAULT, 0
        if upscaled:
            print("Upscaling: the render size patch did not apply; the game draws at 1920x1080")
            upscaled, output = None, RENDER_DEFAULT
    elif extra:
        env["GOW3_DMEM_REPORT_MB"] = str(4608 + extra)
        pool = int(env.get("GOW3_DMEM_MB", "0") or 0)
        env["GOW3_DMEM_MB"] = str(max(pool, 4608 + extra + 448))
    # The host's renderer follows the game's own size (no host scaling of its render targets), and
    # the in-game menu shows it.
    env["GOW3_GAME_RESOLUTION"] = f"{size[0]}x{size[1]}"
    # The movies (the cutscenes) keep 16:9 in a display of this aspect ratio, with black bars.
    if aspect_name in applied:
        env["GOW3_DISPLAY_ASPECT"] = f"{output[0]}x{output[1]}"
    if upscaler_hooks:
        env.setdefault("GOW3_VELOCITY_PS", "dbdfaf50")
        env.setdefault("GOW3_VELOCITY_LAYOUT", "view_clip")
        env.setdefault("GOW3_UPSCALE_BEFORE_CS", "8bfb95d1")
        env.setdefault("GOW3_FRAME_CS", "15d34dcb")
        env.setdefault("GOW3_JITTER_PS", "a1f94b7c")
        # The characters' own motion comes from the other games' vertex shaders: camera motion only.
        env.setdefault("GOW3_CHARACTER_MOTION", "0")
        env["GOW3_SCENE_SIZE"] = f"{size[0]}x{size[1]}"
        fsr4 = ROOT / "gow3_pc/fsr4_shaders"
        if fsr4.is_dir():
            env.setdefault("GOW3_FSR4_DIR", str(fsr4))
        if upscaled:
            env["GOW3_RENDER_RES"] = f"{size[0]}x{size[1]}"
            env["GOW3_OUTPUT_RES"] = f"{output[0]}x{output[1]}"
            print(f"Upscaling: {size[0]}x{size[1]} -> {output[0]}x{output[1]} ({upscaler}, quality "
                  f"preset {preset})", flush=True)
    else:
        env["GOW3_UPSCALER"] = "none"
    print(f"Resolution: the game draws at {size[0]}x{size[1]}"
          + (f" ({extra} MiB more for its render targets)" if extra else ""), flush=True)
    if red_zone:
        command += ["--red-zone", str(red_zone)]
    if not args.boot:
        command.append("--check-imports")
    if args.cpu_only:
        command.append("--cpu-only")
    name = ("boot" if args.boot else "check-imports") + ("-cpu" if args.cpu_only else "")
    result = {"command": command, "update": str(update) if update else None, "timeout_seconds": args.seconds}
    capture = None
    if args.capture:
        # The presenter writes the displayed frame each time the trigger file appears.
        capture = out / "captures" / str(time.time_ns())
        capture.mkdir(parents=True)
        env["GOW3_FRAME_DUMP_TRIGGER"] = str(capture / "trigger")
        env["GOW3_DUMP_DIR"] = str(capture)
        result["capture_directory"] = str(capture)
    start = time.monotonic()
    with (out / (name + ".log")).open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        # Test scripts stop this instance by its PID, never a game the player started.
        (out / "probe.pid").write_text(str(process.pid), encoding="ascii")
        deadline = start + args.seconds if args.seconds else None
        result["timed_out"] = False
        while process.poll() is None:
            if deadline and time.monotonic() >= deadline:
                process.kill()
                result["timed_out"] = True
                break
            try:
                process.wait(timeout=args.capture or 1)
            except subprocess.TimeoutExpired:
                if capture:
                    (capture / "trigger").touch()
        result["exit_code"] = process.wait()
    if capture:
        convert_frames(capture)
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    (out / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    attempt = out / "attempts" / f"{name}-{time.time_ns()}"
    attempt.mkdir(parents=True)
    shutil.copy2(out / (name + ".log"), attempt / (name + ".log"))
    print(json.dumps(result, indent=2))
    print(f"Log: {out / (name + '.log')}")
    return 124 if result["timed_out"] else result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
