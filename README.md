# The Last of Us Remastered — PC port

Runs the PS4 **The Last of Us Remastered** (CUSA00552, with its update 1.11) natively on Windows.
The game's own x86-64 code runs directly on the PC; a runtime supplies the PS4 system libraries,
and a Vulkan renderer (the shadPS4 video core) executes the PS4 GPU work.

Discord Server : https://discord.gg/9mnjjezcNj

> **No game files are in this repository.** You need your own copy of the game (the dumped
> CUSA00552 folder, and its update). This project is not affiliated with Sony Interactive
> Entertainment or Naughty Dog.

## Why

Why The Last Of Us Remastered (PS4 version) is superior to The Last Of Us Part I (remake)?

- The PS4 remaster preserves the artistic style of the PS3 version and just improves the resolution and graphics.
- Last version with multiplayer support, having multiplayer code is much easier to implement community/unofficial multiplayer
- Use less resources and storage space (no 4K textures, less polygons, no ray-tracing) than the remake version, which has a history of performance problems and poor optimization on PC.

## Status

Work in progress, reported playable from start to finish by a player, with one crash along the way
that is not fixed yet (a workaround is under Known issues), and with rough edges.

- **Frame rate:** on a laptop with a Core i7-8750H and a GTX 1050 Ti Mobile (4 GB), 60 FPS in light
  areas and about 40-44 FPS in the heaviest area tested, where the CPU side of the renderer is the
  limit (the draw recording thread at about 87%): lowering the resolution barely helps there.
- **Frame pacing:** frame times swing while the game runs below its limit, and a fixed 30 FPS limit
  (launcher, or the in-game menu) makes motion clearly more even.
- **Known issues:**
  - NPCs moving in jumps or bobbing up and down (GitHub #57, #63): the game read some results of
    its GPU memory copy shader (`Cs_MemCpy`) a frame late. They are read back now; in a save where
    they jumped every time, they stopped. Tested on one save and one GPU so far: report it if they
    still jump somewhere (`GOW3_TLOU_MEMCPY_READBACK=0` turns the fix off for comparison). Thanks
    to KyrieFrost, whose experimental patch in #63 pointed at that shader.
  - A frame that compiles many new shaders can pause the game for a moment, mostly the first time
    an area is played. The game's own frame watchdog is patched from 2 s to 20 s for that
    (`TLOU_FIXES=none` leaves the game unpatched). If a load ever hangs, start with
    `GOW3_FRAMES_IN_FLIGHT=0` (slower).
  - Pistol and rifle shots that went through scenery with no effect (firefly pendants that could
    not be shot down, GitHub #29, #45; walls that took no bullet holes from a pistol or rifle while
    a shotgun marked them) are fixed. The game asks the GPU, with an occlusion query, whether a
    character is under the crosshair; while one is, the player's bullets ignore scenery so thin
    cover in front of a target does not stop them. The runtime made up the answers to occlusion
    queries (always "visible"), so the bullets ignored scenery all the time; the queries are now
    real Vulkan occlusion queries (`GOW3_OCCLUSION_QUERIES=0` goes back to the made-up answers).
    The bullet impacts' mesh ray casts (and the characters' foot probes, as in the NPC jumps)
    also came back to the CPU before the GPU had written them; those results are always read back
    now, also with `GOW3_TLOU_MEMCPY_READBACK=0`.
  - On NVIDIA graphics cards older than Turing (GTX 900 and 10 series), the game's cloth solver for
    its largest cloths (`Cs_ClothSimulate512`) lost the Vulkan device in Pittsburgh, once Henry and
    Sam appear: the runtime stopped with `Device lost during submit`, and Windows' System event log
    shows `nvlddmkm` "SKEDCHECK19_CTA_REGISTER_CONSUMPTION failed". A replay of that route lost it
    on 17 of 18 runs, on drivers 582.53 and 582.78, whatever was changed in the shader. Its one
    workgroup of 512 threads needs all the registers of a multiprocessor; on those cards it now
    runs as 256 threads that simulate 2 vertices each, with the same arithmetic (the shader
    compiler's "lanes", `lane_coarsening.cpp`): no device loss in 6 replays of
    that route. "Cloth workaround" in the launcher's Startup tab, unchecked
    (`GOW3_PASCAL_CLOTH512=1`), runs the solver as it is; `GOW3_PASCAL_CLOTH512=skip` does not
    run it there (that cloth then follows the body). Newer cards run it as it is (an RTX 5060
    went through the same scene). The two-stage draw
    pipeline, blamed before, made the loss more frequent but was not its cause, and is on again
    ("Draw pipeline" in the launcher's Startup tab, `GOW3_DRAW_PIPE=0` turns it off).
  - Intel graphics: an Arc B580 lost the Vulkan device at the logos or the main menu, and a UHD
    630 (Intel driver 31.0.101.2145) showed a white or magenta menu and lost the device when a
    save loaded. Found on the UHD 630 and changed for Intel's Windows driver only:
    - the game's shaders ask for float rounding toward zero, which Intel's driver applies more
      widely than NVIDIA's: the menu scene came out white. It is left out there (`GOW3_RTZ`);
    - a 90-thread prefix sum (cs `cf1b98a8`) has workgroup barriers inside an `if` that differs
      between threads; on the PS4 every wave still runs them, Intel's driver waits forever. Such
      ifs now run for every thread, their LDS writes still following the condition
      (`GOW3_UNIFORM_BARRIERS`; on NVIDIA too since #55, below);
    - large textures were converted with more work groups per dispatch than Intel allows
      (Vulkan validation error), now in pieces; depth/colour copies through a buffer (Intel has
      no `VK_KHR_maintenance8`) copy every level instead of stopping the game;
    - after the God of War III port (SpartanPort): vertex input dynamic state off
      (`GOW3_VERTEX_INPUT_DYNAMIC=1` on), LDS barriers like NVIDIA's (`GOW3_LDS_BARRIERS=0`),
      the compute subgroup size fixed (`GOW3_SUBGROUP_PIN=0`), and sample shading left out on
      one-sample draws (`GOW3_SAMPLE_SHADING_1X=1`).
    The UHD 630 then plays (about 10 FPS at 1080p; 3 replays of 5 minutes without a device loss,
    where all 3 lost it before). With these changes an Intel Arc card was reported to run well,
    with no crash in the session played. The shader cache is rebuilt once on such a GPU. Three cloth shaders still have barriers in such ifs, inside loops (`106beafd`,
    `e44070db`, `20c1dc43`); they did not hang it in the replays tried.
  - Volumetric light with hard edges: sun rays with bright streaks pointing at the sun on RTX
    cards, clearest in photo mode at the start of Pittsburgh, and dark wedges across the screen
    ([#55](https://github.com/ravencaio/tlou-remastered-pc/issues/55)). The game averages its
    radial light shafts with a prefix sum (`CS_RadialAccumulate`, cs `cf1b98a8`) whose workgroup
    barriers sit inside an `if` that differs between threads, as on Intel above. GTX 10 series
    cards happened to get it right; Turing and newer let threads that skip a barrier meet the
    next one, so the sums raced. Those ifs now run for every thread on NVIDIA too
    (`GOW3_UNIFORM_BARRIERS=0` goes back). On NVIDIA cards of every generation, dark or bright wedges through the screen position of a
    volumetric light swept the screen as the camera moved (seen at night in the suburbs). The
    shaders that set up the light's sampling lines (`CS_VolDirectLightBlockInfoCreateOpt`, cs
    `462b5273`, and the line samplers `05866787`, `bb1d430e`) keep arrays in registers, picked
    with a loop that serves the GPU's lanes one index at a time. Two faults there: compute
    shaders were given 4 registers fewer than they allocate, so the arrays lost their top
    elements and a block of lines got infinities; and on NVIDIA's 32-lane groups those loops left
    the last lanes of a group unserved, so a block of lines missed the light's volume. Each lane
    now uses its own index in such loops (same result, `GOW3_WATERFALL_LANES=0` goes back), and
    compute shaders get their full register count. Compute shaders compile again once.
  - A thin light or dark outline where the soft shadows of characters and vehicles meet the
    ground ([#5](https://github.com/ravencaio/tlou-remastered-pc/issues/5),
    [#66](https://github.com/ravencaio/tlou-remastered-pc/issues/66)). Until 0.2.1 it was much
    worse at any Resolution but 1920x1080, with the shadows drawn away from the objects and
    missing under feet and wheels: the game writes its render size into those shaders when it
    loads them, and the shader cache reused shaders translated at another size (every cache made
    before the Resolution setting worked was a 1080p one). Such shaders now compile again once
    after the Resolution changes. The rest is the game's own: it draws those shadows at a quarter
    of its render size, computing each from the top-left pixel of a 4x4 block but blending them
    with the depth of the block's farthest pixel, so along an edge the ground took a character's
    unshadowed value (a light band) and the character's edge the ground's (a dark one), on a PS4
    too (a PS4 Pro's 3200x1800 makes it thinner). The port now blends with the depth the shadows
    were computed at (`GOW3_QUARTER_DEPTH=0`: the game's). Where a foot or a wheel touches the
    ground both have the same depth, so the ground there still takes some of the object's value;
    those shadows are now drawn at half the render size instead of a quarter (960x540 at 1080p,
    finer than a PS4 Pro's 800x450), which halves that ring at some GPU cost. Both are switches in
    the launcher's Video section: **Contact shadows at half resolution** (`TLOU_AMBIENT_SHADOWS`)
    and **Align contact shadows with object edges** (`GOW3_QUARTER_DEPTH`). The first start after
    the update recompiles the materials' shaders once.
- **Not available for this game yet:** a true mouse camera (the mouse drives the right stick
  instead, though no longer capped at the stick's top speed), the draw distance sliders and
  keyboard button prompts, which need the game's own camera code, settings block and prompt
  textures to be located.

## Requirements

- Windows 10 or 11 (64-bit).
- A CPU with AVX, SSE4.2 and POPCNT: Intel 2nd-generation Core (Sandy Bridge) or newer, or any
  AMD Ryzen. The game's code also uses F16C, BMI1, MOVBE and LZCNT; on CPUs older than Haswell
  (Sandy / Ivy Bridge) those instructions are replaced by equivalent code when the game starts
  (`gow3_pc/scripts/isa_emulation.py`, from the
  [Sandy Bridge fork by vitolasanha](https://github.com/vitolasanha/tlou-remastered-pc-sandy-bridge-fix);
  `TLOU_ISA_REWRITE=auto|force|off`, default auto). The first start on such a CPU takes a little
  longer. Sandy Bridge's integrated graphics have no Vulkan, so a graphics card is needed there.
- A graphics card or integrated GPU with Vulkan 1.3 and a current driver. Intel's integrated
  graphics from the 6th generation (Skylake) on reach Vulkan 1.3 only with Intel's own current
  driver; a laptop maker's older driver can stop at 1.2.
- Video memory: 4 GB or more is comfortable at 1080p. Cards of 2 or 3 GB get the launcher's
  **Video memory: 2-3 GB (saving mode)** automatically (`GOW3_LOW_VRAM`): smaller reserves, old
  textures dropped sooner and, when the card is full, data in system memory instead of a crash.
  Pair it with 1280x720 or 1600x900 (or an upscaler) and without PS4 Pro shadows.
- About 6 GB of free RAM for the game.

## Playing

**From a release:** extract the zip, open `TLOU Launcher.exe`, choose the game folder under Folders
and press Play. Nothing else needs installing. After that, a shortcut with the game's icon appears
beside the launcher. `TLOU Game.exe` starts the game directly with the last settings selected in
the launcher.

**From this repository** (needs Python 3.10 or newer):

1. Put the game folder `CUSA00552` next to this repository's folder, and the update next to it as
   `CUSA00552-patch` (or name another update folder with `TLOU_UPDATE=<folder>`). An update you
   only have as a PKG unpacks there with `python tools/install_update.py <update.pkg> <game folder>`.
2. Build the runtime (below), then start the game with `launcher/TLOU Launcher.exe`,
   `Start-TLOU.cmd`, or `python tools/probe_tlou.py --boot --seconds 0 [--game <game folder>]`.
3. Saves and settings stay in `analysis/tlou-user`; the log is `analysis/tlou/boot.log`. Each
   session also records your controller input (`analysis/tlou/pad-record-*.txt`, the newest 5),
   so a crash can be replayed on the same route; `TLOU_RECORD_PAD=0` turns that off.
4. With this repository on a hard disk, put the shader cache (`analysis/tlou-user/cache`, about
   250 MB in some 12,000 files) on an SSD: the game reads all of it in its first seconds, and
   from a hard disk that takes long enough for the game to build pipelines itself, pausing for
   seconds. With the game closed, move the folder, then link it back:
   `mklink /J analysis\tlou-user\cache <folder on the SSD>` (in `cmd`).

The launcher's Resolution is the size the game draws at (1280x720 to 3840x2160 or 5120x1440;
any `<width>x<height>` in `output_res` of `tlou.ini`; `TLOU_RESOLUTION` overrides it): the game's
own table of render sizes, which the PS4 Pro used, is patched at start, and above 1080p its render
targets get more GPU memory, as on a PS4 Pro. It takes effect on the next start; the in-game menu
shows it under Image > Information. Higher sizes cost GPU time (a GTX 1050 Ti Mobile: about 15 FPS
at 3840x2160).

Ultrawide monitors: a Resolution of another aspect ratio than 16:9 (2560x1080, 3440x1440,
3840x1600, 5120x1440, ...) also patches the game's camera to show more at the sides with the same
vertical view, keeps the interface 16:9 in the middle (its full-screen layers, such as the menus'
dimming, the fades and the rifle scope's black frame, reach the screen's edges), and puts the
cutscene movies between black bars instead of stretching them. These patches follow SanyaGames227's
[tlou_r_shadps4_ultrawide](https://github.com/SanyaGames227/tlou_r_shadps4_ultrawide) for
shadPS4. The width must be a multiple of 16 (the game asserts it; other widths are rounded down).
`TLOU_ASPECT=0` leaves the image stretched.

The launcher sets the display, frame limit and folders, in nine languages; its Debug menu switches
each optimization off, some of them while the game runs. The in-game menu (Insert, or R3+L2 on a
controller) holds the graphics, controls and frame rate settings.

**Keyboard and mouse:** WASD or the arrow keys move, the mouse turns the camera, right and left
mouse buttons aim and fire, Space or Enter is Cross, C Circle, F Square, E Triangle, Left Shift
L1, Tab R1, V L3, T R3, Esc or Backspace Options, 1 2 3 4 (and the mouse wheel) the d-pad. The
launcher's Controls window and the in-game menu rebind them.

**Without a DualShock 4:** the touchpad (backpack, crafting) is Caps Lock on the keyboard or
Back/View on an Xbox-style controller. Shaking the controller is G on the keyboard or the
Share/Capture button (`GOW3_KEY_MAP` / `GOW3_PAD_MAP` `shake=...` rebind it). The flashlight
never flickers out by default, so it needs no shake: Celogamez's GoldHEN cheat "Flashlight Does
Not Turn Off" (`TLOU_CHEATS=none` turns it off).

**Walking on the keyboard:** Left Ctrl (`walk=...` in the key map) keeps the left stick, keys or
controller, at 75% of a full push. The launcher's Controls window and the in-game menu choose
whether the key is held or switches walking on and off, and how far the stick goes
(`walk_toggle`, `walk_speed` in tlou.ini; `GOW3_WALK_TOGGLE`, `GOW3_WALK_SPEED`).

**Vibration:** the launcher's Controls window (Controller tab, which also shows a drawn controller
following the controller and the bound keys) and the in-game menu turn the controller's vibration
off (`vibration=0` in tlou.ini; `GOW3_VIBRATION=0`).

**Aim assist:** the launcher's Controls window (Controller and Mouse tabs) turns it off
(`aim_assist=0` in tlou.ini; `TLOU_AIM_ASSIST=0`), from the next start of the game. Off, every aim
setting the game loads (per weapon and difficulty) gets its four assist lists emptied, as in the
game's own `no-aim-assist` setting: the crosshair is not helped to follow enemies, and each weapon
keeps its speeds, zoom and stick curves.

**Languages:** the game's language menus offer all 21 of its languages, not only the four of the
Americas release (`TLOU_ALL_LANGUAGES=0` keeps those four): English, French, Italian, German,
Spanish (Spain and Latin America), Portuguese (Portugal and Brazil), Dutch, Danish, Finnish,
Norwegian, Swedish, Polish, Russian, Czech, Greek, Turkish, Japanese, Korean and Chinese
(traditional). The executable is the same in every region and picks the region's languages from a
table; the menus and subtitles of all of them are already in the Americas files. The launcher's
Game language is the PS4 system language the game sees (`system_language` in tlou.ini;
`GOW3_LANGUAGE`, a PS4 language id): the game starts in it, and offers Korean and Chinese only when
it is one of those. Spoken dialogue in German, Italian, Spanish (Spain), Portuguese (Portugal),
Polish, Russian or Japanese needs that dub's files from your copy of another region's release
(`<language>-sp.psarc`, and `<language>-mp.psarc` for multiplayer, in
`build/ps4/main/sound1/streams/speech` of the game folder; the update already carries their
changes). The Dialogue setting of the language menu offers only the dubs whose files are in the
game or update folder (the game would otherwise ask to download that "language pack"), and
PlayGo reports the others as not installed, so a save that asks for one gets English
(`GOW3_PLAYGO_ALL_LOCAL=1` reports every dub as installed, as before).

## Building

From an MSYS2 **CLANG64** shell:

```bash
pacman -S --needed git patch \
  mingw-w64-clang-x86_64-{toolchain,cmake,ninja,pkgconf,python} \
  mingw-w64-clang-x86_64-{vulkan-headers,vulkan-loader,vulkan-memory-allocator,glslang} \
  mingw-w64-clang-x86_64-{sdl3,fmt,boost,robin-map,xxhash,ffmpeg,zydis} \
  mingw-w64-clang-x86_64-{spirv-tools,spirv-headers,spirv-cross}
git submodule update --init --recursive
gow3_pc/build.sh                 # the runtime: gow3_pc/out/gow3-probe.exe
```

Clone with `core.autocrlf=false`: line-ending conversion breaks the FSR-Vulkan patches.

`gow3_pc/build.sh` also builds `launcher/TLOU Launcher.exe`. With a profile in
`gow3_pc/pgo/tlou.profdata` it builds with PGO and LTO (about 10% more FPS in the heaviest area
tested); the profile is not in the repository, and without it the build is plain. `GOW3_PGO=off`
gives quicker development builds. A new profile needs a clean build.

`tools/package_windows.sh` builds `dist/TLOU-Remastered-windows.zip`, the release: the runtime and
its DLLs, the launcher, the scripts and a Python of its own. `tools/package_check.py` refuses game
files and paths of the build machine.

The launcher reads its background and icon from the player's own copy of the game. A local icon goes
in `launcher/assets/icon.ico` (not in the repository).

## Repository layout

| Path | Contents |
|---|---|
| `gow3_pc/` | The runtime: C runtime (`src/`), GPU library (`gpu/`: shadPS4 video core and shims), preparation scripts (`scripts/`). Diagnostics settings: [gow3_pc/README.md](gow3_pc/README.md). |
| `launcher/` | The launcher (`tlou_launcher.pyw`, Tkinter) and its starter (`tlou-launcher.c`). |
| `tools/package_windows.sh` | The Windows release zip. |
| `tools/probe_tlou.py` | Prepares the game's executable (with its update) and starts the runtime with this game's settings. |
| `tools/tlou_pad.py`, `tools/tlou_bench.py` | Scripted controller input, frame captures and analysis; a repeatable frame rate measurement on a test profile. |
| `tools/tlou_replay.py`, `tools/tlou_compare_shots.py` | A recorded route replayed on a copy of the player's saves, reporting a lost device, with frames saved at the same moments (and two runs' frames compared). |
| `tools/tlou_find_shader.py` | The game file a shader comes from, by the hash the logs name it by (or shaders by file name). |
| `tools/decode_dump.py`, other `tools/` | Image dump decoding and other diagnostics. |
| `tools/install_update.py` | Unpacks an update PKG next to the game folder (PkgTool, `tools/pkgtool`). |

## Credits and license

- The runtime comes from BilalReloaded's PS4 port runtime
  ([GitHub](https://github.com/bc160201983)), which started from FelipePrado0's
  [SpartanPort](https://github.com/FelipePrado0/SpartanPort) (gow3_pc; whose internal names, such as
  `GOW3_*` settings, it keeps), and the [shadPS4](https://github.com/shadps4-emu/shadPS4) video core.
- The camera motion for temporal anti-aliasing (`CameraMotion::OnVelocityConstants` and the motion
  vector mode) comes from SuperMedo's PS4 port ([GitHub](https://github.com/Supermedo), GPL v2).
- Windows red-zone protection and the reader-preferring memory table lock come from HayatoG's
  [tlou_pc](https://github.com/HayatoG/tlou_pc) (GPL-2.0-or-later); the protection follows
  shadPS4's design.
- The 1 ms timer, the power throttling opt-out, the faster guest condition variables, the idle job
  worker patch's method and the package's Python copy come from BilalReloaded's
  [Uncharted: The Nathan Drake Collection port](https://github.com/bc160201983/Uncharted-The-Nathan-Drake-Collection-PC-port).
- **NVIDIA DLSS:** DLAA uses NVIDIA DLSS technology through the DLSS bridge adapted from
  [IFreemz/shadPS4-Bloodborne-DLSS-FSR](https://github.com/IFreemz/shadPS4-Bloodborne-DLSS-FSR)
  (MIT). NVIDIA, RTX and DLSS are trademarks of NVIDIA Corporation.
- AMD FSR 4: the v07 INT8 model as built by [Q2RTX](https://github.com/FireBurn/Q2RTX) from AMD's
  MIT-licensed FidelityFX SDK source.
- Third-party code: FSR (AMD), imgui, magic_enum, miniz, xbyak, LibAtrac9, FFmpeg, SDL3, PkgTool
  (LibOrbisPkg, LGPL-3.0).
- Licensed under the GNU GPL v2 or later (see `LICENSE`).
