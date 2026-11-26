#!/usr/bin/env bash
# Builds dist/TLOU-Remastered-windows/ and dist/TLOU-Remastered-windows.zip: a folder that runs on a
# Windows 10/11 x64 PC with a Vulkan GPU, without Python or MSYS2 installed.
#   TLOU Launcher.exe    opens the settings launcher
#   TLOU Game.exe        starts the game using the launcher's saved settings
#   python\              Windows Python with Tkinter and Pillow (tools/copy_python.py)
#   launcher\ tools\     the launcher and the start script (tools/probe_tlou.py)
#   gow3_pc\scripts\     the preparation scripts; gow3_pc\out\ the runtime and its DLLs
# No game file ships, the launcher's icon included (launcher/assets is the cover art): each player
# selects their own copy of the game in the launcher, which reads its art from there.
# From Git Bash, in the repository. --no-build uses the runtime already in gow3_pc/out.
# PYTHON_HOME is a Windows Python 3.10+ with Pillow (default: the `python` on PATH).
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
name=TLOU-Remastered-windows
# The Windows Python (looked up before the CLANG64 tools go first on PATH: MSYS2 has its own).
python=${PYTHON:-python}
python_home=${PYTHON_HOME:-$("$python" -c 'import sys, os; print(os.path.dirname(sys.executable))')}
python_home=$(cygpath -u "$python_home")
msys=${MSYS2_ROOT:-/c/msys64}
msys=${msys%/}

if [[ ${1:-} != --no-build ]]; then
    bash gow3_pc/build.sh
fi
[[ -d $msys/clang64/bin ]] && export PATH="$msys/clang64/bin:$PATH"
[[ -f gow3_pc/out/gow3-probe.exe ]] || { echo 'Build the runtime first: gow3_pc/build.sh' >&2; exit 1; }
[[ -f gow3_pc/out/pydeps/iced_x86/__init__.py ]] || {
    echo 'Need iced-x86: python -m pip install --no-cache-dir --target gow3_pc/out/pydeps iced-x86' >&2; exit 1; }

stage=dist/stage/$name
rm -rf -- dist/stage
mkdir -p "$stage/gow3_pc/out/pydeps" "$stage/gow3_pc/scripts" "$stage/tools" "$stage/launcher"

# The runtime without debug information, and every DLL it loads from the CLANG64 tree.
llvm-strip --strip-debug -o "$stage/gow3_pc/out/gow3-probe.exe" gow3_pc/out/gow3-probe.exe
ldd gow3_pc/out/gow3-probe.exe | awk '/clang64/ {print $3}' | sort -u | while read -r dll; do
    cp "$dll" "$stage/gow3_pc/out/"
done
cp -r gow3_pc/out/pydeps/iced_x86 gow3_pc/out/pydeps/iced_x86-*.dist-info "$stage/gow3_pc/out/pydeps/"
rm -rf -- "$stage/gow3_pc/out/pydeps/iced_x86/__pycache__"   # (paths of this machine)
# iced_x86's modules ship as bytecode only. Its generated Code.py (only constants and docstrings)
# is flagged as JS/TrojanDropper.ModiLoader by ESET, Google and Ikarus and as Script.Trojan.Agent
# by GData on VirusTotal, and with it the whole package (#88, #112, #148). The Python that ships is
# a copy of this one, so the bytecode matches it. The type stubs are not needed to run.
"$python_home/python.exe" -I -m compileall -q -b -o 2 "$(cygpath -w "$stage/gow3_pc/out/pydeps/iced_x86")"
find "$stage/gow3_pc/out/pydeps/iced_x86" -name '*.py' -delete
rm -f -- "$stage/gow3_pc/out/pydeps/iced_x86/_iced_x86_py.pyi"
cp gow3_pc/scripts/*.py "$stage/gow3_pc/scripts/"
cp tools/probe_tlou.py tools/run_game.py tools/apply_update.py tools/gow3_data.py tools/tlou_render_settings.py "$stage/tools/"
cp launcher/tlou_launcher.pyw launcher/updater.py "$stage/launcher/"
# The starters, without the icon: the icon is the game's cover.
bash launcher/build-starters.sh "$stage/TLOU Launcher.exe" "$stage/TLOU Game.exe"
# The package's version: the launcher's updates (launcher/updater.py) compare it with the releases.
cp tools/VERSION "$stage/VERSION"
cp gow3_pc/packaging/windows/README-player.txt "$stage/README.txt"
cp gow3_pc/LICENSE "$stage/LICENSE.txt"

# Licenses of the sources built into the runtime, and of every MSYS2 package whose DLL ships.
licenses=$stage/licenses
mkdir -p "$licenses"
for dir in gow3_pc/gpu/third_party/* gow3_pc/third_party/*; do
    for file in "$dir"/LICENSE* "$dir"/COPYING* "$dir"/COPYRIGHT*; do
        [[ -f $file ]] || continue
        mkdir -p "$licenses/$(basename -- "$dir")"
        cp "$file" "$licenses/$(basename -- "$dir")/"
    done
done
pacman=$msys/usr/bin/pacman
dlls=()
for dll in "$stage"/gow3_pc/out/*.dll; do
    dlls+=("$(cygpath -m "$msys/clang64/bin/$(basename -- "$dll")")")
done
mapfile -t packages < <({ "$pacman" -Qqo "${dlls[@]}" 2>/dev/null || true; } | sort -u)
{ "$pacman" -Qlq "${packages[@]}" | grep '/share/licenses/.*[^/]$' || true; } | while read -r file; do
    file=${file%$'\r'}
    name_part=${file#*/share/licenses/}
    mkdir -p "$licenses/msys2/$(dirname -- "$name_part")"
    cp "$msys$file" "$licenses/msys2/$name_part"
done
{
    echo "Third-party packages whose DLLs ship in gow3_pc\\out (MSYS2 CLANG64 builds)."
    echo "Their sources: https://repo.msys2.org/mingw/sources/ and https://packages.msys2.org/"
    "$pacman" -Qi "${packages[@]}" | awk '/^Name /{print ""} /^(Name|Version|Licenses|URL) /{print}'
} > "$licenses/THIRD-PARTY.txt"

# NVIDIA DLSS (DLAA and the quality presets on RTX cards): NVIDIA's unmodified nvngx_dlss.dll and
# the bridge gow3_dlss.dll (MIT, gow3_pc/gpu/dlss_bridge), the only part linked with the DLSS SDK;
# the runtime loads it at run time. tools/fetch_aa_files.sh fetches the pinned SDK (kept in
# downloads/) and builds the bridge with MSVC (Visual Studio 2022 or its Build Tools). NVIDIA's
# license ships beside the bridge's. Neither DLL needs the Visual C++ runtime. About 30 MiB of the
# zip. Without an RTX card the menus say DLSS needs one and FSR 3.1 runs.
bash tools/fetch_aa_files.sh dlss
cp gow3_pc/out/gow3_dlss.dll gow3_pc/out/nvngx_dlss.dll "$stage/gow3_pc/out/"
mkdir -p "$licenses/nvidia-dlss" "$licenses/gow3-dlss-bridge"
dlss_version=$(sed -n 's/^    version=//p' tools/fetch_aa_files.sh | head -1)   # the pinned SDK
cp "downloads/dlss-sdk-$dlss_version/LICENSE.txt" "$licenses/nvidia-dlss/LICENSE.txt"
cp gow3_pc/gpu/dlss_bridge/LICENSE.txt "$licenses/gow3-dlss-bridge/LICENSE.txt"

# Intel XeSS: Intel's unmodified libxess.dll (pinned and checksummed by tools/fetch_aa_files.sh,
# kept in downloads/ after the first run), redistributable under the Intel Simplified Software
# License, which ships beside it with its third-party notices. About 56 MiB of the zip. It runs on
# GPUs with VK_EXT_mutable_descriptor_type and needs the Visual C++ 2015-2022 runtime (x64); on
# the others the game says why and uses FSR 3.1.
bash tools/fetch_aa_files.sh xess
cp gow3_pc/out/libxess.dll "$stage/gow3_pc/out/"
mkdir -p "$licenses/xess"
cp gow3_pc/out/libxess-LICENSE.txt "$licenses/xess/LICENSE.txt"
cp gow3_pc/out/libxess-third-party-programs.txt "$licenses/xess/third-party-programs.txt"

"$python_home/python.exe" tools/copy_python.py "$(cygpath -w "$python_home")" "$(cygpath -w "$stage/python")"
"$python_home/python.exe" -I tools/package_check.py "$(cygpath -w "$stage")" "$(cygpath -w "dist/$name.zip")"
rm -rf -- "dist/$name"
mv "$stage" "dist/$name"
rm -rf -- dist/stage
ls -la "dist/$name.zip"
