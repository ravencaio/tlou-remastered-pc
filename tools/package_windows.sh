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
cp gow3_pc/scripts/*.py "$stage/gow3_pc/scripts/"
cp tools/probe_tlou.py tools/run_game.py tools/apply_update.py "$stage/tools/"
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

"$python_home/python.exe" tools/copy_python.py "$(cygpath -w "$python_home")" "$(cygpath -w "$stage/python")"
"$python_home/python.exe" -I tools/package_check.py "$(cygpath -w "$stage")" "$(cygpath -w "dist/$name.zip")"
rm -rf -- "dist/$name"
mv "$stage" "dist/$name"
rm -rf -- dist/stage
ls -la "dist/$name.zip"
