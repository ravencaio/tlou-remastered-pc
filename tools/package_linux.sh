#!/usr/bin/env bash
# Builds dist/TLOU-Remastered-linux/ and dist/TLOU-Remastered-linux.zip: a folder that runs on a
# 64-bit Linux PC with a Vulkan driver, without Python installed.
#   TLOU-Launcher        opens the settings launcher
#   TLOU-Game            starts the game using the launcher's saved settings
#   python/              a standalone CPython with Tkinter (python-build-standalone) and Pillow
#   launcher/ tools/     the launcher and the start script (tools/probe_tlou.py)
#   gow3_pc/scripts/     the preparation scripts; gow3_pc/out/ the runtime and libgow3gpu.so
# The runtime links the distribution's own Vulkan loader, SDL3 and FFmpeg: they do not ship
# (licenses/THIRD-PARTY.txt lists them), so the package runs where those libraries have the
# sonames of the system it was built on. No game file ships (tools/package_check.py).
# Run in the repository on Linux. --no-build uses the runtime already in gow3_pc/out.
# .github/workflows/linux-package.yml runs it for each release in an Arch Linux container.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
name=TLOU-Remastered-linux
# python-build-standalone's install_only_stripped build (no debug symbols), pinned by its SHA-256 (kept in downloads/).
python_url=https://github.com/astral-sh/python-build-standalone/releases/download/20261124/cpython-3.12.16%2B20261124-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
python_sha256=7e1adaea26c959d13aa1120be1ae62a1ea1bb52adcb382679fffe9217e22adff
python_tar=downloads/cpython-3.12.16+20261124-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz

if [[ ${1:-} != --no-build ]]; then
    # The GPU library's sources are compiled by absolute path: keep this checkout's path out of
    # the asserts' file names (package_check.py refuses paths of the building machine).
    export CFLAGS="${CFLAGS:-} -ffile-prefix-map=$PWD=." CXXFLAGS="${CXXFLAGS:-} -ffile-prefix-map=$PWD=."
    bash gow3_pc/build.sh
fi
[[ -x gow3_pc/out/gow3-probe && -f gow3_pc/out/gpu/libgow3gpu.so ]] || {
    echo 'Build the runtime first: gow3_pc/build.sh' >&2; exit 1; }
for tool in patchelf strip zip; do
    command -v "$tool" >/dev/null || { echo "Need $tool" >&2; exit 1; }
done

stage=dist/stage/$name
rm -rf -- dist/stage
mkdir -p "$stage/gow3_pc/out/gpu" "$stage/gow3_pc/out/pydeps" "$stage/gow3_pc/scripts" "$stage/tools" \
    "$stage/launcher" downloads

# The runtime and the GPU library without debug information. The build's rpath also names this
# checkout's out/gpu: the package keeps only $ORIGIN/gpu.
strip --strip-debug -o "$stage/gow3_pc/out/gow3-probe" gow3_pc/out/gow3-probe
strip --strip-debug -o "$stage/gow3_pc/out/gpu/libgow3gpu.so" gow3_pc/out/gpu/libgow3gpu.so
patchelf --set-rpath '$ORIGIN/gpu' "$stage/gow3_pc/out/gow3-probe"
patchelf --remove-rpath "$stage/gow3_pc/out/gpu/libgow3gpu.so"
chmod 755 "$stage/gow3_pc/out/gow3-probe"
libraries=$(ldd "$stage/gow3_pc/out/gow3-probe")
echo "Libraries the runtime loads from the system:"
echo "$libraries" | sed 's/^/  /'
if [[ $libraries == *'not found'* ]]; then
    echo 'The packaged runtime has unresolved libraries' >&2; exit 1
fi

# The Python that runs the launcher and the scripts.
if [[ ! -f $python_tar ]] || ! echo "$python_sha256  $python_tar" | sha256sum -c --quiet -; then
    curl -fL --retry 3 -o "$python_tar.part" "$python_url"
    echo "$python_sha256  $python_tar.part" | sha256sum -c --quiet -
    mv "$python_tar.part" "$python_tar"
fi
tar -xzf "$python_tar" -C "$stage"            # python/
# The zip stores a symbolic link as a copy of its target: keep one interpreter, python3, as a file,
# and the library under the name the interpreter loads.
bin=$stage/python/bin
cp --remove-destination -- "$(readlink -f -- "$bin/python3")" "$bin/python3.tmp"
find "$bin" -maxdepth 1 -name 'python*' ! -name python3.tmp -delete
mv "$bin/python3.tmp" "$bin/python3"
find "$stage/python/lib" -maxdepth 1 -type l -name 'libpython*' -delete
python=$stage/python/bin/python3
lib=$(echo "$stage"/python/lib/python3.*)
"$python" -I -m pip install --no-cache-dir --disable-pip-version-check --no-warn-script-location -q Pillow
"$python" -I -m pip install --no-cache-dir --disable-pip-version-check -q --target "$stage/gow3_pc/out/pydeps" iced-x86
# Not needed to run: tests, IDLE, pip and its installers, the static library and headers.
rm -rf -- "$lib"/{test,idlelib,ensurepip,turtledemo,lib2to3,pydoc_data} "$lib"/site-packages/pip* \
    "$lib"/config-3.*/*.a "$stage/python/include" "$stage/python/share" "$stage"/gow3_pc/out/pydeps/bin
find "$stage/python" "$stage/gow3_pc/out/pydeps" -name __pycache__ -type d -prune -exec rm -rf -- {} +
find "$stage/python/bin" -maxdepth 1 \( -name 'pip*' -o -name 'idle*' -o -name 'pydoc*' -o -name '2to3*' \) -delete

cp gow3_pc/scripts/*.py "$stage/gow3_pc/scripts/"
cp tools/probe_tlou.py tools/run_game.py tools/apply_update.py tools/gow3_data.py tools/tlou_render_settings.py "$stage/tools/"
cp launcher/tlou_launcher.pyw launcher/updater.py "$stage/launcher/"

# The starters: the package's Python on the launcher and on run_game.py (TLOU Game.exe's job).
cat > "$stage/TLOU-Launcher" <<'EOF'
#!/bin/sh
# Opens the launcher of The Last of Us Remastered (PC port) with the package's Python.
here=$(dirname -- "$(readlink -f -- "$0")")
exec "$here/python/bin/python3" "$here/launcher/tlou_launcher.pyw" "$@"
EOF
cat > "$stage/TLOU-Game" <<'EOF'
#!/bin/sh
# Starts the game with the settings last saved in the launcher, without opening it.
here=$(dirname -- "$(readlink -f -- "$0")")
exec "$here/python/bin/python3" "$here/tools/run_game.py" "$@"
EOF
chmod 755 "$stage/TLOU-Launcher" "$stage/TLOU-Game"

# The package's version: the launcher's updates (launcher/updater.py) compare it with the releases.
cp tools/VERSION "$stage/VERSION"
cp gow3_pc/packaging/linux/README-player.txt "$stage/README.txt"
cp gow3_pc/LICENSE "$stage/LICENSE.txt"

# Licenses of the sources built into the runtime, of the Python that ships, and the list of the
# system libraries the runtime needs (not shipped).
licenses=$stage/licenses
mkdir -p "$licenses/python"
for dir in gow3_pc/gpu/third_party/* gow3_pc/third_party/*; do
    for file in "$dir"/LICENSE* "$dir"/COPYING* "$dir"/COPYRIGHT*; do
        [[ -f $file ]] || continue
        mkdir -p "$licenses/$(basename -- "$dir")"
        cp "$file" "$licenses/$(basename -- "$dir")/"
    done
done
cp "$lib/LICENSE.txt" "$licenses/python/LICENSE.txt"
{
    echo "The runtime (gow3_pc/out/gow3-probe) uses these libraries of the system; they do not ship."
    echo "Built on: $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo
    echo "$libraries" | awk '$3 ~ /^\// && $1 !~ /libgow3gpu/ {print $1}' | sort -u
    if command -v pacman >/dev/null; then
        echo
        echo "Arch Linux packages that provide them:"
        echo "$libraries" | awk '$3 ~ /^\// && $1 !~ /libgow3gpu/ {print $3}' | sort -u |
            { xargs pacman -Qqo 2>/dev/null || true; } | sort -u | xargs -r pacman -Q
    fi
    echo
    echo "python/: CPython from https://github.com/astral-sh/python-build-standalone ($python_url),"
    echo "with Pillow; their licenses are in python/ and licenses/python."
} > "$licenses/THIRD-PARTY.txt"

"$python" -I tools/package_check.py "$stage" "dist/$name.zip"
rm -rf -- "dist/$name"
mv "$stage" "dist/$name"
rm -rf -- dist/stage
ls -la "dist/$name.zip"
