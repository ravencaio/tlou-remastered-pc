#!/usr/bin/env bash
# Builds dist/TLOU-Remastered-linux/ and dist/TLOU-Remastered-linux.zip: a folder that runs on a
# 64-bit Linux PC with a Vulkan driver, without Python installed.
#   TLOU-Launcher        opens the settings launcher
#   TLOU-Game            starts the game using the launcher's saved settings
#   python/              a standalone CPython with Tkinter (python-build-standalone) and Pillow
#   launcher/ tools/     the launcher and the start script (tools/probe_tlou.py)
#   gow3_pc/scripts/     the preparation scripts; gow3_pc/out/ the runtime and libgow3gpu.so
#   gow3_pc/out/lib/     every library the runtime needs beyond the system's own (FFmpeg, SDL3,
#                        Zydis, xxHash, the Vulkan loader, X11): the package does not depend on
#                        the distribution's versions of them (a Steam Deck has few of Arch's)
# The system provides glibc, the C++ runtime, the graphics driver and what SDL3 opens itself
# (licenses/THIRD-PARTY.txt lists both sets). No game file ships (tools/package_check.py).
# Run in the repository on Linux. --no-build uses the runtime already in gow3_pc/out; build it
# with GOW3_PORTABLE=1 and PKG_CONFIG_PATH on tools/build_ffmpeg_min.sh's FFmpeg, as the release
# workflow does (a build with the distribution's FFmpeg ships all of FFmpeg's own dependencies).
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
    GOW3_PORTABLE=1 bash gow3_pc/build.sh
fi
[[ -x gow3_pc/out/gow3-probe && -f gow3_pc/out/gpu/libgow3gpu.so ]] || {
    echo 'Build the runtime first: gow3_pc/build.sh' >&2; exit 1; }
for tool in patchelf strip zip; do
    command -v "$tool" >/dev/null || { echo "Need $tool" >&2; exit 1; }
done

stage=$PWD/dist/stage/$name   # absolute: ldd then prints the shipped libraries by full path
rm -rf -- dist/stage
mkdir -p "$stage/gow3_pc/out/gpu" "$stage/gow3_pc/out/pydeps" "$stage/gow3_pc/scripts" "$stage/tools" \
    "$stage/launcher" downloads

# The runtime and the GPU library without debug information. The build's rpath also names this
# checkout's out/gpu: the package keeps only $ORIGIN/gpu.
strip --strip-debug -o "$stage/gow3_pc/out/gow3-probe" gow3_pc/out/gow3-probe
strip --strip-debug -o "$stage/gow3_pc/out/gpu/libgow3gpu.so" gow3_pc/out/gpu/libgow3gpu.so
patchelf --set-rpath '$ORIGIN/lib:$ORIGIN/gpu' "$stage/gow3_pc/out/gow3-probe"
patchelf --set-rpath '$ORIGIN/../lib' "$stage/gow3_pc/out/gpu/libgow3gpu.so"
chmod 755 "$stage/gow3_pc/out/gow3-probe"

# Libraries that ship in gow3_pc/out/lib: every one the runtime loads (FFmpeg, SDL3, Zydis,
# xxHash, the Vulkan loader, X11/xcb, ...) except those that must be the system's own: glibc,
# the graphics driver's (GL/EGL/DRM) and what SDL3 opens itself (Wayland, audio, udev, D-Bus).
# A distribution's libraries change soname from one release to the next: SteamOS (Steam Deck)
# has few of Arch's current ones. The release workflow builds FFmpeg small
# (tools/build_ffmpeg_min.sh) and the C++ runtime into libgow3gpu.so (GOW3_PORTABLE), so the
# system's libstdc++ stays the one its Vulkan driver needs (libgcc_s, always there, too).
system_lib='^(linux-vdso|ld-linux[a-z0-9-]*|libc|libm|libdl|libpthread|librt|libresolv|libutil|libmvec|libgcc_s|libGL|libEGL|libGLX|libGLdispatch|libOpenGL|libgbm|libdrm[a-z_]*|libwayland-[a-z]*|libxkbcommon[a-z0-9-]*|libdecor-[0-9]*|libasound|libpulse[a-z-]*|libpipewire-[0-9.]*|libjack|libudev|libdbus-1)\.so'
mkdir -p "$stage/gow3_pc/out/lib"
# Where the build found its libraries outside the loader's paths (tools/build_ffmpeg_min.sh's
# prefix): searched to copy them, never in the checks below.
build_dirs=
for module in libavcodec sdl3; do
    build_dirs+=$(pkg-config --variable=libdir "$module" 2>/dev/null || true):
done
for elf in "$stage/gow3_pc/out/gow3-probe" "$stage/gow3_pc/out/gpu/libgow3gpu.so"; do
    LD_LIBRARY_PATH=$build_dirs${LD_LIBRARY_PATH:-} ldd "$elf" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}'
done | sort -u | while read -r soname path; do
    [[ $soname =~ $system_lib || $soname == libgow3gpu.so ]] && continue
    case $soname in libstdc++.so*)
        echo "$soname comes from the building system: build with GOW3_PORTABLE=1" >&2; exit 1;;
    esac
    cp -L -- "$path" "$stage/gow3_pc/out/lib/$soname"
done
for so in "$stage"/gow3_pc/out/lib/*.so*; do
    strip --strip-unneeded "$so"
    patchelf --set-rpath '$ORIGIN' "$so"
    chmod 644 "$so"
done
libraries=$(env -u LD_LIBRARY_PATH ldd "$stage/gow3_pc/out/gow3-probe")
echo "Libraries the runtime loads (gow3_pc/out/lib ships with it):"
echo "$libraries" | sed 's/^/  /'
if [[ $libraries == *'not found'* ]]; then
    echo 'The packaged runtime has unresolved libraries' >&2; exit 1
fi
# What is not shipped must be a system library; a missing ldd of a bundled one's dependency
# would show here too.
outside=$(echo "$libraries" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' |
    while read -r soname path; do
        [[ $path == "$stage/"* || $soname =~ $system_lib ]] || echo "$soname"
    done)
if [[ -n $outside ]]; then
    echo "Libraries loaded from the building system but not shipped: $outside" >&2; exit 1
fi
# The oldest glibc the package runs on: the highest GLIBC_ symbol version any shipped program
# or library asks for (SteamOS 3.7: glibc 2.41).
glibc=$(find "$stage/gow3_pc/out" -type f \( -name '*.so*' -o -name gow3-probe \) -exec objdump -T {} + 2>/dev/null |
    grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "Needs glibc $glibc or newer"
# The Steam Deck's glibc (SteamOS 3.7: 2.41) is the floor; GOW3_MAX_GLIBC raises it on purpose.
max_glibc=${GOW3_MAX_GLIBC:-2.41}
if [[ $(printf '%s\n%s\n' "$glibc" "$max_glibc" | sort -V | tail -1) != "$max_glibc" ]]; then
    echo "The package needs glibc $glibc, newer than $max_glibc (Steam Deck): it would not start there" >&2
    exit 1
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

# Licenses of the sources built into the runtime, of the libraries and the Python that ship, and
# the list of the system libraries the runtime needs (not shipped).
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
# The shipped libraries' licenses: Arch's package of each (/usr/share/licenses), FFmpeg's from
# tools/build_ffmpeg_min.sh's prefix.
shipped=$(echo "$libraries" | awk -v dir="$stage/gow3_pc/out/lib/" 'index($3, dir) == 1 {print $1}' | sort -u)
system=$(echo "$libraries" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' | grep -v -F "$stage/" | sort -u)
origins=
for soname in $shipped; do
    # The Arch package of the system copy it was taken from.
    source=$(ldconfig -p 2>/dev/null | awk -v n="$soname" '$1 == n {print $NF; exit}')
    owner=
    if [[ -n $source ]] && command -v pacman >/dev/null; then
        owner=$({ pacman -Qo "$source" 2>/dev/null || true; } | awk '{print $(NF-1), $NF}')
    fi
    if [[ -n $owner ]]; then
        origins+="  $soname: Arch $owner"$'
'
        if [[ -d /usr/share/licenses/${owner% *} ]]; then
            mkdir -p "$licenses/${owner% *}"
            cp -rL /usr/share/licenses/"${owner% *}"/. "$licenses/${owner% *}/"
        fi
    else
        origins+="  $soname"$'
'
    fi
done
ffmpeg_source=
if [[ $shipped == *libavcodec* ]]; then
    ffmpeg_prefix=$(pkg-config --variable=prefix libavcodec)
    if [[ -f $ffmpeg_prefix/share/licenses/ffmpeg/SOURCE ]]; then
        mkdir -p "$licenses/ffmpeg"
        cp "$ffmpeg_prefix"/share/licenses/ffmpeg/* "$licenses/ffmpeg/"
        ffmpeg_source=$(cat "$licenses/ffmpeg/SOURCE")
    fi
fi
{
    echo "Built on: $(. /etc/os-release && echo "$PRETTY_NAME"); needs glibc $glibc or newer."
    echo
    echo "Libraries that ship in gow3_pc/out/lib (their licenses are in this folder):"
    printf '%s' "$origins"
    if [[ -n $ffmpeg_source ]]; then
        echo "  FFmpeg's (LGPL 2.1, decoders only) were built from $ffmpeg_source"
    fi
    echo
    echo "Libraries of the system the runtime uses; they do not ship:"
    echo "$system" | awk '{print "  " $1}'
    echo
    echo "python/: CPython from https://github.com/astral-sh/python-build-standalone ($python_url),"
    echo "with Pillow; their licenses are in python/ and licenses/python."
} > "$licenses/THIRD-PARTY.txt"

"$python" -I tools/package_check.py "$stage" "dist/$name.zip"
rm -rf -- "dist/$name"
mv "$stage" "dist/$name"
rm -rf -- dist/stage
ls -la "dist/$name.zip"
