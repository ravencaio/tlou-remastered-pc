#!/usr/bin/env bash
# Builds the starters TLOU Launcher.exe (tlou-launcher.c) and TLOU Game.exe (tlou-game.c): stripped,
# with version information and a manifest (tlou-starter.rc). Small unsigned programs that start a
# hidden interpreter without them look like malware loaders to antivirus heuristics (issues #88,
# #112, #148). Needs clang and windres (MSYS2 CLANG64).
# usage: build-starters.sh <TLOU Launcher.exe path> <TLOU Game.exe path> [--icon]
#   --icon  gives TLOU Launcher.exe launcher/assets/icon.ico when it is there (not in the repository)
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
launcher_out=$1
game_out=$2
icon=${3:-}
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

version=$(tr -d '\r\n' < "$here/../tools/VERSION")
IFS=. read -r major minor patch <<< "$version"
patch=${patch%%[^0-9]*}
printf '#define TLOU_VERSION %d,%d,%d,0\n#define TLOU_VERSION_STR "%s"\n' \
    "$major" "$minor" "${patch:-0}" "$version" > "$work/tlou-version.h"

build() {  # <source> <output> <windres options...>
    local source=$1 output=$2
    shift 2
    windres -I "$here" -I "$work" "$@" "$here/tlou-starter.rc" -O coff -o "$work/resources.o"
    clang -O2 -s -Wall -Wextra -Werror -static -municode -mwindows "$here/$source" \
        "$work/resources.o" -o "$output"
}

launcher_options=(-DTLOU_STARTER_LAUNCHER)
if [[ $icon == --icon && -f $here/assets/icon.ico ]]; then
    launcher_options+=(-DTLOU_STARTER_ICON)
fi
build tlou-launcher.c "$launcher_out" "${launcher_options[@]}"
build tlou-game.c "$game_out" -DTLOU_STARTER_GAME
