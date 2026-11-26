#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""The Last of Us Remastered (CUSA00552 1.11): the game's render settings, listed and overridden.

The settings are the struct `render-settings` (0x1110 bytes) of the game's DC data: the default
(*default-render-settings* in bin/dc1/render-settings.bin) and one per level area and cutscene
(bin/dc1/render-settings/...). The game copies the one in use into its current settings
(0x1A88070) and animates its working copies toward it. Overrides written here (by name) go to
tlou-user/render-settings.txt of the data folder (analysis/, or GOW3_DATA_DIR); tools/
probe_tlou.py passes them to the runtime (src/runtime_render_settings.c), which writes them into
every render-settings struct of each DC module the game loads, so they hold in every area and
cutscene. The launcher's "Lighter effects" (TLOU_LIGHT_EFFECTS=1) is a set of these, applied
before the player's own.

  python tools/tlou_render_settings.py list [WORD...]       settings, with defaults and ranges
  python tools/tlou_render_settings.py set NAME=VALUE...    override (bool: 0/1)
  python tools/tlou_render_settings.py unset NAME... | clear | show

Where the names come from (reverse engineered, 10 Nov 2026): render-settings.bin also holds the
game's development menu for these settings, 18 struct types, each a list of fields (name, type,
offset in its struct, range). Which struct sits where in render-settings comes from the code that
applies a struct (0xBCE930: one block per struct, each at a constant offset), in LAYOUT below. The
DC string ids are CRC-32 (MSB first, polynomial 0x04C11DB7, no initial value or final xor).
"""
import argparse
import hashlib
import lzma
import os
import struct
import sys
import zlib
from pathlib import Path

import gow3_data   # tools/gow3_data.py: the player data's root folder (GOW3_DATA_DIR)

ROOT = Path(__file__).resolve().parent.parent
USER = gow3_data.data_root(ROOT) / "tlou-user"
SIZE = 0x1110
RENDER_SETTINGS = 0x9EC73B22                 # sid32("render-settings"), the struct's type
FLOAT, BOOL, INT, ENUM, STRUCT = 0x64797872, 0xECDDEB34, 0x5616BE0A, 0x5F6F8E12, 0xA58BAECB
TYPE_LIST = 0x45713B76                       # render-settings.bin's list of the 18 struct types
# The apply function's blocks (0xBCE930..0xBCF4C6): (type list index, offset in render-settings,
# name used here). The names are this tool's; the game's menu titles are not in the executable.
LAYOUT = [
    (0, 0x000, "lighting"), (1, 0x320, "shadows"), (2, 0x480, "fog"), (3, 0x550, "wind"),
    (4, 0x600, "sun"), (5, 0x6A0, "sky"), (2, 0x7E0, "fog-2"), (6, 0x8B0, "particles"),
    (7, 0xAD0, "tonemap"), (8, 0xB30, "lut"), (9, 0xBE0, "post"), (10, 0xE60, "rain"),
    (11, 0xF20, "rain-2"), (12, 0xFA0, "character"), (13, 0x1000, "ocean"), (14, 0x1030, "misc"),
]
# Struct fields inside those (type id -> type list index): the lights of "lighting" (0x0, 0x50,
# 0xA0), the per-effect tints of "particles" (0x8F0..0xAB0) and the cloud layers of "sky".
NESTED = {0x462B3C5F: 15, 0x8F594893: 16, 0x6AA979F9: 17}
KIND = {FLOAT: "f", BOOL: "b", INT: "i", ENUM: "i"}


def sid32(text):
    h = 0
    for c in text.encode():
        h ^= c << 24
        for _ in range(8):
            h = ((h << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if h & 0x80000000 else (h << 1) & 0xFFFFFFFF
    return h


# --- PSARC (big-endian table of contents; names in entry 0; entries found by the MD5 of the name:
# the update's patch.psarc does not keep them in the order of the names) ---
def psarc_read(path, wanted):
    with open(path, "rb") as f:
        magic, _, _, comp, toc_len, esz, count, block, _ = struct.unpack(">4sHH4sIIIII", f.read(32))
        if magic != b"PSAR":
            raise ValueError(f"{path}: not a PSARC")
        toc = f.read(toc_len - 32)
        entries = []
        for i in range(count):
            e = toc[i * esz:(i + 1) * esz]
            entries.append((e[:16], struct.unpack(">I", e[16:20])[0],
                            int.from_bytes(e[20:25], "big"), int.from_bytes(e[25:30], "big")))
        width = 2 if block <= 0x10000 else 3 if block <= 0x1000000 else 4
        raw = toc[count * esz:]
        sizes = [int.from_bytes(raw[i:i + width], "big") for i in range(0, len(raw) - width + 1, width)]

        def read(index):
            _, first, length, offset = entries[index]
            out = bytearray()
            f.seek(offset)
            while len(out) < length:
                stored = sizes[first]
                first += 1
                chunk = f.read(stored or block)
                want = min(block, length - len(out))
                if len(chunk) != want:
                    chunk = zlib.decompress(chunk) if comp == b"zlib" else lzma.decompress(chunk)
                out += chunk
            return bytes(out)

        names = read(0).replace(b"\0", b"\n").decode("utf-8", "replace").split("\n")
        by_md5 = {e[0]: i for i, e in enumerate(entries)}
        for name in names:
            if name.lstrip("/") == wanted:
                index = by_md5.get(hashlib.md5(name.encode()).digest())
                if index is not None:
                    return read(index)
    return None


def render_settings_module(game, update):
    """bin/dc1/render-settings.bin of the update (patch.psarc), else of the game (bin.psarc)."""
    places = []
    if update:
        places.append((update / "patch.psarc", "build/ps4/main/bin/dc1/render-settings.bin"))
    places.append((game / "build/ps4/main/bin.psarc", "dc1/render-settings.bin"))
    for archive, name in places:
        if archive.is_file():
            data = psarc_read(archive, name)
            if data:
                return data, f"{archive.name}:{name}"
    raise SystemExit("render-settings.bin not found (game folder and update: --game, TLOU_UPDATE)")


# --- DC modules: "00CD", version, text size, ..., entry count, entries {sid name, sid type, ptr};
# after the text, a byte count and a bitmap with one bit per 8-byte word that holds a pointer. ---
class Dc:
    def __init__(self, data):
        self.d = data
        magic, _, self.text, _, _, count, entries, _ = struct.unpack_from("<IIIIIIII", data, 0)
        if magic != 0x44433030:
            raise ValueError("not a DC module")
        self.entries = [struct.unpack_from("<IIQ", data, entries + 16 * i) for i in range(count)]

    def u32(self, at):
        return struct.unpack_from("<I", self.d, at)[0]

    def q(self, at):
        return struct.unpack_from("<Q", self.d, at)[0]

    def cstr(self, at):
        end = self.d.index(b"\0", at)
        return self.d[at:end].decode("ascii")

    def entry(self, name):
        return next((ptr for n, _, ptr in self.entries if n == name), None)


def fields(dc):
    """[(name, kind, offset, default, low, high)] for every setting LAYOUT places."""
    types = dc.entry(TYPE_LIST)
    default = dc.entry(sid32("*default-render-settings*"))
    if types is None or default is None:
        raise SystemExit("render-settings.bin: unexpected layout (not version 1.11's)")
    groups = [dc.q(dc.q(types + 8 * i) + 8) for i in range(18)]

    def members(index):
        group = groups[index]
        items, count = dc.q(group), dc.q(group + 8)
        return [dc.q(items + 8 * i) for i in range(count)]

    out = []

    def walk(index, base, prefix):
        for item in members(index):
            tag = dc.u32(item - 4)
            name = prefix + dc.cstr(dc.q(item))
            offset = base + dc.u32(item + 8)
            if tag == STRUCT:
                nested = NESTED.get(dc.u32(item + 20))
                if nested is not None:
                    walk(nested, offset, name + ".")
                continue
            kind = KIND.get(tag)
            if kind is None or offset >= SIZE:
                continue
            low = high = None
            if kind == "f":
                low, high = struct.unpack_from("<2f", dc.d, item + 24)
                value = struct.unpack_from("<f", dc.d, default + offset)[0]
            elif kind == "b":
                value = dc.d[default + offset]
            else:
                value = struct.unpack_from("<i", dc.d, default + offset)[0]
            out.append((name, kind, offset, value, low, high))

    first = [dc.cstr(dc.q(members(i)[0])) for i in (0, 7, 14)]
    if first != ["sun-light", "tonemap-exposure-method", "player-damage-fx-index"]:
        raise SystemExit(f"render-settings.bin: unexpected struct types {first} (not version 1.11's)")
    for index, base, label in LAYOUT:
        walk(index, base, label + ".")
    return out


def find_update(game):
    chosen = os.environ.get("TLOU_UPDATE", "")
    if chosen == "0":
        return None
    for candidate in [Path(chosen)] if chosen else [game.parent / (game.name + s) for s in ("-patch", "-update")]:
        if (candidate / "patch.psarc").is_file():
            return candidate
    return None


def load_overrides(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("#")[0].split()
            if len(parts) == 4:
                out[parts[3]] = (int(parts[0], 16), parts[1], parts[2])
    return out


def save_overrides(path, overrides):
    lines = ["# The Last of Us Remastered render settings overrides: tools/tlou_render_settings.py writes",
             "# this file, the runtime reads it (offset in render-settings, f/i/b, value, name)."]
    for name, (offset, kind, value) in sorted(overrides.items(), key=lambda kv: kv[1][0]):
        lines.append(f"{offset:#06x} {kind} {value} {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def show_value(kind, value):
    return f"{value:.6g}" if kind == "f" else str(value)


def default_game():
    """The game folder chosen in the launcher (tlou-user/launcher.json), else the one
    next to this folder."""
    import json
    try:
        chosen = json.loads((USER / "launcher.json").read_text(encoding="utf-8"))["folders"]["game"]
        if chosen:
            return Path(chosen)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return ROOT.parent / "CUSA00552"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", type=Path, default=default_game())
    parser.add_argument("--file", type=Path, default=USER / "render-settings.txt",
                        help=f"the overrides file (default: {USER / 'render-settings.txt'})")
    parser.add_argument("command", choices=("list", "set", "unset", "clear", "show"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    overrides = load_overrides(args.file)
    if args.command in ("clear", "show"):
        if args.command == "clear":
            save_overrides(args.file, {})
            print(f"No overrides ({args.file})")
        for name, (_, kind, value) in sorted(overrides.items()) if args.command == "show" else ():
            print(f"{name} = {value}")
        return 0
    if args.command == "unset":
        for name in args.args:
            if overrides.pop(name, None) is None:
                print(f"{name}: not overridden")
        save_overrides(args.file, overrides)
        return 0
    game = args.game.resolve()
    data, source = render_settings_module(game, find_update(game))
    table = fields(Dc(data))
    by_name = {f[0]: f for f in table}
    if args.command == "list":
        words = [w.lower() for w in args.args]
        print(f"# {len(table)} settings from {source}; * = overridden")
        for name, kind, offset, value, low, high in table:
            if words and not all(w in name for w in words):
                continue
            mark = "*" if name in overrides else " "
            limits = f"  [{low:g}, {high:g}]" if low is not None else ""
            now = f"  -> {overrides[name][2]}" if name in overrides else ""
            print(f"{mark} {name:70s} {kind} {offset:#06x} {show_value(kind, value):>12s}{limits}{now}")
        return 0
    # set NAME=VALUE...
    for pair in args.args:
        name, _, text = pair.partition("=")
        field = by_name.get(name)
        if not field or not text:
            close = [n for n in by_name if name.split(".")[-1] in n][:5]
            print(f"{name}: unknown setting" + (f" (similar: {', '.join(close)})" if close else ""))
            return 1
        _, kind, offset, _, low, high = field
        try:
            value = float(text) if kind == "f" else int(text, 0)
        except ValueError:
            print(f"{name}: '{text}' is not a {'number' if kind == 'f' else 'whole number'}")
            return 1
        if kind == "b" and value not in (0, 1):
            print(f"{name}: a switch takes 0 or 1")
            return 1
        if low is not None and not low <= value <= high:
            print(f"{name}: {value:g} is outside the game's menu range [{low:g}, {high:g}] (set anyway)")
        overrides[name] = (offset, kind, f"{value:.9g}" if kind == "f" else str(value))
        print(f"{name} = {overrides[name][2]}")
    save_overrides(args.file, overrides)
    print(f"Saved {args.file}: applied the next time the game starts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
