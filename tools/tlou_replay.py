#!/usr/bin/env python3
"""Replays a recorded route on a test profile and reports whether the Vulkan device was lost.

The player's sessions record their controller input (analysis/tlou/pad-record-*.txt, time 0 at
the first press, see tools/probe_tlou.py). This tool starts the game on a test profile whose
saves are a copy of the player's (analysis/tlou-test-replay, restored from its pristine copy
before each run, so the game's autosaves never move the starting point), waits --menu-wait
seconds for the title menu, starts the replay (the token "replay" in the pad file) and watches the
log for --play seconds. The shader cache and the settings stay the player's (probe_tlou.py keeps
them in analysis/tlou-user), and the environment follows the launcher's saved switches and
Startup choices (analysis/tlou-user/launcher.json), so a run matches what the player runs.

  python tools/tlou_replay.py --label base --record pad-record-20261108-001709.txt
  python tools/tlou_replay.py --label test GOW3_GPU_CHECKPOINTS=4 --skip e44070db
  python tools/tlou_replay.py --label pics --shots 0:120:3      frames every 3 s of the replay
  python tools/tlou_replay.py --snapshot                        copy the player's saves again

Each run's log is kept as analysis/tlou/replay_<label>.log. The game is stopped by its own PID.
Written on 10 Nov 2026 to bisect a device loss in Pittsburgh: the replay reproduced it on most
runs, with GPU checkpoints (GOW3_GPU_CHECKPOINTS=4) nearly every time.
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
OUT = ROOT / "analysis/tlou"
USER = ROOT / "analysis/tlou-user"
LOG = OUT / "boot.log"
PAD = USER / "probe-pad.txt"
SKIP = USER / "skip.txt"
PID = OUT / "probe.pid"
PROFILE = ROOT / "analysis/tlou-test-replay"
PRISTINE = ROOT / "analysis/tlou-test-replay-pristine"


def launcher_env():
    """The environment the launcher gives the game (launcher/tlou_launcher.pyw, game_env)."""
    switches = {"motion_blur": False, "job_sleep": True, "pro_shadows": False,
                "background_shaders": True}
    overrides = {}
    try:
        saved = json.loads((USER / "launcher.json").read_text(encoding="utf-8"))
        switches.update({k: v for k, v in saved.items() if k in switches})
        overrides = saved.get("overrides", {})
    except (OSError, ValueError):
        pass
    env = dict(overrides)
    env.setdefault("GOW3_NATIVE_TSC", "1")
    env.setdefault("GOW3_GUEST_PRIORITY", "1")
    env["TLOU_JOB_SLEEP"] = "1" if switches["job_sleep"] else "0"
    env["TLOU_PRO_SHADOWS"] = "1" if switches["pro_shadows"] else "0"
    if switches["background_shaders"]:
        env["GOW3_ASYNC_SHADERS"] = "1"
        env["GOW3_ASYNC_SHADERS_LOCKED"] = "0"
    env["TLOU_RECORD_PAD"] = "0"
    return env


def snapshot():
    """The pristine test profile: a copy of the player's saves."""
    if PRISTINE.exists():
        shutil.rmtree(PRISTINE)
    PRISTINE.mkdir(parents=True)
    for name in ("savedata", "download0"):
        if (USER / name).exists():
            shutil.copytree(USER / name, PRISTINE / name)
    for name in ("data", "temp0"):
        (PRISTINE / name).mkdir(exist_ok=True)
    print(f"Test profile saves copied from {USER} to {PRISTINE}")


def log_text():
    try:
        return LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run")
    parser.add_argument("--record", default="", help="recording in analysis/tlou (default: newest)")
    parser.add_argument("--menu-wait", type=float, default=60,
                        help="seconds from the game's start to the title menu's first press")
    parser.add_argument("--play", type=float, default=240, help="seconds after the replay starts")
    parser.add_argument("--skip", default="", help="shader hashes to skip (hex, comma separated)")
    parser.add_argument("--shots", default="",
                        help="save the frame at these seconds of the replay (start:end:step)")
    parser.add_argument("--snapshot", action="store_true",
                        help="copy the player's saves into the pristine test profile and exit")
    parser.add_argument("--pristine", default="",
                        help="another pristine profile folder under analysis/ (default tlou-test-replay-pristine)")
    parser.add_argument("vars", nargs="*", help="NAME=value for the game's environment")
    args = parser.parse_args()

    global PRISTINE
    if args.pristine:
        PRISTINE = ROOT / "analysis" / args.pristine
    if args.snapshot or not PRISTINE.exists():
        snapshot()
        if args.snapshot:
            return 0
    records = sorted(OUT.glob("pad-record-*.txt"))
    record = OUT / args.record if args.record else (records[-1] if records else None)
    if not record or not record.exists():
        print("No recording (analysis/tlou/pad-record-*.txt): play a session first")
        return 2

    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    shutil.copytree(PRISTINE, PROFILE)
    env = dict(os.environ)
    env.update(launcher_env())
    env["GOW3_PAD_REPLAY"] = str(record)
    for item in args.vars:
        name, _, value = item.partition("=")
        env[name] = value

    PID.unlink(missing_ok=True)
    captures_root = OUT / "captures"
    before = set(captures_root.iterdir()) if captures_root.exists() else set()
    extra = ["--capture", "9999"] if args.shots else []
    with open(OUT / f"replay_{args.label}.probe.txt", "w", encoding="utf-8") as out:
        probe = subprocess.Popen([sys.executable, str(ROOT / "tools/probe_tlou.py"), "--boot",
                                  "--seconds", str(int(args.menu_wait + args.play + 120)),
                                  "--saves", str(PROFILE)] + extra,
                                 env=env, stdout=out, stderr=subprocess.STDOUT)
        game_pid = ""
        for _ in range(900):
            if PID.exists() and (game_pid := PID.read_text().strip()):
                break
            if probe.poll() is not None:
                break
            time.sleep(0.2)
        if not game_pid:
            print(f"{args.label}: the game did not start")
            return 2
        time.sleep(args.menu_wait)
        if args.skip:
            SKIP.write_text(args.skip.replace(",", " "), encoding="ascii")
        PAD.write_text("replay", encoding="ascii")
        started = time.monotonic()

        shots, taken, capture = [], [], None
        if args.shots:
            first, last, step = (float(x) for x in args.shots.split(":"))
            while first <= last:
                shots.append(first)
                first += step
            new = set(captures_root.iterdir()) - before
            capture = max(new, key=lambda d: d.stat().st_mtime) if new else None
        result, when = "no device loss", None
        while time.monotonic() - started < args.play:
            if shots:
                due = shots[0] - (time.monotonic() - started)
                if due <= 0:
                    if capture:
                        (capture / "trigger").touch()
                        taken.append(shots[0])
                    shots.pop(0)
                    continue
                time.sleep(min(due, 2))
            else:
                time.sleep(2)
            text = log_text()
            if "Device lost" in text or "STOP:" in text:
                result = "device lost" if "Device lost" in text else "stopped"
                when = time.monotonic() - started
                break
            if probe.poll() is not None:
                result, when = f"the probe ended ({probe.returncode})", time.monotonic() - started
                break
        subprocess.run(["taskkill", "/PID", game_pid, "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        try:
            probe.wait(timeout=30)
        except subprocess.TimeoutExpired:
            probe.kill()

    text = log_text()
    shutil.copyfile(LOG, OUT / f"replay_{args.label}.log")
    fps = re.findall(r"Frame stats: ([0-9.]+) FPS", text)
    worlds = re.findall(r"Switching world: from \S* to (\S+)", text)
    print(f"== {args.label}: {result}" + (f" {when:.0f} s into the replay" if when else "")
          + f" ({record.name}; worlds {worlds})")
    print("FPS: " + " ".join(fps))
    for line in text.splitlines():
        if "GPU checkpoint:" in line:
            print(line[:200])
    if capture:
        (capture / "shots.txt").write_text(" ".join(f"{t:g}" for t in taken), encoding="ascii")
        print(f"Frames: {capture} ({len(taken)} shots; tools/tlou_compare_shots.py compares two)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
