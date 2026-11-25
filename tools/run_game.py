#!/usr/bin/env python3
"""Start the game with the settings the launcher saved, without opening the launcher.

The launcher keeps its switches (motion blur, job sleep, PS4 Pro shadows, background shaders,
profiler, folders) in analysis/tlou-user/launcher.json. This reads that file with the launcher's own
code (launcher/tlou_launcher.pyw, whose main() only runs as a script) and builds the same
environment and command line its Play button uses, then runs tools/probe_tlou.py with them.

Usage: run_game.py [--print] [--seconds N] [probe_tlou.py arguments...]
  --print     show the environment and command line instead of running the game
  --seconds   time limit for the game (0: none), as the launcher's --seconds

TLOU Game.exe (launcher/tlou-game.c) runs this file with pythonw, so no console window appears.
"""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "launcher/tlou_launcher.pyw"


def launcher_module():
    """tlou_launcher.pyw as a module: its settings readers and game_env/game_command."""
    if not LAUNCHER.is_file():
        raise SystemExit(f"Launcher not found: {LAUNCHER}")
    loader = importlib.machinery.SourceFileLoader("tlou_launcher", str(LAUNCHER))
    spec = importlib.util.spec_from_loader("tlou_launcher", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def notify(message, title="The Last of Us Remastered"):
    """pythonw (TLOU Game.exe) has no console: tell the player with a message box."""
    if sys.stderr is not None:
        print(message, file=sys.stderr)
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, str(message), title, 0x10)
    except Exception:
        pass


def game_running():
    """The launcher refuses a second run; so does this one: two would share the same files."""
    if sys.platform == "win32":
        # Bytes, not text: tasklist writes in the console code page, which need not decode. stdin from
        # DEVNULL: pythonw (TLOU Game.exe) has no stdin handle to give the child. stdout can still come
        # back None under pythonw, which counts as "not running" rather than stopping the game.
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq gow3-probe.exe", "/NH"],
                                 stdin=subprocess.DEVNULL, capture_output=True,
                                 creationflags=0x08000000)
        except (OSError, ValueError):
            return False
        return b"gow3-probe.exe" in (out.stdout or b"").lower()
    # POSIX: creationflags is Windows-only (it raises ValueError here), and tasklist does not
    # exist. Ask the same question as the launcher's game_running(): a running gow3-probe. A
    # missing pgrep (a stripped-down system) leaves the game startable rather than blocking it.
    try:
        return subprocess.run(["pgrep", "-f", "gow3-probe"], capture_output=True).returncode == 0
    except OSError:
        return False


def python_exe():
    """python.exe beside this interpreter: pythonw has no output for the run's log."""
    candidate = Path(sys.executable).with_name("python.exe")
    return str(candidate if candidate.is_file() else sys.executable)


def open_run_log(launcher):
    """The run log to write the game's output to; a log that cannot be written must not stop the
    game (a locked or read-only file, a restricted folder), so the next place is tried."""
    import tempfile
    for path in (launcher.RUN_LOG, ROOT / "analysis/tlou/game-run.log",
                 Path(tempfile.gettempdir()) / "tlou-run.log"):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path, path.open("w", encoding="utf-8")
        except OSError:
            continue
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print", action="store_true",
                        help="show the environment and command line instead of running the game")
    parser.add_argument("--seconds", type=int, default=0, help="time limit for the game (0: none)")
    args, extra = parser.parse_known_args()

    launcher = launcher_module()
    switches = launcher.read_switches()
    overrides = launcher.read_overrides()
    folders = launcher.read_folders()
    env = launcher.game_env(switches, overrides=overrides, folders=folders)
    command = launcher.game_command(python_exe(), folders, args.seconds) + extra

    if args.print:
        shown = {k: v for k, v in sorted(env.items()) if k.startswith(("GOW3_", "TLOU_"))}
        print(json.dumps({"command": command, "environment": shown, "folders": folders,
                          "switches": switches}, indent=2))
        return 0

    if switches["profile"]:
        Path(launcher.PROFILE_ENV["GOW3_DUMP_DIR"]).mkdir(parents=True, exist_ok=True)
    if game_running():
        notify("The game is already running. Close it before starting another one.")
        return 2
    log, output = open_run_log(launcher)
    if output is None:
        return subprocess.call(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
    with output:
        if sys.stderr is not None:   # pythonw (TLOU Game.exe) has no console
            print(f"Running: {' '.join(command)}", file=sys.stderr)
            print(f"Log: {log}", file=sys.stderr)
        return subprocess.call(command, cwd=ROOT, env=env, stdout=output,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)



def report_failure(error):
    """pythonw has no console: leave a reason where the launcher's run log is."""
    import traceback
    path = ROOT / "analysis/tlou/launcher-run.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write("run_game.py stopped:\n")
            traceback.print_exc(file=output)
    except OSError:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as error:
        report_failure(error)
        raise
