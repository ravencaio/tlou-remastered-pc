"""tools/run_game.py's second-instance check, on both platforms and without a game."""
import importlib.machinery
import importlib.util
from pathlib import Path
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tlou_run_game", REPO / "tools/run_game.py")
run_game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_game)


class GameRunningTest(unittest.TestCase):
    def check(self, platform, result=None, error=None):
        fake_sys = types.SimpleNamespace(platform=platform)
        with mock.patch.object(run_game, "sys", fake_sys), \
             mock.patch.object(run_game.subprocess, "run", return_value=result, side_effect=error) as call:
            running = run_game.game_running()
        return running, call

    @staticmethod
    def result(stdout=b"", returncode=0):
        return types.SimpleNamespace(stdout=stdout, returncode=returncode)

    def test_windows_finds_the_probe(self):
        running, call = self.check("win32", self.result(b"gow3-probe.exe  1234 Console  1  1,000 K"))
        self.assertTrue(running)
        self.assertEqual(call.call_args.args[0][0], "tasklist")
        self.assertEqual(call.call_args.kwargs["creationflags"], 0x08000000)

    def test_windows_no_match(self):
        running, _ = self.check("win32", self.result(b"INFO: No tasks are running which match the specified criteria."))
        self.assertFalse(running)

    def test_posix_finds_the_probe_without_windows_flags(self):
        running, call = self.check("linux", self.result(b"1234\n"))
        self.assertTrue(running)
        self.assertEqual(call.call_args.args[0], ["pgrep", "-f", "gow3-probe"])
        self.assertNotIn("creationflags", call.call_args.kwargs)

    def test_posix_no_match(self):
        running, _ = self.check("linux", self.result(returncode=1))
        self.assertFalse(running)

    def test_posix_without_pgrep_starts_the_game(self):
        running, _ = self.check("linux", error=OSError("pgrep not found"))
        self.assertFalse(running)


if __name__ == "__main__":
    unittest.main()
