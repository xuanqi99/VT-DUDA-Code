"""Execute the actual VisDA sweep/lookup control flow with model calls stubbed."""

import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "UDA/ELS/train_els_visda.py"


class AccuracyArray(list):
    def argmax(self):
        return max(range(len(self)), key=self.__getitem__)


class VisDAEpsilonTests(unittest.TestCase):
    def sweep_namespace(self, args, main):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body if isinstance(node, ast.If) and
                     isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and
                     node.test.left.id == "__name__")
        naming = next(node for node in entry.body if isinstance(node, ast.FunctionDef) and node.name == "_eps_dir_name")
        sweep = next(node for node in entry.body if isinstance(node, ast.If) and
                     ast.unparse(node.test) == "args.eps >= 1")
        default_values = next(node for node in tree.body if isinstance(node, ast.Assign) and
                              any(isinstance(target, ast.Name) and target.id == "EPS_SWEEP_DEFAULT" for target in node.targets))
        namespace = {"args": args, "main": main, "np": SimpleNamespace(array=AccuracyArray),
                     "osp": os.path, "os": os, "exit": lambda: None}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[default_values, naming, sweep], type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace

    def test_sweep_separates_checkpoints_and_records_actual_epsilon(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eps=1.0, log=temporary, phase="train")
            calls = []

            def fake_main(options, eps):
                calls.append((options.eps, eps, options.log))
                # Mirror the native checkpoint naming, so overwrite detection is meaningful.
                Path(options.log, "latest.pth").touch()
                return eps

            namespace = self.sweep_namespace(args, fake_main)
            expected = namespace["EPS_SWEEP_DEFAULT"]
            self.assertEqual([call[0] for call in calls], expected)
            self.assertEqual([call[1] for call in calls], expected)
            self.assertEqual(len({call[2] for call in calls}), 9)
            self.assertEqual(len(list(Path(temporary).rglob("latest.pth"))), 9)
            self.assertTrue(all(Path(call[2]).name.startswith("eps") for call in calls))
            self.assertEqual((args.eps, args.log), (1.0, temporary))

    def test_selected_epsilon_test_keeps_exact_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eps=0.1, log=temporary, phase="test")
            calls = []
            self.sweep_namespace(args, lambda options, eps: calls.append((options.eps, eps, options.log, options.phase)))
            self.assertEqual(calls, [(0.1, 0.1, temporary, "test")])
            self.assertFalse(list(Path(temporary).iterdir()))

    def test_test_phase_prefers_epsilon_checkpoint(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loading = next(node for node in main.body if isinstance(node, ast.If) and
                       ast.unparse(node.test) == "args.phase != 'train'")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "best_with_eps0.1.pth").touch()
            (root / "best.pth").touch()
            loaded = []
            namespace = {"args": SimpleNamespace(eps=0.1, phase="test"), "eps": 0.1,
                         "logger": SimpleNamespace(get_checkpoint_path=lambda name: str(root / f"{name}.pth")),
                         "osp": os.path, "classifier": object(),
                         "_load_checkpoint_into_model": lambda model, path, **kwargs: loaded.append(path)}
            exec(compile(ast.Module(body=[loading], type_ignores=[]), str(SOURCE), "exec"), namespace)
            self.assertEqual(loaded, [str(root / "best_with_eps0.1.pth")])


if __name__ == "__main__":
    unittest.main()
