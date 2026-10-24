"""Regression checks for real VisDA list layouts without loading SDXL."""

import ast
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


def load_visda_scanner():
    source = Path(__file__).resolve().parents[1] / "diffusion" / "train_synthesis_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    scanner = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_extract_paths_and_captions")
    namespace = {"os": os, "logging": logging, "__name__": __name__}
    exec(compile(ast.Module(body=[scanner], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[scanner.name]


class VisDAListPathsTest(unittest.TestCase):
    def test_standard_split_local_and_absolute_lists(self):
        scan = load_visda_scanner()
        with tempfile.TemporaryDirectory() as folder:
            dataset_root = Path(folder)
            split_root = dataset_root / "train"
            image = split_root / "car" / "sample.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"path fixture; image pixels are not read")
            cases = (
                (dataset_root, "train/car/sample.png"),
                (split_root, "car/sample.png"),
                (split_root, str(image)),
                (dataset_root, "sample.png"),
            )
            for image_root, entry in cases:
                with self.subTest(image_root=image_root, entry=entry):
                    dataset = SimpleNamespace(root=str(image_root), dataset_root=str(dataset_root),
                                              classes=["car"], data_list=[(entry, 0)])
                    paths, captions = scan(dataset, "train")
                    self.assertEqual([Path(path) for path in paths], [image])
                    self.assertEqual(captions, ["a car"])


if __name__ == "__main__":
    unittest.main()
