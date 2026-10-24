"""Lightweight checks for experiment coverage and synthesis command wiring."""

import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generation_grid", ROOT / "scripts" / "generate_grid.py")
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)


class GenerationGridTests(unittest.TestCase):
    def test_default_grid_covers_exact_cartesian_product(self):
        variants = grid.build_variants(Path("outputs"))
        ddim = [v for v in variants if v["pipeline"] == "ddim_inversion"]
        classes = [v for v in variants if v["pipeline"] == "class_condition"]
        self.assertEqual(len(ddim), 50)
        self.assertEqual(len(classes), 10)
        self.assertEqual(len({v["image_root"] for v in variants}), 60)
        self.assertEqual({(v["token_strength"], v["noise_level"]) for v in ddim},
                         {(s / 10, n / 10) for s in range(6, 16) for n in range(6, 11)})
        self.assertEqual({v["token_strength"] for v in classes}, {s / 10 for s in range(6, 16)})

    def test_commands_wire_distinct_strength_and_noise_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            image = source / "Alarm_Clock" / "example.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            args = grid.build_parser().parse_args([
                "--source_root", str(source), "--checkpoint_dir", str(root / "checkpoint"),
                "--output_root", str(root / "output"), "--reference_seeds", "2",
            ])
            images = grid.source_classes(source)
            variants = grid.build_variants(root / "output")
            ddim = variants[0]
            commands = list(grid.commands_for_variant(args, ddim, images))
            self.assertEqual(len(commands), 2)
            command = commands[0]
            self.assertIn("--no_prompt", command)
            self.assertIn("--skip_visualizations", command)
            self.assertEqual(command[command.index("--concept_token_strength") + 1], "0.6")
            self.assertEqual(command[command.index("--invert_noise_scale") + 1], "0.6")
            self.assertEqual(command[command.index("--concept_image") + 1], str(image))
            self.assertEqual(Path(command[command.index("--output_image") + 1]).parent.name, "Alarm_Clock")
            class_variant = next(v for v in variants if v["pipeline"] == "class_condition")
            class_command, = grid.commands_for_variant(args, class_variant, images)
            self.assertEqual(class_command[class_command.index("--concept_token_strength") + 1], "0.6")
            self.assertNotIn("--invert_noise_scale", class_command)

    def test_reference_names_do_not_collide_across_extensions_or_nested_dirs(self):
        source = Path("source")
        output = Path("output")
        names = [grid.reference_output_path(output, source, source / relative, 0)
                 for relative in ("class/a.jpg", "class/a.png", "class/nested/a.jpg")]
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(path.parent == output / "class" for path in names))

    def test_dry_run_plans_without_inference_and_preserves_completed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "a_class"
            source.mkdir(parents=True)
            (source / "input.jpg").touch()
            output = root / "output"
            argv = ["--source_root", str(source.parent), "--checkpoint_dir", str(root / "missing_checkpoint"),
                    "--output_root", str(output), "--dry_run"]
            with patch.object(grid.subprocess, "run") as run:
                self.assertEqual(grid.main(argv), 0)
                run.assert_not_called()
            manifest_path = output / "generation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(len(manifest["variants"]), 60)
            self.assertEqual(sum(v["expected_images"] for v in manifest["variants"]), 550)
            # Repeating an identical dry run is safe; completed provenance is protected.
            self.assertEqual(grid.main(argv), 0)
            manifest["status"] = "complete"
            grid.write_manifest(manifest_path, manifest)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    grid.main(argv)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "complete")


if __name__ == "__main__":
    unittest.main()
