import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_grid import build_variants
from scripts.train_uda_grid import build_parser, generated_samples, make_plan
from UDA.generated_data import dataset_classes, read_image_list


class UDAGridTests(unittest.TestCase):
    def options(self, root, *extra):
        return build_parser().parse_args([
            "--generation-manifest", str(root / "generation_manifest.json"),
            "--dataset", "OfficeHome", "--target-list", str(root / "target.txt"),
            "--target-image-root", str(root / "real images"), "--output-root", str(root / "uda"), *extra])

    def test_full_cartesian_grid_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema_version": 1, "variants": build_variants(root / "images")}
            plan, _, _, _ = make_plan(self.options(root, "--methods", "els", "mcc", "--seeds", "0", "1"), manifest)
            self.assertEqual(plan["combination_count"], 500)
            self.assertEqual(plan["run_count"], 2000)
            pairs = {(run["ddim_variant"], run["class_variant"]) for run in plan["runs"]}
            self.assertEqual(len(pairs), 500)
            self.assertEqual(len({run["log_dir"] for run in plan["runs"]}), 2000)
            for run in plan["runs"]:
                self.assertTrue(Path(run["command"][1]).is_file())
                self.assertEqual(len(run["source_lists"]), 2)
                self.assertNotIn("real_target.txt", " ".join(run["source_lists"]))

    def test_subsets_and_identity_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema_version": 1, "variants": build_variants(root / "images")}
            args = self.options(root, "--ddim-token-strengths", "0.6", "1.5", "--ddim-noise-levels", "1.0",
                                "--class-token-strengths", "0.8", "--train-args", "--epochs", "1")
            plan, _, _, _ = make_plan(args, manifest)
            self.assertEqual(plan["run_count"], 2)
            self.assertTrue(all(run["ddim_noise_level"] == 1.0 for run in plan["runs"]))
            args.train_args = ["--source-list=wrong.txt"]
            with self.assertRaisesRegex(ValueError, "cannot override"):
                make_plan(args, manifest)

    def test_label_mapping_preserves_supplied_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Exit Sign", "Drill"):
                folder = root / name
                folder.mkdir()
                (folder / "sample with spaces.png").write_bytes(b"test")
            samples = generated_samples(root, ["Drill", "Exit_Sign"])
            self.assertEqual({Path(path).parent.name: label for path, label in samples}, {"Drill": 0, "Exit Sign": 1})
            with self.assertRaisesRegex(ValueError, "Missing generated classes"):
                generated_samples(root, ["Drill", "Exit_Sign", "Bottle"])

    def test_image_list_validation_and_classes(self):
        self.assertEqual({name: len(dataset_classes(name)) for name in ("OfficeHome", "Office31", "VisDA2017")},
                         {"OfficeHome": 65, "Office31": 31, "VisDA2017": 12})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image with spaces.png").touch()
            listing = root / "target.txt"
            listing.write_text("# comment\n\nimage with spaces.png 1\n", encoding="utf-8")
            self.assertEqual(read_image_list(listing, root, 2), [(str((root / "image with spaces.png").resolve()), 1)])
            listing.write_text("image with spaces.png 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                read_image_list(listing, root, 2)

    def test_plan_cli_does_not_require_images_or_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema_version": 1, "variants": build_variants(root / "missing_images")}
            (root / "generation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            args = self.options(root, "--dry-run")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/train_uda_grid.py"),
                "--generation-manifest", str(args.generation_manifest), "--dataset", args.dataset,
                "--target-list", str(args.target_list), "--target-image-root", str(args.target_image_root),
                "--output-root", str(args.output_root), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((root / "uda/uda_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["run_count"], 500)
            self.assertFalse(plan["lists_prepared"])
            self.assertFalse((root / "uda/image_lists").exists())

    def test_prepare_only_uses_completed_variants_and_writes_correct_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = build_variants(root / "images", token_strengths=[0.6], noise_levels=[1.0])
            for variant in variants:
                variant["status"] = "complete"
                variant["expected_images"] = 65
                for name in dataset_classes("OfficeHome"):
                    folder = Path(variant["image_root"]) / name
                    folder.mkdir(parents=True)
                    (folder / "sample.png").touch()
            target_root = root / "real images"
            target_root.mkdir()
            (target_root / "real.png").touch()
            (root / "target.txt").write_text("real.png 0\n", encoding="utf-8")
            (root / "generation_manifest.json").write_text(json.dumps({"schema_version": 1, "variants": variants}), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/train_uda_grid.py"),
                       "--generation-manifest", str(root / "generation_manifest.json"), "--dataset", "OfficeHome",
                       "--target-list", str(root / "target.txt"), "--target-image-root", str(target_root),
                       "--output-root", str(root / "uda"), "--prepare-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((root / "uda/uda_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(plan["lists_prepared"])
            self.assertEqual(plan["run_count"], 1)
            for listing in plan["runs"][0]["source_lists"]:
                samples = read_image_list(listing, root, 65)
                self.assertEqual(len(samples), 65)
                self.assertEqual({label for _, label in samples}, set(range(65)))
            variants[0]["status"] = "planned"
            (root / "generation_manifest.json").write_text(json.dumps({"schema_version": 1, "variants": variants}), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Generation must complete", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
