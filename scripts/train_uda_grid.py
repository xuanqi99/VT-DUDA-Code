"""Train UDA on DDIM/class dataset pairs with matching adapter strengths."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from UDA.generated_data import dataset_classes, read_image_list

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ADAPTER_STRENGTHS = tuple(value / 10 for value in range(6, 11))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=["OfficeHome", "Office31", "VisDA2017"], required=True)
    parser.add_argument("--target-list", type=Path, required=True,
                        help="Real target list: image path followed by integer label in the supplied dataset's CLASSES order.")
    parser.add_argument("--target-image-root", type=Path, required=True,
                        help="Root for relative image paths in --target-list.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", choices=["els", "mcc"], nargs="+", default=["els"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--adapter-strengths", type=float, nargs="+", choices=ADAPTER_STRENGTHS,
                        help="Select matching DDIM/class adapter strengths; defaults to all strengths in the manifest.")
    parser.add_argument("--ddim-token-strengths", type=float, nargs="+")
    parser.add_argument("--ddim-noise-levels", type=float, nargs="+")
    parser.add_argument("--class-token-strengths", type=float, nargs="+")
    parser.add_argument("--gpu", default="0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write the plan without loading images or running training (default).")
    mode.add_argument("--prepare-only", action="store_true", help="Validate all selected datasets and write lists without training.")
    mode.add_argument("--run", action="store_true", help="Validate, prepare and sequentially execute all planned training commands.")
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional native training options, e.g. --train-args --epochs 50 --eps 0.1 (ELS only).")
    return parser


def normalized_class(name):
    return name.casefold().replace("_", " ").strip()


def generated_samples(image_root, classes):
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Generated dataset directory does not exist: {image_root}")
    expected = {normalized_class(name): index for index, name in enumerate(classes)}
    samples, present = [], set()
    for folder in sorted(image_root.iterdir()):
        if not folder.is_dir():
            continue
        key = normalized_class(folder.name)
        if key not in expected:
            raise ValueError(f"Unknown class folder {folder}; expected supplied {len(classes)}-class mapping")
        label = expected[key]
        if label in present:
            raise ValueError(f"Ambiguous duplicate class folder: {folder}")
        images = sorted(p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Empty generated class: {folder}")
        samples.extend((str(image), label) for image in images)
        present.add(label)
    missing = [name for index, name in enumerate(classes) if index not in present]
    if missing:
        raise ValueError(f"Missing generated classes under {image_root}: {', '.join(missing)}")
    return samples


def select_variants(manifest, pipeline, strengths=None, noise_levels=None, adapter_strengths=None):
    variants = []
    for variant in manifest.get("variants", []):
        if variant.get("pipeline") != pipeline:
            continue
        if adapter_strengths and variant["adapter_strength"] not in adapter_strengths:
            continue
        if strengths and variant["token_strength"] not in strengths:
            continue
        if noise_levels and variant.get("noise_level") not in noise_levels:
            continue
        variants.append(variant)
    if not variants:
        raise ValueError(f"No {pipeline} variants match the selected parameter values")
    for field, selected in (("token_strength", strengths), ("noise_level", noise_levels),
                            ("adapter_strength", adapter_strengths)):
        if selected and set(selected) - {variant.get(field) for variant in variants}:
            raise ValueError(f"Requested {pipeline} {field} values absent from the manifest")
    return variants


def write_image_list(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image} {label}\n" for image, label in samples), encoding="utf-8")


def make_plan(args, manifest):
    schema_version = manifest.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("Expected generation manifest schema_version 1 or 2")
    variants = []
    for variant in manifest.get("variants", []):
        variant = dict(variant)
        if schema_version == 2 and "adapter_strength" not in variant:
            raise ValueError(f"Schema 2 variant is missing adapter_strength: {variant.get('id')}")
        adapter_strength = float(variant.get("adapter_strength", 1.0))
        if adapter_strength not in ADAPTER_STRENGTHS:
            raise ValueError(f"Unsupported adapter strength {adapter_strength} in variant {variant.get('id')}")
        variant["adapter_strength"] = adapter_strength
        variants.append(variant)
    manifest = dict(manifest, variants=variants)
    protected = {"--source-list", "--target-list", "--target-image-root", "--log", "--seed", "--gpu",
                 "-d", "--data", "-s", "--source", "-t", "--target"}
    if any(argument.split("=", 1)[0] in protected for argument in args.train_args):
        raise ValueError("--train-args cannot override dataset, log, seed or GPU options recorded by the grid")
    ddim = select_variants(manifest, "ddim_inversion", args.ddim_token_strengths, args.ddim_noise_levels,
                           args.adapter_strengths)
    class_variants = select_variants(manifest, "class_condition", args.class_token_strengths,
                                    adapter_strengths=args.adapter_strengths)
    ddim_adapters = {variant["adapter_strength"] for variant in ddim}
    class_adapters = {variant["adapter_strength"] for variant in class_variants}
    if ddim_adapters != class_adapters:
        raise ValueError("Selected DDIM and class variants must have matching adapter strengths; "
                         f"DDIM-only: {sorted(ddim_adapters - class_adapters)}, "
                         f"class-only: {sorted(class_adapters - ddim_adapters)}")
    adapters = sorted(ddim_adapters)
    pairs = [(ddim_variant, class_variant)
             for adapter in adapters
             for ddim_variant, class_variant in itertools.product(
                 (variant for variant in ddim if variant["adapter_strength"] == adapter),
                 (variant for variant in class_variants if variant["adapter_strength"] == adapter))]
    selected = ddim + class_variants
    ids = [variant["id"] for variant in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate generation variant IDs")
    for variant in selected:
        if Path(variant["id"]).name != variant["id"] or variant["id"] in {".", ".."}:
            raise ValueError(f"Unsafe variant ID: {variant['id']}")
    output_root = args.output_root.expanduser().resolve()
    lists = {variant["id"]: output_root / "image_lists" / f"{variant['id']}.txt" for variant in selected}
    target_list = output_root / "image_lists" / "real_target.txt"
    runs = []
    for method, seed, (ddim_variant, class_variant) in itertools.product(
        dict.fromkeys(args.methods), dict.fromkeys(args.seeds), pairs
    ):
        adapter_strength = ddim_variant["adapter_strength"]
        run_id = f"{ddim_variant['id']}__{class_variant['id']}"
        log_dir = output_root / f"adapter_{adapter_strength:.1f}" / method / f"seed_{seed}" / run_id
        if method == "mcc":
            entry = PROJECT_ROOT / "UDA/MCC/examples/domain_adaptation/image_classification/train_mcc.py"
        elif args.dataset == "VisDA2017":
            entry = PROJECT_ROOT / "UDA/ELS/train_els_visda.py"
        else:
            entry = PROJECT_ROOT / "UDA/ELS/train_els.py"
        command = [sys.executable, str(entry), str(args.target_image_root.expanduser().resolve()),
                   "-d", args.dataset, "-s", "grid_ddim", "grid_class", "-t", "real_target",
                   "--source-list", str(lists[ddim_variant["id"]]), str(lists[class_variant["id"]]),
                   "--target-list", str(target_list), "--target-image-root", str(args.target_image_root.expanduser().resolve()),
                   "--log", str(log_dir), "--seed", str(seed), "--gpu", args.gpu]
        if args.dataset == "VisDA2017":
            command.append("--per-class-eval")
        command.extend(args.train_args)
        runs.append({"method": method, "seed": seed, "adapter_strength": adapter_strength,
                     "ddim_variant": ddim_variant["id"],
                     "class_variant": class_variant["id"], "ddim_token_strength": ddim_variant["token_strength"],
                     "ddim_noise_level": ddim_variant["noise_level"], "class_token_strength": class_variant["token_strength"],
                     "source_lists": [str(lists[ddim_variant["id"]]), str(lists[class_variant["id"]])],
                     "target_list": str(target_list), "log_dir": str(log_dir), "command": command})
    return {"schema_version": 2, "dataset": args.dataset, "generation_manifest": str(args.generation_manifest.resolve()),
            "generation_schema_version": schema_version, "adapter_strengths": adapters,
            "adapter_groups": [{"adapter_strength": adapter,
                                "combination_count": sum(ddim_variant["adapter_strength"] == adapter
                                                         for ddim_variant, _ in pairs)} for adapter in adapters],
            "class_names": dataset_classes(args.dataset),
            "source_policy": "concatenate_one_ddim_and_one_class_variant_with_matching_adapter_strength",
            "target_policy": "real_target_labels_used_only_for_evaluation", "target_input_list": str(args.target_list.resolve()),
            "target_image_root": str(args.target_image_root.resolve()), "combination_count": len(pairs),
            "run_count": len(runs), "lists_prepared": False, "runs": runs}, selected, lists, target_list


def main():
    args = build_parser().parse_args()
    try:
        manifest = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
        plan, selected, lists, target_list = make_plan(args, manifest)
        if args.run or args.prepare_only:
            incomplete = [variant["id"] for variant in selected if variant.get("status") != "complete"]
            if incomplete:
                raise ValueError("Generation must complete before preparing/training UDA; incomplete variants: " + ", ".join(incomplete))
            if args.run:
                occupied = [run["log_dir"] for run in plan["runs"] if any(Path(run["log_dir"]).rglob("*.pth"))]
                if occupied:
                    raise FileExistsError(f"Existing training checkpoints under {occupied[0]}; choose a fresh --output-root")
            target_samples = read_image_list(args.target_list, args.target_image_root, len(plan["class_names"]))
            # Validate every selected dataset before writing files or starting a run.
            samples = {variant["id"]: generated_samples(variant["image_root"], plan["class_names"]) for variant in selected}
            for variant in selected:
                expected = variant.get("expected_images")
                actual = len(samples[variant["id"]])
                if expected is not None and actual != expected:
                    raise ValueError(f"{variant['id']}: manifest expects {expected} images, found {actual}")
            for variant_id, records in samples.items():
                write_image_list(lists[variant_id], records)
            write_image_list(target_list, target_samples)
            plan["lists_prepared"] = True
        output_root = args.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        plan_path = output_root / "uda_manifest.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Planned {plan['combination_count']} DDIM/class pairs; {plan['run_count']} method/seed runs.")
        print(f"Manifest: {plan_path}")
        if not args.run:
            print("No training started. Use --run to execute the plan.")
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        for index, run in enumerate(plan["runs"], 1):
            print(f"[{index}/{plan['run_count']}] {run['method']} {run['ddim_variant']} + {run['class_variant']}", flush=True)
            subprocess.run(run["command"], cwd=PROJECT_ROOT, env=env, check=True)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
