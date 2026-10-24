"""Generate the post-fine-tuning DDIM and class-conditioned dataset grid."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_STRENGTHS = tuple(value / 10 for value in range(6, 16))
NOISE_LEVELS = tuple(value / 10 for value in range(6, 11))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Fine-tuned LoRA directory containing concept_encoder.pt.")
    parser.add_argument("--source_root", required=True,
                        help="One source domain with class/image subdirectories.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for synthesis subprocesses.")
    parser.add_argument("--pipeline", choices=("both", "ddim_inversion", "class_condition"), default="both")
    parser.add_argument("--token_strengths", type=float, nargs="+", choices=TOKEN_STRENGTHS,
                        default=list(TOKEN_STRENGTHS))
    parser.add_argument("--noise_levels", type=float, nargs="+", choices=NOISE_LEVELS,
                        default=list(NOISE_LEVELS))
    parser.add_argument("--images_per_class", type=positive_int, default=50,
                        help="Images per class for each class-condition variant.")
    parser.add_argument("--reference_seeds", type=positive_int, default=1,
                        help="Generated samples per source reference for each DDIM variant.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=positive_int, default=25)
    parser.add_argument("--concept_token_count", type=positive_int, default=None,
                        help="Inferred from the fine-tuned concept encoder when omitted (DDIM branch).")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--invert_guidance_scale", type=float, default=0.0)
    parser.add_argument("--reference_cross_attn_scale", type=float, default=1.0)
    parser.add_argument("--class_cross_attn_scale", type=float, default=0.9)
    parser.add_argument("--center_crop", action="store_true",
                        help="Use center crops in the DDIM branch, matching its existing option.")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Optional exact source class folder names.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Write a planned manifest and print counts; do not import ML libraries or generate images.")
    return parser


def source_classes(source_root, requested_classes=None):
    if not source_root.is_dir():
        raise ValueError(f"Source domain directory does not exist: {source_root}")
    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if requested_classes is not None:
        requested = set(requested_classes)
        missing = requested - {path.name for path in class_dirs}
        if missing:
            raise ValueError(f"Unknown source class folders: {sorted(missing)}")
        class_dirs = [path for path in class_dirs if path.name in requested]
    images = {}
    for class_dir in class_dirs:
        candidates = sorted(path for path in class_dir.rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not candidates:
            raise ValueError(f"No source images in class directory: {class_dir}")
        images[class_dir.name] = candidates
    if not images:
        raise ValueError(f"No class folders found in source domain: {source_root}")
    return images


def build_variants(output_root, token_strengths=TOKEN_STRENGTHS, noise_levels=NOISE_LEVELS,
                   pipeline="both"):
    """Return 50 DDIM and 10 class-condition dataset variants by default."""
    variants = []
    for strength in sorted(set(token_strengths)):
        if pipeline in ("both", "ddim_inversion"):
            for noise in sorted(set(noise_levels)):
                variants.append({
                    "id": f"ddim_token_{strength:.1f}_noise_{noise:.1f}",
                    "pipeline": "ddim_inversion",
                    "token_strength": strength,
                    "noise_level": noise,
                    "image_root": str((output_root / "ddim_inversion" /
                                       f"token_{strength:.1f}" / f"noise_{noise:.1f}").resolve()),
                    "status": "planned",
                })
        if pipeline in ("both", "class_condition"):
            variants.append({
                "id": f"class_token_{strength:.1f}",
                "pipeline": "class_condition",
                "token_strength": strength,
                "noise_level": None,
                "image_root": str((output_root / "class_condition" /
                                   f"token_{strength:.1f}").resolve()),
                "status": "planned",
            })
    return variants


def reference_output_path(image_root, source_root, source_image, seed):
    relative = source_image.relative_to(source_root)
    # Include the suffix and relative path in the digest to distinguish a.jpg/a.png
    # and nested images with the same stem, while retaining the class folder.
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
    return image_root / relative.parts[0] / f"{source_image.stem}_{digest}_seed{seed}.png"


def commands_for_variant(args, variant, images):
    common = ["--base_model_id", args.base_model_id, "--device", args.device,
              "--num_inference_steps", str(args.num_inference_steps),
              "--guidance_scale", str(args.guidance_scale),
              "--concept_token_strength", str(variant["token_strength"])]
    image_root = Path(variant["image_root"])
    if variant["pipeline"] == "class_condition":
        yield [args.python, str(PROJECT_ROOT / "diffusion" / "synthesize_from_noise.py"),
               "--checkpoint_dir", str(Path(args.checkpoint_dir).resolve()),
               "--source_root", str(Path(args.source_root).resolve()),
               "--output_dir", str(image_root),
               "--images_per_class", str(args.images_per_class),
               "--seed", str(args.seed),
               "--cross_attn_scale", str(args.class_cross_attn_scale),
               *common, "--classes", *images.keys()]
        return
    for class_images in images.values():
        for source_image in class_images:
            for offset in range(args.reference_seeds):
                seed = args.seed + offset
                output_image = reference_output_path(
                    image_root, Path(args.source_root).resolve(), source_image, seed
                )
                command = [args.python, str(PROJECT_ROOT / "diffusion" / "synthesize_from_reference.py"),
                           "--lora_path", str(Path(args.checkpoint_dir).resolve()),
                           "--input_image", str(source_image),
                           "--concept_image", str(source_image),
                           "--output_dir", str(image_root),
                           "--output_image", str(output_image),
                           "--skip_visualizations", "--no_prompt",
                           "--invert_noise_scale", str(variant["noise_level"]),
                           "--invert_guidance_scale", str(args.invert_guidance_scale),
                           "--cross_attn_scale", str(args.reference_cross_attn_scale),
                           "--seed", str(seed), *common]
                if args.center_crop:
                    command.append("--center_crop")
                if args.concept_token_count is not None:
                    command.extend(["--concept_token_count", str(args.concept_token_count)])
                yield command


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        images = source_classes(Path(args.source_root).resolve(), args.classes)
    except ValueError as error:
        parser.error(str(error))
    output_root = Path(args.output_root).resolve()
    source_root = Path(args.source_root).resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        parser.error("--source_root and --output_root must be separate directories, with neither inside the other")
    variants = build_variants(output_root, args.token_strengths, args.noise_levels, args.pipeline)
    reference_count = sum(len(paths) for paths in images.values())
    for variant in variants:
        variant["expected_images"] = (reference_count * args.reference_seeds
                                      if variant["pipeline"] == "ddim_inversion"
                                      else len(images) * args.images_per_class)
    manifest = {
        "schema_version": 1,
        "stage": "post_finetuning_generation",
        "status": "planned",
        "source_root": str(Path(args.source_root).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "classes": list(images),
        "parameters": {key: value for key, value in vars(args).items() if key != "dry_run"},
        "ddim_noise_semantics": "(1-noise_level)*source_latent + noise_level*fully_inverted_latent; 1.0 uses the full inversion result",
        "class_condition_semantics": "class text prompt plus scaled source visual tokens from pure noise",
        "variants": variants,
    }
    manifest_path = output_root / "generation_manifest.json"
    # A dry run must not replace completion records from an existing generation run.
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            parser.error(f"A different or already executed manifest exists: {manifest_path}; "
                         "choose a new --output_root to preserve its provenance")
    write_manifest(manifest_path, manifest)
    ddim_count = sum(v["pipeline"] == "ddim_inversion" for v in variants)
    class_count = sum(v["pipeline"] == "class_condition" for v in variants)
    print(f"Generation grid: {ddim_count} DDIM inversion variants + {class_count} class-condition variants")
    print(f"Source: {len(images)} classes, {reference_count} reference images")
    print(f"Planned final images: {sum(v['expected_images'] for v in variants)}")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        return 0
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "concept_encoder.pt").is_file():
        parser.error(f"Missing fine-tuned checkpoint: {checkpoint_dir / 'concept_encoder.pt'}")
    manifest["status"] = "running"
    write_manifest(manifest_path, manifest)
    for index, variant in enumerate(variants, 1):
        print(f"[{index}/{len(variants)}] {variant['id']}", flush=True)
        variant["status"] = "running"
        write_manifest(manifest_path, manifest)
        try:
            for command in commands_for_variant(args, variant, images):
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            variant["status"] = "failed"
            manifest["status"] = "failed"
            write_manifest(manifest_path, manifest)
            print(f"Generation failed for {variant['id']}: {error}", file=sys.stderr)
            return 1
        variant["status"] = "complete"
        write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    write_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
