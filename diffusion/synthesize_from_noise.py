import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin

from ldm.modules.diffusionmodules.openaimodel_enc import Encoder4
from synthesize_from_reference import (
    TerraPatcher, _predict_eps, _prepare_sdxl_conditioning, adapter_strength_value, latents_to_pil,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "sd" / "models" / "stabilityai" / "stable-diffusion-xl-base-1.0" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
DEFAULT_MODEL_PATH = str(next(DEFAULT_MODEL_DIR.iterdir())) if DEFAULT_MODEL_DIR.exists() and any(DEFAULT_MODEL_DIR.iterdir()) else "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser():
    parser = argparse.ArgumentParser("Noise-to-target synthesis")
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "noise_synthesis"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adapter_strength", type=adapter_strength_value, default=1.0,
                        help="Multiplier for LoRA weight updates during generation.")
    parser.add_argument("--cross_attn_scale", type=float, default=0.9)
    parser.add_argument("--concept_token_strength", type=float, default=1.0,
                        help="Multiplier applied to source visual tokens, independently of CFG and LoRA scales.")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--images_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=None)
    return parser


def list_classes(source_root: Path, classes):
    if classes:
        return classes
    return sorted([p.name for p in source_root.iterdir() if p.is_dir()])


def normalize_class_name(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def make_image_transform():
    return transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def sample_source_image(source_root: Path, class_name: str) -> Path | None:
    class_dir = source_root / class_name
    if not class_dir.exists():
        matches = [p for p in source_root.iterdir() if p.is_dir() and normalize_class_name(p.name) == normalize_class_name(class_name)]
        class_dir = matches[0] if matches else None
    if class_dir is None or not class_dir.exists():
        return None
    candidates = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not candidates:
        return None
    return random.choice(candidates)


def load_concept_encoder(pipe, checkpoint_dir: Path, device: str):
    concept_path = checkpoint_dir / "concept_encoder.pt"
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concept encoder: {concept_path}")
    with torch.no_grad():
        embeds = pipe.encode_prompt(prompt=["x"], device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
    embed_dim = embeds.shape[-1]
    state_dict = torch.load(concept_path, map_location="cpu")
    latent_unit = int(state_dict.get("encoder.17.bias", torch.empty(0)).shape[0] or state_dict["encoder.17.weight"].shape[0])
    encoder = Encoder4(d=128, context_dim=embed_dim, latent_unit=latent_unit, num_channels=3).to(device=pipe.device, dtype=torch.float32)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    return encoder, latent_unit, embed_dim


@torch.no_grad()
def extract_concept_tokens(image_path: Path, image_transform, encoder, token_count: int, embed_dim: int, device: str):
    image = exif_transpose(Image.open(image_path)).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    tokens = encoder(tensor).view(1, token_count, embed_dim)
    return tokens


def main():
    args = build_parser().parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_model_id, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(str(checkpoint_dir))
    patcher = TerraPatcher(pipe, lora_state_dict)
    patcher.apply(time_t=1.0, adapter_strength=args.adapter_strength)

    encoder, token_count, embed_dim = load_concept_encoder(pipe, checkpoint_dir, device)
    image_transform = make_image_transform()
    class_names = list_classes(source_root, args.classes)

    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        prompt = f"a {normalize_class_name(class_name)}"
        for local_seed in range(args.images_per_class):
            seed_value = args.seed + local_seed
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            source_image = sample_source_image(source_root, class_name)
            if source_image is None:
                raise FileNotFoundError(f"No source image found for class {class_name} under {source_root}")
            concept_tokens = extract_concept_tokens(source_image, image_transform, encoder, token_count, embed_dim, device)
            concept_tokens = concept_tokens * args.concept_token_strength
            prompt_embeds, _, added_cond, do_cfg = _prepare_sdxl_conditioning(
                pipe,
                prompt,
                args.guidance_scale,
                device=device,
                dtype=pipe.unet.dtype,
                crop_coords=(0, 0),
                concept_tokens=concept_tokens,
            )
            generator = torch.Generator(device=device).manual_seed(seed_value)
            latents = torch.randn((1, pipe.unet.config.in_channels, 128, 128), generator=generator, device=device, dtype=torch.float32)
            latents = latents * getattr(pipe.scheduler, "init_noise_sigma", 1.0)
            pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
            for t in pipe.scheduler.timesteps:
                t_batch = torch.full((latents.shape[0],), int(t.item() if torch.is_tensor(t) else t), device=device, dtype=torch.long)
                eps_pred = _predict_eps(
                    pipe=pipe,
                    latents=latents,
                    t=t_batch,
                    prompt_embeds=prompt_embeds,
                    added_cond=added_cond,
                    guidance_scale=args.guidance_scale,
                    do_cfg=do_cfg,
                    cross_attention_kwargs={"scale": args.cross_attn_scale},
                )
                latents = pipe.scheduler.step(eps_pred, t, latents, generator=generator).prev_sample
            image = latents_to_pil(pipe, latents)
            image.save(class_dir / f"{class_name}_seed{seed_value}.png")

    patcher.reset()


if __name__ == "__main__":
    main()
