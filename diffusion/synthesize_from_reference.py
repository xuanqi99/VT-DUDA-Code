import argparse
import math
import os
from typing import Dict, Tuple, Any
from pathlib import Path         

import torch
from PIL import Image
from torchvision import transforms
from PIL.ImageOps import exif_transpose         

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.loaders import LoraLoaderMixin
import types
       
import hashlib
from torchvision.transforms import functional as TF
from ldm.modules.diffusionmodules.openaimodel_enc import Encoder4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "sd" / "models" / "stabilityai" / "stable-diffusion-xl-base-1.0" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
DEFAULT_MODEL_PATH = str(next(DEFAULT_MODEL_DIR.iterdir())) if DEFAULT_MODEL_DIR.exists() and any(DEFAULT_MODEL_DIR.iterdir()) else "stabilityai/stable-diffusion-xl-base-1.0"


def _target_layer_from_sd_name(k: str) -> str:
                                                                                                
    if ".processor.to_" in k:
        target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
        target_layer = target_layer.replace("to_out", "to_out[0]")
    else:
        target_layer = k.split(".lora.")[0]
    for i in range(10):
        target_layer = target_layer.replace(f".{i}", f"[{i}]")
    return target_layer


class TerraPatcher:
    def __init__(self, pipe: StableDiffusionXLPipeline, lora_state_dict: Dict[str, torch.Tensor]):
        self.pipe = pipe
        self.device = pipe.device
        self.dtype = pipe.dtype
                                                            
        self.unet_lora = {k: v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        self._applied = []                                                             

    def reset(self):
                                            
        for weight, original in reversed(self._applied):
            with torch.no_grad():
                weight.copy_(original)
        self._applied.clear()

    @torch.no_grad()
    def apply(self, time_t: float = 1.0, adapter_strength: float = 1.0):
        """
        Apply a scaled Terra LoRA update at time t in {0, 1}:
        dW = B @ (I + Ω * t)^T @ A, or dW = B @ A if Ω is absent.
        """
                                                                      
        strength = float(adapter_strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("adapter_strength must be finite and nonnegative")
        self.reset()
        if strength == 0:
            return

                                                                
        for k, a1 in list(self.unet_lora.items()):
            if not k.endswith("down.weight"):
                continue

            k_up = k.replace("down.weight", "up.weight")
            k_om = k.replace("down.weight", "diagnal.omegas_weights")
            if k_up not in self.unet_lora:
                continue

            a1 = self.unet_lora[k].to(self.device, dtype=torch.float32)
            b1 = self.unet_lora[k_up].to(self.device, dtype=torch.float32)

            omegas = self.unet_lora.get(k_om, None)
            if omegas is not None:
                omegas = omegas.to(self.device, dtype=torch.float32)
                rank = omegas.shape[0]
                eye = torch.eye(rank, device=self.device, dtype=torch.float32)
                matrix = eye + omegas * float(time_t)
                dW = b1 @ matrix.T @ a1
            else:
                                                   
                dW = b1 @ a1

                                      
            layer_key = k[len("unet."):] if k.startswith("unet.") else k
            target_layer = _target_layer_from_sd_name(layer_key)
            weight = getattr(eval(f"self.pipe.unet.{target_layer}"), "weight")
            dW_cast = (strength * dW).to(self.pipe.unet.dtype).to(weight.device)

            original = weight.detach().clone()
            weight.add_(dW_cast)
            self._applied.append((weight, original))


def adapter_strength_value(value: str) -> float:
    strength = float(value)
    if not math.isfinite(strength) or strength < 0:
        raise argparse.ArgumentTypeError("adapter strength must be finite and nonnegative")
    return strength


def preprocess_image(image_path: str, size: int = 1024, center_crop: bool = False, seed: int = 42) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(image_path)
    image = exif_transpose(image).convert("RGB")                                        

                          
    image = TF.resize(image, size, interpolation=transforms.InterpolationMode.BILINEAR)
    w, h = image.size

                                                                
    if center_crop or (w == size and h == size):
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
    else:
                                                                  
        key = f"{image_path}|{seed}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        max_top = max(0, h - size)
        max_left = max(0, w - size)
        top = int(int.from_bytes(digest[:8], "little") % (max_top + 1)) if max_top > 0 else 0
        left = int(int.from_bytes(digest[8:16], "little") % (max_left + 1)) if max_left > 0 else 0

    image = TF.crop(image, top, left, size, size)

    t = transforms.Compose(
        [
            transforms.ToTensor(),         
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                          
        ]
    )
    return t(image).unsqueeze(0), (top, left)


@torch.no_grad()
def latents_to_pil(pipe: StableDiffusionXLPipeline, latents: torch.Tensor) -> Image.Image:
                                                   
    latents = latents.to(torch.float32)                                     
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.to(dtype=torch.float32).decode(latents).sample
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def save_image(img: Image.Image, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

                                                                                
def _select_device(user_device: str) -> tuple[str, torch.dtype]:
    if user_device.startswith("cuda") and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            if (major, minor) < (7, 5):
                print(f"[warn] CUDA compute capability {major}.{minor} is unsupported by this PyTorch build. Falling back to CPU.")
                return "cpu", torch.float32
            return "cuda", torch.float16
        except Exception as e:
            print(f"[warn] CUDA capability check failed ({e}). Falling back to CPU.")
            return "cpu", torch.float32
    return "cpu", torch.float32

                                                                 
def _encode_prompt_manual(pipe: StableDiffusionXLPipeline, prompt: str, negative_prompt: str | None, device, dtype, do_cfg: bool, crop_coords: Tuple[int, int] = (0, 0)):
                  
    def _tok(tok, text):
        out = tok(
            text,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return out.input_ids.to(device), (out.attention_mask.to(device) if "attention_mask" in out else None)

              
    ids1_pos, mask1_pos = _tok(pipe.tokenizer, prompt)
    ids2_pos, mask2_pos = _tok(pipe.tokenizer_2, prompt)

    out1_pos = pipe.text_encoder(ids1_pos, attention_mask=mask1_pos, output_hidden_states=True, return_dict=True)
    out2_pos = pipe.text_encoder_2(ids2_pos, attention_mask=mask2_pos, output_hidden_states=True, return_dict=True)

    hid1_pos = out1_pos.hidden_states[-2]
    hid2_pos = out2_pos.hidden_states[-2]
    prompt_pos = torch.cat([hid1_pos, hid2_pos], dim=-1).to(device=device, dtype=dtype)

                                                       
    if hasattr(out2_pos, "text_embeds") and out2_pos.text_embeds is not None:
        pooled_pos = out2_pos.text_embeds
    elif hasattr(out2_pos, "pooled_output") and out2_pos.pooled_output is not None:
        pooled_pos = out2_pos.pooled_output
    else:
        pooled_pos = out2_pos.last_hidden_state[:, 0, :] if hasattr(out2_pos, "last_hidden_state") else out2_pos[0][:, 0, :]
    pooled_pos = pooled_pos.to(device=device, dtype=dtype)

    if do_cfg:
                                                
        neg_text = negative_prompt if negative_prompt is not None else ""
        ids1_neg, mask1_neg = _tok(pipe.tokenizer, neg_text)
        ids2_neg, mask2_neg = _tok(pipe.tokenizer_2, neg_text)

        out1_neg = pipe.text_encoder(ids1_neg, attention_mask=mask1_neg, output_hidden_states=True, return_dict=True)
        out2_neg = pipe.text_encoder_2(ids2_neg, attention_mask=mask2_neg, output_hidden_states=True, return_dict=True)

        hid1_neg = out1_neg.hidden_states[-2]
        hid2_neg = out2_neg.hidden_states[-2]
        prompt_neg = torch.cat([hid1_neg, hid2_neg], dim=-1).to(device=device, dtype=dtype)

        if hasattr(out2_neg, "text_embeds") and out2_neg.text_embeds is not None:
            pooled_neg = out2_neg.text_embeds
        elif hasattr(out2_neg, "pooled_output") and out2_neg.pooled_output is not None:
            pooled_neg = out2_neg.pooled_output
        else:
            pooled_neg = out2_neg.last_hidden_state[:, 0, :] if hasattr(out2_neg, "last_hidden_state") else out2_neg[0][:, 0, :]
        pooled_neg = pooled_neg.to(device=device, dtype=dtype)

        prompt_embeds = torch.cat([prompt_neg, prompt_pos], dim=0)
        pooled = torch.cat([pooled_neg, pooled_pos], dim=0)
    else:
        prompt_embeds = prompt_pos
        pooled = pooled_pos

              
    proj_dim = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    proj_dim = getattr(proj_dim, "projection_dim", None) or pooled.shape[-1]
    add_time_ids = pipe._get_add_time_ids(
        original_size=(1024, 1024),
        crops_coords_top_left=tuple(map(int, crop_coords)),
        target_size=(1024, 1024),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=proj_dim,
    ).to(device)
    if do_cfg:
        add_time_ids = add_time_ids.repeat(2, 1)

    added_cond = {"time_ids": add_time_ids, "text_embeds": pooled}
                                             
    return prompt_embeds, pooled, added_cond, do_cfg

def _prepare_sdxl_conditioning(pipe: StableDiffusionXLPipeline, prompt: str, guidance_scale: float, device, dtype, crop_coords: Tuple[int, int], concept_tokens: torch.Tensor | None = None):
    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    negative_prompt = "" if do_cfg else None

    enc = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
    )

                             
    try:
        if do_cfg:
            prompt_embeds = enc[0]
            neg_embeds = enc[1]
            pooled = enc[-2]
            neg_pooled = enc[-1]
            neg_embeds = neg_embeds.to(device=device, dtype=dtype)
            neg_pooled = neg_pooled.to(device=device, dtype=dtype)
        else:
            prompt_embeds = enc[0]
            pooled = enc[2]
            neg_embeds = neg_pooled = None
    except Exception:
        prompt_embeds = pooled = neg_embeds = neg_pooled = None

                                 
    if pooled is None or prompt_embeds is None:
        prompt_embeds, pooled, added_cond, do_cfg = _encode_prompt_manual(
            pipe, prompt, negative_prompt, device, dtype, do_cfg, crop_coords
        )
        if concept_tokens is not None:
            tokens = concept_tokens.repeat(2, 1, 1) if do_cfg else concept_tokens
            prompt_embeds = torch.cat([prompt_embeds, tokens.to(prompt_embeds)], dim=1)
        return prompt_embeds, pooled, added_cond, do_cfg

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled = pooled.to(device=device, dtype=dtype)

    proj_dim = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    proj_dim = getattr(proj_dim, "projection_dim", None)
    if proj_dim is None:
        proj_dim = pooled.shape[-1]

    add_time_ids = pipe._get_add_time_ids(
        original_size=(1024, 1024),
        crops_coords_top_left=tuple(map(int, crop_coords)),
        target_size=(1024, 1024),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=proj_dim,
    ).to(device)

    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pooled = torch.cat([neg_pooled, pooled], dim=0)
        add_time_ids = add_time_ids.repeat(2, 1)
    added_cond = {"time_ids": add_time_ids, "text_embeds": pooled}
                                   
    if concept_tokens is not None:
        concept_tokens_in = concept_tokens.repeat(2, 1, 1) if do_cfg else concept_tokens
        prompt_embeds = torch.cat([prompt_embeds, concept_tokens_in.to(prompt_embeds.dtype)], dim=1)
                                                        
    return prompt_embeds, pooled, added_cond, do_cfg

def _get_alpha_terms(scheduler: DDIMScheduler, t: torch.Tensor, device):
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=torch.long)
    t = t.to(device=device, dtype=torch.long)
    alpha_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[t]
    alpha_cumprod = torch.clamp(alpha_cumprod, min=1e-20, max=1.0)
    a = torch.sqrt(alpha_cumprod)
    b = torch.sqrt(1 - alpha_cumprod)
                                           
    while a.ndim < 4:
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
    return a, b

@torch.no_grad()
def _predict_eps(
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond: Dict[str, torch.Tensor],
    do_cfg: bool,
    guidance_scale: float,
    cross_attention_kwargs: Dict[str, Any] | None = None,
):
    latents_device = pipe.unet.device
    latents_dtype = pipe.unet.dtype
    t = t.to(device=latents_device, dtype=torch.long)

    latents_fp32 = latents.to(device=latents_device, dtype=torch.float32)
    latents_in = latents_fp32.to(dtype=latents_dtype)

    if do_cfg:
        latents_in = torch.cat([latents_in, latents_in], dim=0)
        latents_fp32 = torch.cat([latents_fp32, latents_fp32], dim=0)
        t_in = torch.cat([t, t], dim=0)
    else:
        t_in = t

    prompt_embeds_in = prompt_embeds.to(device=latents_device, dtype=latents_dtype)

    added_cond_in = {}
    for k, v in added_cond.items():
        target = v.to(device=latents_device)
        if torch.is_floating_point(target):
            target = target.to(dtype=latents_dtype)
        added_cond_in[k] = target

    model_out = pipe.unet(
        latents_in,
        t_in,
        encoder_hidden_states=prompt_embeds_in,
        added_cond_kwargs=added_cond_in,
        cross_attention_kwargs=cross_attention_kwargs,
    ).sample.to(torch.float32)

    pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
    if pred_type == "v_prediction":
        a_t, b_t = _get_alpha_terms(pipe.scheduler, t_in, device=latents_device)
        eps_all = a_t * model_out + b_t * latents_fp32
    else:
        eps_all = model_out

    if do_cfg:
        eps_uncond, eps_text = eps_all.chunk(2, dim=0)
        return eps_uncond + guidance_scale * (eps_text - eps_uncond)
    return eps_all


                                                                                     

def _normalize_officehome_label(name: str) -> str:
                                      
    return name.replace("_", " ").strip().lower()

def build_officehome_prompt_from_path(image_path: str) -> str:
                                                                     
    cls_raw = Path(image_path).parent.name
    cls_norm = _normalize_officehome_label(cls_raw)
                                                     
    return f"a {cls_norm}"


def main():
    parser = argparse.ArgumentParser("Reference-guided synthesis")
    parser.add_argument("--base_model_id", type=str, required=False, default=DEFAULT_MODEL_PATH, help="Path or repo id of the SDXL base model")
    parser.add_argument("--lora_path", type=str, required=False, default=str(PROJECT_ROOT / "checkpoints" / "synthesis"), help="Path to Terra LoRA weights directory")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=False, default=str(PROJECT_ROOT / "outputs" / "reference_synthesis"), help="Directory to save visualizations")
    parser.add_argument("--output_image", type=str, default=None,
                        help="Exact final image path (requires --num_seeds 1).")
    parser.add_argument("--skip_visualizations", action="store_true",
                        help="Save only the final image; exclude inputs and inversion previews from datasets.")
    parser.add_argument("--vae_path", type=str, default=None, help="Optional VAE path (fp16-fix, etc.)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adapter_strength", type=adapter_strength_value, default=1.0,
                        help="Multiplier for LoRA weight updates during inversion and generation.")
                              
    parser.add_argument("--invert_guidance_scale", type=float, default=0.0, help="Guidance scale used only during DDIM inversion")
    parser.add_argument(
        "--prompt",
                                                                                
        nargs="+",
        type=str,
        default=None,
        help="Class-conditioned prompt. If omitted, derive from the input image OfficeHome class folder, e.g., 'a alarm clock'."
    )
                                                                         
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Use empty text prompt ('') so generation is guided only by DDIM prior and concept tokens."
    )
                                                                           
    parser.add_argument(
        "--prior_only",
        action="store_true",
        help="Sample unconditionally from DDIM prior: disable prompts and concept tokens, start from pure noise."
    )
    parser.add_argument("--t_noise", type=int, choices=[0, 1], default=0, help="Terra t used during noising visualization")
    parser.add_argument("--t_denoise", type=int, choices=[0, 1], default=1, help="Terra t used during denoising (generation)")
    parser.add_argument("--save_every", type=int, default=5, help="Save every N steps of noising visualization")
    parser.add_argument("--cross_attn_scale", type=float, default=1)
                                                 
    parser.add_argument("--center_crop", action="store_true", default=False, help="Use CenterCrop(size) like training when --center_crop was used")
                   
    parser.add_argument("--save_pure_noise", action="store_true", help="Save a pure-noise latent visualization as the sampling start")
    parser.add_argument("--use_pure_noise_start", action="store_true", help="Use pure-noise latents as the generation start instead of inversion result")
    parser.add_argument("--concept_encoder_path", type=str,
                        default=None,
                        help="Path to concept_encoder.pt (Encoder4 weights).")
    parser.add_argument("--concept_token_count", type=int, default=None,
                        help="Number of concept tokens; inferred from the checkpoint when omitted.")
    parser.add_argument("--concept_token_strength", type=float, default=1, help="Scaling factor for concept tokens.")
    parser.add_argument("--concept_token_indices", type=str, default=None,
                        help="Comma-separated indices subset, e.g. '0,2,5'. If None use all.")
    parser.add_argument("--concept_image", type=str, default=None,
                        help="Optional image path for concept extraction; defaults to --input_image.")
    parser.add_argument("--invert_noise_scale", type=float, default=0.85,
                        help="Blend source and fully inverted latents: (1-scale)*source + scale*inverted. "
                             "1.0 uses the full inversion result; it does not replace it with Gaussian noise.")
                          
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of different seeds to run (images per command).")
                                                     
    parser.add_argument(
        "--gen_uncond",
        action="store_true",
        help="During t_denoise generation, ignore prompt and concept tokens; use only DDIM prior latents."
    )

    args = parser.parse_args()
    if args.output_image is not None and args.num_seeds != 1:
        parser.error("--output_image requires --num_seeds 1")
    if not 0.0 <= args.invert_noise_scale <= 1.0:
        parser.error("--invert_noise_scale must be between 0 and 1")
    if args.concept_encoder_path is None:
        args.concept_encoder_path = os.path.join(args.lora_path, "concept_encoder.pt")

                                                             
    if isinstance(args.prompt, list):
        args.prompt = " ".join(args.prompt)

                                                 
    if args.prior_only:
                              
        args.no_prompt = True
        args.use_pure_noise_start = True
        args.concept_image = None
        print("[info] PRIOR-ONLY mode: no prompt, no concept tokens; sampling unconditionally from DDIM prior (pure noise).")

                                                                             
    if args.no_prompt:
                                                       
        class_prompt = ""
        print("[info] Using EMPTY text prompt; generation is guided only by DDIM prior and concept tokens.")
    else:
        class_prompt = args.prompt if (args.prompt and args.prompt.strip()) else build_officehome_prompt_from_path(args.input_image)
        print(f"[info] Using class prompt: '{class_prompt}'")

                                                                           
    device, torch_dtype = _select_device(args.device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
    )
    if args.vae_path:
                                 
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch_dtype)
        pipe.vae = vae

                                                              
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    try:
        pipe.vae.config.force_upcast = True
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()                                                  
    except Exception:
        pass

                                                                                  
    _orig_decode = pipe.vae.decode                
    def _decode_cast(self, z, *args, **kwargs):
        z = z.to(dtype=self.post_quant_conv.weight.dtype)
        return _orig_decode(z, *args, **kwargs)
    pipe.vae.decode = types.MethodType(_decode_cast, pipe.vae)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

                                                                                      
    use_concept = args.concept_image is not None
    if use_concept:
        if not os.path.exists(args.concept_encoder_path):
            raise FileNotFoundError(f"Missing concept encoder: {args.concept_encoder_path}")
        ce_sd = torch.load(args.concept_encoder_path, map_location="cpu")
        checkpoint_token_count = int(ce_sd["encoder.17.weight"].shape[0])
        if args.concept_token_count is None:
            args.concept_token_count = checkpoint_token_count
        elif args.concept_token_count != checkpoint_token_count:
            raise ValueError(
                f"--concept_token_count={args.concept_token_count} does not match "
                f"checkpoint token count {checkpoint_token_count}"
            )
                                                                        
        tmp_embeds = pipe.encode_prompt(prompt=["x"], device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
        embed_dim = tmp_embeds.shape[-1]
        concept_encoder = Encoder4(d=128,                                                            
                                   context_dim=embed_dim,
                                   latent_unit=args.concept_token_count,
                                   num_channels=3).to(device, dtype=torch.float32)
        concept_encoder.load_state_dict(ce_sd, strict=True)
        concept_encoder.eval()

        def _parse_indices(spec: str | None, max_n: int):
            if not spec:
                return list(range(max_n))
            out = []
            for p in spec.split(","):
                p = p.strip()
                if not p:
                    continue
                i = int(p)
                if 0 <= i < max_n:
                    out.append(i)
            return out or list(range(max_n))

        selected_token_indices = _parse_indices(args.concept_token_indices, args.concept_token_count)

                                                                                 
        def _extract_concept_tokens(img_tensor: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                raw = concept_encoder(img_tensor.to(device=device, dtype=torch.float32))                              
            raw = raw.view(img_tensor.size(0), args.concept_token_count, embed_dim)
            raw = raw[:, selected_token_indices, :] * args.concept_token_strength
            return raw                                 
                                                                      

                         
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.skip_visualizations:
        save_image(Image.open(args.input_image).convert("RGB"), os.path.join(args.output_dir, "input.png"))

                             
    lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(args.lora_path)
    patcher = TerraPatcher(pipe, lora_state_dict)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

                                                                       
    concept_image_path = args.concept_image if use_concept else None

    seeds = [int(args.seed) + i for i in range(args.num_seeds)]
    for si, s in enumerate(seeds):
        img_tensor, crop_coords = preprocess_image(
            args.input_image, size=1024, center_crop=args.center_crop, seed=s
        )
        img_tensor = img_tensor.to(device, dtype=torch.float32)

                                                     
        if use_concept:
            if concept_image_path == args.input_image:
                concept_tensor = img_tensor
            else:
                concept_tensor, _ = preprocess_image(concept_image_path, size=1024, center_crop=args.center_crop, seed=s)
                concept_tensor = concept_tensor.to(device, dtype=torch.float32)
            concept_tokens = _extract_concept_tokens(concept_tensor)                 
        else:
            concept_tokens = None

                             
        with torch.no_grad():
            latents_dist = pipe.vae.encode(img_tensor)
            latents = latents_dist.latent_dist.mean * pipe.vae.config.scaling_factor
            latents = latents.to(device=device, dtype=torch.float32)

                                     
        latents_x0 = latents.clone()

        generator = torch.Generator(device=device).manual_seed(s)

                                     
        patcher.apply(time_t=float(args.t_noise), adapter_strength=args.adapter_strength)
        prompt_embeds_inv, pooled_inv, added_cond_inv, do_cfg_inv = _prepare_sdxl_conditioning(
            pipe, class_prompt, args.invert_guidance_scale, device=device,
            dtype=pipe.unet.dtype, crop_coords=crop_coords, concept_tokens=concept_tokens                           
        )

                                                               
        if si == 0 and not args.skip_visualizations:
            try:
                save_image(latents_to_pil(pipe, latents), os.path.join(args.output_dir, "input_reencoded.png"))
            except Exception:
                pass

                             
        timesteps = pipe.scheduler.timesteps
        x = latents.clone()
        for i in range(len(timesteps) - 1, 0, -1):
            t = timesteps[i]
            t_prev = timesteps[i - 1]
            t_batch = torch.full((x.shape[0],), int(t.item() if torch.is_tensor(t) else t), device=device, dtype=torch.long)

            eps_pred = _predict_eps(
                pipe=pipe,
                latents=x,
                t=t_batch,
                prompt_embeds=prompt_embeds_inv,
                added_cond=added_cond_inv,
                do_cfg=do_cfg_inv,
                guidance_scale=args.invert_guidance_scale,
            )
            a_t, b_t = _get_alpha_terms(pipe.scheduler, t_batch, device=device)

            t_prev_batch = torch.full((x.shape[0],), int(t_prev.item() if torch.is_tensor(t_prev) else t_prev), device=device, dtype=torch.long)
            a_prev, b_prev = _get_alpha_terms(pipe.scheduler, t_prev_batch, device=device)

                                                    
            x0_pred = (x - b_t * eps_pred) / a_t
                                                 
            x = a_prev * x0_pred + b_prev * eps_pred
                                                                  

            if not args.skip_visualizations and (i % max(1, args.save_every) == 0 or i == 1):
                try:
                    save_image(
                        latents_to_pil(pipe, x),
                        os.path.join(args.output_dir, f"inversion_seed_{s}_t_{int(t_prev)}.png"),
                    )
                except Exception:
                    pass

                                          
                                                              
        noise_scale = max(min(args.invert_noise_scale, 1.0), 0.0)                  
        latents_start = (1.0 - noise_scale) * latents_x0 + noise_scale * x

                                    
        if args.save_pure_noise or args.use_pure_noise_start:
            pure_noise = torch.randn(latents.shape, generator=generator, device=device, dtype=torch.float32)
            pure_noise = pure_noise * getattr(pipe.scheduler, "init_noise_sigma", 1.0)
            if args.save_pure_noise and not args.skip_visualizations:
                try:
                    save_image(latents_to_pil(pipe, pure_noise),
                               os.path.join(args.output_dir, f"pure_noise_seed_{s}.png"))
                except Exception:
                    pass
            if args.use_pure_noise_start:
                latents_start = pure_noise

                                        
        patcher.apply(time_t=float(args.t_denoise), adapter_strength=args.adapter_strength)

                                                      
        if args.gen_uncond:
            gen_prompt = ""                        
            gen_concept_tokens = None                     
        else:
            gen_prompt = class_prompt
            gen_concept_tokens = concept_tokens

        if use_concept:
            with torch.no_grad():
                gen_prompt_embeds, gen_pooled, gen_added_cond, gen_do_cfg = _prepare_sdxl_conditioning(
                    pipe,
                    gen_prompt,                                             
                    args.guidance_scale,
                    device=device,
                    dtype=pipe.unet.dtype,
                    crop_coords=crop_coords,
                    concept_tokens=gen_concept_tokens,                                  
                )
                pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
                latents_gen = latents_start.to(device=pipe.unet.device, dtype=torch.float32)
                for t in pipe.scheduler.timesteps:
                    t_batch = torch.full(
                        (latents_gen.shape[0],),
                        int(t.item() if torch.is_tensor(t) else t),
                        device=device,
                        dtype=torch.long,
                    )
                    eps_pred = _predict_eps(
                        pipe=pipe,
                        latents=latents_gen,
                        t=t_batch,
                        prompt_embeds=gen_prompt_embeds,
                        added_cond=gen_added_cond,
                        do_cfg=gen_do_cfg,
                        guidance_scale=args.guidance_scale,
                        cross_attention_kwargs={"scale": args.cross_attn_scale},
                    )
                    latents_gen = pipe.scheduler.step(
                        eps_pred,
                        t,
                        latents_gen,
                        generator=generator,
                    ).prev_sample
                gen_image = latents_to_pil(pipe, latents_gen)
        else:
            with torch.no_grad():
                images = pipe(
                    prompt=gen_prompt,                                          
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    latents=latents_start.to(device=pipe.unet.device, dtype=pipe.unet.dtype),
                    cross_attention_kwargs={"scale": args.cross_attn_scale},
                    original_size=(1024, 1024),
                    target_size=(1024, 1024),
                    crops_coords_top_left=tuple(map(int, crop_coords)),
                ).images
            gen_image = images[0]

        save_image(gen_image, args.output_image or os.path.join(args.output_dir, f"generated_seed_{s}.png"))

    patcher.reset()
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()


                                                                                                                                                                                                                                                           
