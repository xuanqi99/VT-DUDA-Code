# VT-DUDA: Visual Token Conditioning for Diffusion-guided Unsupervised Domain Adaptation

Official PyTorch implementation for **Transactions on Machine Learning Research (TMLR), 2026**.

**Authors:** Xuan Qi, Daniele Berardini, Dario Serez, Vito Paolo Pastore, Vittorio Murino.

[Paper](https://openreview.net/forum?id=Y956680PCe) · [Project Page](https://xuanqi99.github.io/VT-DUDA/)

## Overview

VT-DUDA encodes source images into visual tokens and concatenates them with diffusion text embeddings. Source and target domain adapters are learned with an SDXL backbone. After fine-tuning, the target adapter synthesizes labeled target-style images through visual-token-conditioned DDIM inversion or class-conditioned generation from noise. The resulting datasets train downstream ELS or MCC adaptation models.

The original module layout is retained:

```text
diffusion/
  train_synthesis_model.py       # Domain adapters and visual-token encoder
  synthesize_from_reference.py   # DDIM inversion / translation
  synthesize_from_noise.py       # Class prompts + visual tokens, from noise
  ldm/                          # Encoder4 and attention/encoder dependencies
UDA/
  ELS/                          # ELS training, VisDA entry point and datasets
  MCC/                          # MCC training and bundled tllib subset
  generated_data.py             # Optional image-list inputs
scripts/
  generate_grid.py              # Post-fine-tuning generation grid
  train_uda_grid.py             # DDIM/class dataset combinations
tests/                         # Grid and conditioning checks
requirements.txt
```

## Installation

Use Python 3.12 and run commands from the repository root. `requirements.txt` matches the release import and CLI checks; it is not a recovered lockfile of the original paper environment.

```bash
git clone https://github.com/xuanqi99/VT-DUDA-Code.git
cd VT-DUDA-Code
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install an appropriate PyTorch 2.5.1 / torchvision 0.20.1 build following the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/generate_grid.py --help
python scripts/train_uda_grid.py --help
```

Training and SDXL synthesis require a suitable GPU environment and substantial device memory. Planner commands and grid tests need only the Python standard library; tensor tests skip when PyTorch is absent. Optional `xformers` and 8-bit optimizers are not required below.

## Datasets

Prepare images locally; datasets and generated collections are excluded from the repository. Use the bundled dataset `CLASSES` order, **not alphabetical folder order**.

| Dataset | Domain folders | UDA domain names | Label definition under `UDA/ELS/common/vision/datasets/` |
| --- | --- | --- | --- |
| Office-31 | `amazon/`, `dslr/`, `webcam/` | `A`, `D`, `W` | `office31.py`, 31 classes |
| Office-Home | `Art/`, `Clipart/`, `Product/`, `Real_World/` | `Ar`, `Cl`, `Pr`, `Rw` | `officehome.py`, 65 classes |
| VisDA-2017 | `train/`, `validation/` | `Synthetic`, `Real` | `visda2017.py`, 12 classes |

For generation, `--source_root` must directly contain `class_name/image` folders. With the official Office-31 layout use, for example, `data/Office31/webcam/images`; the diffusion trainer supports both `domain/class` and `domain/images/class`. For Office-Home fine-tuning, pass the actual folder name explicitly: the original default is `Real World`, while standard archives use `Real_World`.

Downstream training uses text image lists containing a path and integer label per line. Paths can be absolute or relative to the selected image root. The new list input supports spaces in paths. For Office-Home, for example:

```text
Clipart/Drill/example.jpg 0
Clipart/Exit_Sign/example.jpg 1
```

Standard lists are `image_list/amazon.txt`, `dslr.txt`, and `webcam.txt` for Office-31; `Art.txt`, `Clipart.txt`, `Product.txt`, and `Real_World.txt` for Office-Home; and `train.txt` and `validation.txt` for VisDA. VisDA also supports domain-local lists. Ensure labels agree with the supplied `CLASSES` definitions. Target labels are used for evaluation and ignored by the adaptation loss.

## Training

The diffusion trainer jointly learns the adapters and `Encoder4`. These commands use existing script defaults; complete per-task paper configuration files and training logs were not supplied.

Office-Home Art → Clipart:

```bash
python diffusion/train_synthesis_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --instance_prompt "An image" --dataset officehome --instance_data_dir data/OfficeHome --source_domain Art --target_domain Clipart --output_dir checkpoints/Ar2Cl
```

Office-31 Webcam → DSLR:

```bash
python diffusion/train_synthesis_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --instance_prompt "An image" --dataset office31 --instance_data_dir data/Office31 --source_domain webcam --target_domain dslr --output_dir checkpoints/W2D
```

VisDA-2017 Synthetic → Real:

```bash
python diffusion/train_synthesis_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --instance_prompt "An image" --dataset visda2017 --instance_data_dir data/VisDA2017 --output_dir checkpoints/visda
```

VisDA uses `train` → `validation`. See `python diffusion/train_synthesis_model.py --help` for optimization settings. Existing defaults include 20 visual tokens and 128 encoder channels. Checkpoints contain custom LoRA weights and `concept_encoder.pt`; keep both from the same fine-tuning run.

## Image Generation

**The hyperparameter sweep happens after fine-tuning.** Each source/target task reuses its fine-tuned checkpoint; the sweep does not retrain the diffusion model.

| Pipeline | Visual-token strengths | Inversion noise levels | Dataset variants |
| --- | --- | --- | --- |
| DDIM inversion, token conditioning | `0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5` | `0.6 0.7 0.8 0.9 1.0` | **10 × 5 = 50** |
| Class-conditioned generation from noise | `0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5` | Independent noise initialization | **10** |

DDIM token conditioning uses an empty text prompt plus source visual tokens. The class-conditioned branch uses a class text prompt **and scaled source visual tokens**. Token strength scales encoder outputs and is separate from classifier-free guidance and the existing cross-attention/LoRA scale.

The original DDIM script defines noise level through this latent blend:

```text
start = (1 - noise_level) * source_latent + noise_level * fully_inverted_latent
```

**1.0 is the full-noising endpoint of this inversion implementation**: it uses the complete DDIM inversion result. Intermediate values interpolate with the source latent. This parameter does not change the inversion step count; the endpoint is not independent Gaussian noise. The original formula is preserved.

Inspect all settings without loading models or generating images:

```bash
python scripts/generate_grid.py --checkpoint_dir checkpoints/Ar2Cl --source_root data/OfficeHome/Art --output_root outputs/Ar2Cl --dry_run
```

After reviewing `outputs/Ar2Cl/generation_manifest.json`, execute the same grid:

```bash
python scripts/generate_grid.py --checkpoint_dir checkpoints/Ar2Cl --source_root data/OfficeHome/Art --output_root outputs/Ar2Cl
```

```text
outputs/Ar2Cl/
  generation_manifest.json
  ddim_inversion/token_0.6/noise_0.6/<class>/*.png
  ...                           # all 50 strength/noise settings
  class_condition/token_0.6/<class>/*.png
  ...                           # all 10 strengths
```

These are **60 dataset variants**, not 60 image files. Each DDIM variant translates every source image once by default (`--reference_seeds 1`). Each class-conditioned variant has 50 generated images per class (`--images_per_class 50`). Use `--images_per_class 1000` for the VisDA inversion-free sample count described on the project page. The manifest records parameters and expected counts. Only final samples enter these dataset folders.

The conservative DDIM launcher reloads the original single-image pipeline for each reference/seed/configuration; large sweeps can be slow. Use `--pipeline`, `--token_strengths`, `--noise_levels`, or `--classes` for a subset. A different, completed, or interrupted grid needs a fresh output root; automatic resumption is not implemented. Source and output roots must be disjoint.

Single-setting commands are also supported; replace the example paths with real inputs:

```bash
python diffusion/synthesize_from_reference.py --base_model_id stabilityai/stable-diffusion-xl-base-1.0 --lora_path checkpoints/Ar2Cl --input_image data/OfficeHome/Art/Drill/example.jpg --concept_image data/OfficeHome/Art/Drill/example.jpg --no_prompt --concept_token_strength 0.6 --invert_noise_scale 1.0 --output_dir outputs/reference_example
python diffusion/synthesize_from_noise.py --base_model_id stabilityai/stable-diffusion-xl-base-1.0 --checkpoint_dir checkpoints/Ar2Cl --source_root data/OfficeHome/Art --concept_token_strength 0.6 --output_dir outputs/class_example
```

The token count is inferred from the checkpoint. Nondefault encoder channel architectures remain unverified.

## Downstream UDA Training

Combine **one of the 50 DDIM datasets with one of the 10 class-conditioned datasets**. The default Cartesian product gives **50 × 10 = 500 dataset pairs per method and seed**. Each pair is concatenated as the labeled source dataset; real target images remain the adaptation/evaluation domain.

Plan ELS and MCC combinations:

```bash
python scripts/train_uda_grid.py --generation-manifest outputs/Ar2Cl/generation_manifest.json --dataset OfficeHome --target-list data/OfficeHome/image_list/Clipart.txt --target-image-root data/OfficeHome --output-root outputs/uda_Ar2Cl --methods els mcc --seeds 0 --dry-run
```

Replace `--dry-run` with `--prepare-only` to validate completed generation outputs, check labels/counts and write lists. Replace it with `--run` to train sequentially. `uda_manifest.json` records each pair's parameters, source lists, seed, log directory and command. Preparation/training requires completed generation variants and all dataset classes.

To select four pairs, add these options before any `--train-args`:

```text
--ddim-token-strengths 0.6 1.0 --ddim-noise-levels 0.6 1.0 --class-token-strengths 0.8
```

Use `--dataset Office31` or `--dataset VisDA2017` and the appropriate real target list for other benchmarks. VisDA selects the ELS VisDA entry point and classwise evaluation.

**ELS's original default `--eps 1` runs nine epsilon values per dataset pair.** Thus 500 ELS launcher calls can produce 4,500 epsilon runs per seed. To explicitly choose one epsilon, run ELS separately and use, for example, `--methods els --train-args --eps 0.1`. This is an experiment choice; the original default is retained. Do not pass ELS-only flags to MCC. Other native options go after `--train-args`; the grid controls inputs, seeds and log locations.

The native trainers accept `--source-list` (one or more lists), `--target-list`, and `--target-image-root`. MCC's original named dataset loaders were absent from the supplied package, so use list inputs for MCC. Its existing timm backbone fallback is retained. ELS also retains its domain-based input path.

## Evaluation

Use the recorded native training command with the same dataset arguments and actual run directory. Append `--phase test --eps 0.1` for an ELS run trained with epsilon 0.1, or `--phase test` for MCC. Remove any conflicting `--phase train`. For ELS's default sweep, choose the matching `eps...` subdirectory.

After `--prepare-only` creates the lists, an individual MCC run and evaluation are:

```bash
python UDA/MCC/examples/domain_adaptation/image_classification/train_mcc.py data/OfficeHome -d OfficeHome -s grid_ddim grid_class -t real_target --source-list outputs/uda_Ar2Cl/image_lists/ddim_token_0.6_noise_0.6.txt outputs/uda_Ar2Cl/image_lists/class_token_0.6.txt --target-list outputs/uda_Ar2Cl/image_lists/real_target.txt --target-image-root data/OfficeHome --log outputs/mcc_example --seed 0
python UDA/MCC/examples/domain_adaptation/image_classification/train_mcc.py data/OfficeHome -d OfficeHome -s grid_ddim grid_class -t real_target --source-list outputs/uda_Ar2Cl/image_lists/ddim_token_0.6_noise_0.6.txt outputs/uda_Ar2Cl/image_lists/class_token_0.6.txt --target-list outputs/uda_Ar2Cl/image_lists/real_target.txt --target-image-root data/OfficeHome --log outputs/mcc_example --seed 0 --phase test
```

MCC saves `checkpoints/latest.pth` and `checkpoints/best.pth`. ELS saves its epsilon-specific best checkpoint in the run's `checkpoints/` directory. Evaluation prints target accuracy; VisDA supports per-class accuracies and their mean. Feature-analysis modes require modules absent from this source release and are explicitly unavailable.

Lightweight checks without datasets, checkpoints or training:

```bash
python -m compileall -q diffusion UDA scripts tests
python -m unittest discover -s tests -v
python diffusion/train_synthesis_model.py --help
python diffusion/synthesize_from_reference.py --help
python diffusion/synthesize_from_noise.py --help
python UDA/ELS/train_els.py --help
python UDA/ELS/train_els_visda.py --help
python UDA/MCC/examples/domain_adaptation/image_classification/train_mcc.py --help
```

## Pretrained Models

No trained VT-DUDA checkpoints or author-hosted checkpoint links were included. Fine-tune to obtain matching LoRA weights and `concept_encoder.pt`. Scripts use [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), subject to its provider's terms; a local SDXL directory can be supplied instead. Downstream backbones use their existing PyTorch/timm pretrained loaders.

Release validation covers syntax, imports, CLI entry points, grid planning, list preparation and mocked conditioning checks. **Full SDXL training, image synthesis, UDA training and paper accuracies have not been rerun.** No datasets or model weights were downloaded for validation. The legacy optional `--train_text_encoder` path uses an API absent from pinned Diffusers and is unsupported. The trainer's final pipeline reload/custom LoRA handling has not been validated through a full training run. Exact paper recipes, the original environment lockfile and trained checkpoints remain unavailable.

## Citation

Official BibTeX from the project page:

```bibtex
@article{qi2026vtduda,
  title={VT-DUDA: Visual Token Conditioning for Diffusion-guided Unsupervised Domain Adaptation},
  author={Qi, Xuan and Berardini, Daniele and Serez, Dario and Pastore, Vito Paolo and Murino, Vittorio},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=Y956680PCe}
}
```

## Acknowledgments

The implementation builds on [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and SDXL/DreamBooth utilities, [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library) components used by ELS/MCC, and latent-diffusion-style modules. Encoder sources retain references to [OpenAI CLIP](https://github.com/openai/CLIP) and [guided-diffusion](https://github.com/openai/guided-diffusion). Existing source attributions and upstream usage terms apply. No new repository-wide license has been assigned.
