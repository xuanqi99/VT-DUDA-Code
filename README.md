# VT-DUDA: Visual Token Conditioning for Diffusion-guided Unsupervised Domain Adaptation

Official PyTorch implementation of VT-DUDA, **TMLR 2026**.

Xuan Qi, Daniele Berardini, Dario Serez, Vito Paolo Pastore, Vittorio Murino

[Paper](https://openreview.net/forum?id=Y956680PCe) | [Project Page](https://xuanqi99.github.io/VT-DUDA/)

VT-DUDA uses source-image visual tokens to condition diffusion models for target-style image synthesis. The generated images provide labeled training data for downstream unsupervised domain adaptation with ELS and MCC.

## Installation

Python 3.12 and PyTorch 2.5.1. Install the PyTorch build matching your CUDA version.

```bash
git clone https://github.com/xuanqi99/VT-DUDA-Code.git
cd VT-DUDA-Code
python -m pip install -r requirements.txt
```

## Datasets

| Dataset | `--dataset` | Data root | Domains |
| --- | --- | --- | --- |
| Office-31 | `office31` | `data/Office31` | `amazon`, `dslr`, `webcam` |
| Office-Home | `officehome` | `data/OfficeHome` | `Art`, `Clipart`, `Product`, `Real_World` |
| VisDA-2017 | `visda2017` | `data/VisDA2017` | `train`, `validation` |

Generation expects a source domain containing class folders. For the official Office-31 layout, use a path such as `data/Office31/webcam/images`.

UDA image lists contain one `image_path label` pair per line. Labels follow `CLASSES` in `UDA/ELS/common/vision/datasets/`. Typical target lists are `image_list/dslr.txt`, `image_list/Clipart.txt`, and `image_list/validation.txt`, respectively; paths are relative to the dataset root.

## Diffusion Training

Fine-tune the domain adapters and visual-token encoder. Example: Office-Home Art → Clipart.

```bash
python diffusion/train_synthesis_model.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset officehome --instance_data_dir data/OfficeHome \
  --source_domain Art --target_domain Clipart \
  --instance_prompt "An image" --output_dir checkpoints/Ar2Cl
```

For Office-31, set `--dataset office31` and the source/target domains from the table. VisDA uses `--dataset visda2017` with `train` → `validation`. Each fine-tuned checkpoint directory contains LoRA weights and `concept_encoder.pt`.

## Image Generation

After fine-tuning, generate target-style datasets with the same checkpoint:

```bash
python scripts/generate_grid.py \
  --checkpoint_dir checkpoints/Ar2Cl \
  --source_root data/OfficeHome/Art \
  --adapter_strengths 0.6 0.7 0.8 0.9 1.0 \
  --output_root outputs/Ar2Cl
```

| Pipeline | Token strength | Inversion noise level | Settings per adapter |
| --- | --- | --- | --- |
| DDIM inversion | `0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5` | `0.6 0.7 0.8 0.9 1.0` | 50 |
| Class-conditioned generation | `0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5` | — | 10 |

Each adapter strength has 50 DDIM inversion and 10 class-conditioned settings: **250 DDIM and 50 class-conditioned datasets** in total. Adapter strength scales LoRA weight updates during inversion and generation.

DDIM inversion uses source visual tokens; class-conditioned generation uses the class prompt together with source visual tokens. Noise level `1.0` uses the full DDIM inversion result.

Each DDIM setting translates all source images. Class-conditioned generation produces 50 images per class by default; use `--images_per_class 1000` for VisDA. Outputs are grouped under `adapter_0.6/` through `adapter_1.0/`, then by parameters and class. Settings are recorded in `generation_manifest.json`.

## Downstream UDA

Pair DDIM and class-conditioned datasets with the same adapter strength and token strength: **5 adapter strengths × 5 noise levels × 10 token strengths = 250 pairs per method and seed**. Each class-conditioned dataset is paired with the five DDIM noise levels at its matching strengths. Train ELS and MCC with:

```bash
python scripts/train_uda_grid.py \
  --generation-manifest outputs/Ar2Cl/generation_manifest.json \
  --dataset OfficeHome \
  --target-list data/OfficeHome/image_list/Clipart.txt \
  --target-image-root data/OfficeHome \
  --output-root outputs/uda_Ar2Cl \
  --methods els mcc --seeds 0 --run
```

Use `--dataset Office31` or `--dataset VisDA2017` with the corresponding target list for the other benchmarks. ELS retains its default epsilon sweep. To choose one epsilon, run ELS separately with `--methods els --train-args --eps 0.1`.

Use `--adapter_strengths 0.8` for a single generation group and `--adapter-strengths 0.8` for its UDA runs.

Each run's command and log directory are saved in `uda_manifest.json`. For evaluation, use that command with `--phase test` and the trained run's `--log` directory. For ELS, also select the matching `--eps` and epsilon subdirectory. VisDA reports per-class accuracy and mean accuracy.

## Licensing and third-party code

Original VT-DUDA contributions are released under the [MIT License](LICENSE) by Xuan Qi. Reused code retains its original copyright notices and license terms; see [THIRD_PARTY.md](THIRD_PARTY.md) and the [upstream license texts](licenses/). **ELS-specific code is used with permission**, following the same arrangement as [MUSE-Code](https://github.com/xuanqi99/MUSE-Code), and is not being relicensed as original VT-DUDA code. The SDXL pretrained model weights and benchmark datasets are governed by separate terms.

## Citation

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

We thank the authors of [Diffusers](https://github.com/huggingface/diffusers), [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library), [CLIP](https://github.com/openai/CLIP), and [guided-diffusion](https://github.com/openai/guided-diffusion) for their open-source implementations.
