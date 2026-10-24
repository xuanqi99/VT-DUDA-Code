# Release validation

Prepared on 2026-10-24 from the supplied VT-DUDA implementation. The workspace
contained diffusion training/synthesis and ELS/MCC sources, with no existing Git
history, checkpoints, environment lockfile or per-task experiment configuration
files. Project-page files are kept in a separate ignored checkout and are not
part of the code release.

## Checks completed locally

| Check | Result |
| --- | --- |
| Python syntax / AST parsing | 69 source and test files passed |
| Real command-line entry points | All 8 `--help` commands passed |
| README executable script commands | 18 commands checked against real help options and files |
| Regression tests with PyTorch | 17 passed |
| Standard-library-only test run | 14 passed, 3 optional tensor tests skipped |
| Requirements resolution | All pinned direct dependencies satisfied in the verification environment |
| Generated-list loader smoke | PIL/DataLoader samples and Windows pickling passed for ELS/MCC |

Verification used Python 3.12.3 and the dependency versions in `requirements.txt`.
An isolated virtual environment reused the installed PyTorch stack and installed
only the missing small dependencies (`einops`, `tensorboard`, `peft`). A completely
fresh GPU installation has not been tested.

Tests cover exact generation-grid membership (50 DDIM plus 10 class variants),
token/noise argument routing, final-image paths, dry-run behavior and provenance,
500 Cartesian UDA pairs, parameter subsets, canonical labels, completed-output
validation, image-list preparation, VisDA path handling, epsilon checkpoint
separation and mocked conditioning with CPU tensors. The tensor checks exercise
the actual conditioning helper through mocked pipeline interfaces, including the
non-CFG DDIM inversion branch.

## Compatibility repairs

- Removed package imports of files absent from the supplied source; MCC keeps
  its existing timm backbone fallback and uses explicit image lists.
- Exposed dataset roots/domains without changing the original task defaults;
  supported the official Office-31 `domain/images/class` layout and corrected
  VisDA split-local list resolution.
- Corrected positive pooled embeddings in non-CFG conditioning and retained
  visual tokens in the manual prompt fallback.
- Reject missing visual-token checkpoints and infer their token count.
- Kept ELS epsilon values and training losses unchanged, while separating
  VisDA epsilon run directories and resolving evaluation checkpoints.

The generation grid is a post-fine-tuning addition. It preserves the original
DDIM latent interpolation formula and the original inference scales exposed by
the individual scripts. No model-training hyperparameter search was added to
diffusion fine-tuning.

## Limits

No model training, SDXL inference or benchmark evaluation was performed. No
datasets or model weights were downloaded. There are no supplied trained VT-DUDA
checkpoints or verified author checkpoint links, and the exact paper recipes and
original environment remain unavailable. Full custom-LoRA save/reload behavior
and final training completion have not been exercised. The optional legacy
`--train_text_encoder` API is absent in pinned Diffusers. Nondefault encoder
channel architectures and optional feature-analysis modules are unsupported or
unverified as described in the README.

The conservative DDIM launcher starts the original generator separately for
each reference/seed/setting. It has significant model-loading overhead and no
automatic resume for interrupted grids. ELS retains its default nine-epsilon
sweep unless the user explicitly selects an epsilon.

No new repository-wide license or unverified benchmark results are included.
