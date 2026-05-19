# EEG-Eye Adversarial Diffusion Completion

This project now focuses on incomplete multimodal emotion recognition with EEG and eye-tracking data. The original recommendation-system training stack has been removed so the remaining code centers on EEG-Eye missing-modality completion, adversarial alignment, and emotion classification.

## Data Modes

- `window`: reads `../Dataset/processed_eeg_eye_4s`
  - EEG: `[N, 5, 62]`
  - Eye: `[N, 33]`
  - Label: `[N]`
- `trial`: reads `../Dataset/seed-iv`
  - EEG: `[N, 10, 5, 62]`
  - Eye: `[N, 10, 31]`
  - Label: `[N]`

The loader creates missing-modality masks with shape `[N, 2]`, where columns are `[EEG, Eye]` and `1` means available.

## Repository Structure

```text
src/eeg_eye_completion/
  data.py                         # window/trial data loaders and missing masks
  train.py                        # training loop, metrics, checkpointing
  visualization.py                # metrics tables and plot generation
  models/
    adversarial.py                # EEG-Eye encoders, VP-SDE generator, discriminators
    diffusion.py                  # 1D U-Net and diffusion utilities
    transformers_encoder/         # attention blocks used inside the U-Net
scripts/
  train_emotion_adv.py            # command-line training entry point
  compare_runs.py                 # compare multiple result directories
figures/
  eeg_eye_adversarial_framework.svg
tools/
  inspect_feature_zips.py
```

## Model

The model is implemented in `src/eeg_eye_completion/models/adversarial.py`.

- EEG encoder: band attention, learnable channel graph, and trial-mode temporal GRU.
- Eye encoder: MLP for window mode and temporal Transformer for trial mode.
- Generator: conditional VP-SDE latent diffusion with a 1D U-Net score network.
- Discriminators:
  - `D_var`: local variable / band-channel realism.
  - `D_mod`: whole-modality realism.
  - `D_lat`: conditional latent-pair realism.
  - `D_fus`: fused semantic representation realism.
- Emotion head: classifier, learnable emotion prototypes, and fusion consistency.

## Quick Smoke Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/train_emotion_adv.py --smoke --data_mode window --missing_rate 0.5
PYTHONDONTWRITEBYTECODE=1 python3 scripts/train_emotion_adv.py --smoke --data_mode trial --missing_rate 0.5
```

Smoke runs now write `best_model.pth`, `last_model.pth`, and metrics to the selected `--output_dir`
unless `--no_save` is explicitly passed.

## Example Training

```bash
python3 scripts/train_emotion_adv.py \
  --data_mode window \
  --split_protocol predefined \
  --missing_mode random \
  --missing_rate 0.3 \
  --batch_size 64 \
  --stage1_epochs 5 \
  --stage2_epochs 5 \
  --stage3_epochs 20
```

For stricter protocols, replace the predefined split with subject/session splits:

```bash
python3 scripts/train_emotion_adv.py --data_mode window --split_protocol subject --test_ratio 0.2
python3 scripts/train_emotion_adv.py --data_mode window --split_protocol session --test_sessions 3
```

For modality-specific missing experiments:

```bash
python3 scripts/train_emotion_adv.py --data_mode window --missing_mode missing_eeg
python3 scripts/train_emotion_adv.py --data_mode window --missing_mode missing_eye
```

Training now evaluates with target-free diffusion sampling by default. Use
`--eval_mode denoise` for a deterministic one-step diagnostic, or
`--eval_mode teacher` only for the older stochastic one-step diagnostic.
Adversarial and consistency losses are warmed up by default, discriminators use
spectral normalization, and stage-3 early stopping monitors macro-F1.
The diffusion U-Net now uses `--unet_attention critical` by default, which keeps
attention only on the lower-resolution `down3,down4,up4,up3` layers. Use
`--unet_attention all` to reproduce the older full-attention U-Net.
Use `--eval_interval 5` to keep all training updates unchanged while running the
full test-set evaluation every five epochs instead of every epoch.

To evaluate one checkpoint under multiple missing-modality settings:

```bash
python3 scripts/evaluate_checkpoint.py \
  --checkpoint runs/eeg_eye_adv/best_model.pth \
  --missing_modes random,missing_eeg,missing_eye \
  --missing_rates 0.3,0.5 \
  --eval_modes sample,denoise \
  --output_dir runs/eeg_eye_adv/eval
```

## Ablation Switches

- `--no_modality_adv`
- `--no_latent_adv`
- `--no_fusion_adv`
- `--no_variable_adv`
- `--no_prototype`
- `--no_consistency`

Metrics printed during evaluation include accuracy, macro-F1, weighted-F1, EEG/Eye MSE, MAE, and cosine similarity.

## Result Artifacts

Each full training run writes the following files under `--output_dir`:

```text
best_model.pth
last_model.pth
metrics.csv
metrics.jsonl
summary.json
plots/
  classification_metrics.png
  training_losses.png
  generation_error.png
  generation_cosine.png
```

To compare multiple runs after training:

```bash
python3 scripts/compare_runs.py \
  runs/baseline_random_01 \
  runs/baseline_random_03 \
  runs/baseline_missing_eeg \
  runs/baseline_missing_eye \
  --output_dir runs/baseline_comparison
```

This creates `comparison.csv`, `comparison.json`, `recognition_comparison.png`, and `generation_comparison.png`.
