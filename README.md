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
    adversarial.py                # EEG-Eye encoders, SDE/flow generator, discriminators
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
- Generator: conditional latent generator with either VP-SDE noise prediction or Rectified Flow Matching.
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
  --stage1_epochs 10 \
  --stage2_epochs 10 \
  --stage3_epochs 40 \
  --generation_objective sde \
  --diffusion_sampler sde \
  --fusion_type slot_transformer \
  --eval_mode denoise \
  --eval_interval 5
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
Sample-mode completion uses the stochastic VP-SDE sampler by default. DDIM is
kept for SDE ablation with `--diffusion_sampler ddim --ddim_eta 0.0`.
Use `--generation_objective flow` to train the missing-modality generator with
Rectified Flow Matching instead of VP-SDE noise prediction. In that mode the
same U-Net predicts a velocity field from random latent noise to the target
latent, and `sample` integrates that velocity field directly.
Fusion uses a fixed slot-complete Transformer interface by default:
`[EEG slot, Eye slot, Availability slot]`. Missing modality slots are filled
with generated completion latents before fusion. Use `--fusion_type mlp` only for the
older concatenation MLP ablation.
Stage 3 also uses a lightweight sample-consistency objective by default in the
latter half of the stage: `--lambda_sample_cls 0.2 --lambda_sample_distill 0.05
--sample_consistency_interval 5`. It periodically runs the target-free `sample`
path and trains the fusion/classification head to handle sampled completions
without backpropagating through the generative sampler.
Full-modality teacher semantic alignment is enabled by default in stage 3 with
conservative weights: `--lambda_teacher_kd 0.05
--lambda_teacher_fusion 0.02 --lambda_teacher_proto 0.02`. Pass
`--teacher_checkpoint path/to/best_model.pth`
to use a separately trained, frozen full-modality teacher; otherwise the current
model's full EEG+Eye path is used as a self-teacher. The completed path is
trained to match teacher logits, fusion direction, and prototype distribution.
`--missing_eeg_semantic_weight 1.3` and `--missing_eye_semantic_weight 1.1` put
extra weight on stronger missing-modality cases, especially Eye-to-EEG
completion. For Flow Matching, stage-3 sample consistency uses a short
differentiable sampling path by default
(`--differentiable_sample_consistency --differentiable_sample_steps 2`), so
sample semantic losses can update the generator instead of only the fusion head.
Completed fusion representations also receive task-boundary regularization by
default: `--lambda_supcon 0.05` applies supervised contrastive learning within
the batch, and `--lambda_proto_margin 0.05 --proto_margin 0.2` enforces a margin
between the correct class prototype and the hardest negative prototype. The
sample path has lighter matching terms, `--lambda_sample_supcon 0.02` and
`--lambda_sample_proto_margin 0.02`.
Sampling-time self-conditioning is available with `--self_conditioning_sample`;
it feeds the previous predicted target latent back into the next SDE/DDIM/flow
sampling step condition.
Sample-mode fusion can estimate completion uncertainty by drawing multiple
stochastic completions and down-weighting high-variance generated latents:
`--uncertainty_samples 3`. The default `--uncertainty_samples 1` preserves the
single-sample behavior and does not add sampling cost.

To evaluate one checkpoint under multiple missing-modality settings:

```bash
python3 scripts/evaluate_checkpoint.py \
  --checkpoint runs/eeg_eye_adv/best_model.pth \
  --missing_modes random,missing_eeg,missing_eye \
  --missing_rates 0.3,0.5 \
  --eval_modes sample,denoise \
  --generation_objective sde \
  --diffusion_sampler sde \
  --uncertainty_samples 3 \
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
