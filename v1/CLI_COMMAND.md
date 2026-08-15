# Stage 1 v3 and Stage 2 high-level CLI

This document is the reproducible command reference for the completed Stage 1
low-level residual-guidance training and the audited Stage 2 high-level PPO
training transition.

The accepted artifacts are under `outputs/stage1_v3_batch64/`. The three A2
seeds are `20260703`, `20260713`, and `20260723`; the final selected checkpoint
is from seed `20260713`. Keep all periodic checkpoints when preserving the
complete training record.

## 0. Environment and GPU

Use the `612` conda environment. Check the physical GPU before a CUDA run; the
recorded training used GPU, exposed to the process as `cuda:0`.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

## 1. Completed v3 training

The following is the authoritative sequential command used for the three A2
seeds.

```bash
set -euo pipefail

GPU=9
SELECTED_EXPERIMENT=A2
LAMBDA_LOW=1.0
LOW_POTENTIAL_WEIGHT=1

for SEED in 20260703 20260713 20260723
do
  OUT="outputs/stage1_v3_batch64/${SELECTED_EXPERIMENT}/seed_${SEED}"
  if [ -e "${OUT}" ]; then
    echo "Refusing to overwrite existing Stage 1 output: ${OUT}" >&2
    exit 2
  fi
  mkdir -p "${OUT}"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src \
  conda run -n 612 --no-capture-output \
  python -m red_swarm_policy.train_env \
    --device cuda:0 --seed "${SEED}" \
    --training-mode low_only --iterations 80 \
    --low-critic-warmup-updates 5 \
    --low-critic-warmup-critic-steps-per-update 32 \
    --parallel-envs 64 --parallel-backend process \
    --env-worker-threads 1 --env-worker-timeout-s 300 \
    --rollout-steps 64 \
    --assignment-sequence-length 32 --execution-sequence-length 128 \
    --ppo-epochs 4 --critic-updates-per-actor 2 --actor-update-interval 1 \
    --assignment-actor-learning-rate 1e-4 \
    --execution-actor-learning-rate 5e-5 \
    --assignment-critic-learning-rate 3e-4 \
    --execution-critic-learning-rate 3e-4 \
    --assignment-clip-epsilon 0.10 --execution-clip-epsilon 0.20 \
    --assignment-target-kl 0.01 --execution-target-kl 0.01 \
    --assignment-entropy-coef 0.01 --execution-entropy-coef 0.001 \
    --execution-action-distribution radial_tanh_disk \
    --critic-value-head-mode scalar \
    --execution-reward-learning-scale 0.001953125 \
    --execution-value-loss huber --execution-value-huber-delta 1.0 \
    --execution-advantage-normalization per_scenario \
    --execution-actor-loss-weighting per_scenario \
    --execution-post-step-kl-rollback \
    --execution-post-step-kl-limit 0.01 \
    --execution-lr-plateau-patience 3 \
    --execution-lr-plateau-factor 0.5 \
    --execution-min-actor-learning-rate 5e-6 \
    --execution-restore-best-on-lr-reduction \
    --execution-restore-best-on-early-stop \
    --early-stop-validation-patience 8 \
    --gamma-low 1 --lambda-low "${LAMBDA_LOW}" \
    --red-counts 1,2,3,4 --blue-counts 1 --styles many_to_one \
    --red-count-batch-mode stratified --max-missiles-per-target 4 \
    --time-step-s 0.005 --bias-update-interval-s 0.1 \
    --assignment-update-interval-s 5.0 --max-steps 36000 \
    --missile-max-guidance-time-s 180 --missile-boost-time-s 7 \
    --blue-policy rule --blue-rule-decision-interval-s 0.1 \
    --blue-rule-detection-range-m 60000 \
    --high-damage-weight 512 --high-waste-weight 64 \
    --high-potential-weight 1 --high-time-penalty-per-s 2 \
    --terminal-success-reward 0 --terminal-failure-penalty 0 \
    --terminal-timeout-penalty 0 \
    --low-damage-weight 512 --low-potential-weight "${LOW_POTENTIAL_WEIGHT}" \
    --low-missile-failure-penalty 64 \
    --low-load-penalty 0.0008 --low-smooth-penalty 0.0002 \
    --low-time-credit-mode terminal_active_share --low-time-weight 2 \
    --low-option-boundary-potential terminal_zero \
    --validation-interval 5 --validation-seed-start 20261000 \
    --validation-trials-per-blue-count 100 --validation-parallel-envs 100 \
    --latest-checkpoint "${OUT}/stage1_low_latest.pt" \
    --best-checkpoint "${OUT}/stage1_low_best.pt" \
    --checkpoint-interval 5 \
    --metrics-path "${OUT}/stage1_low_metrics.json" \
    --run-manifest-path "${OUT}/run_manifest.json" \
    | tee "${OUT}/stage1_low_training.jsonl"
done
```

## 2. Resume a Stage 1 run

`REMAINING_UPDATES` is the number of additional optimizer updates. The resume
checkpoint restores reward, estimator, scheduler, and validation settings.

```bash
set -euo pipefail

GPU=9
SELECTED_EXPERIMENT=A2
SEED=20260713
REMAINING_UPDATES=10
OUT="outputs/stage1_v3_batch64/${SELECTED_EXPERIMENT}/seed_${SEED}"

CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m red_swarm_policy.train_env \
  --device cuda:0 --training-mode low_only \
  --iterations "${REMAINING_UPDATES}" \
  --parallel-envs 64 --parallel-backend process \
  --env-worker-threads 1 --env-worker-timeout-s 300 \
  --execution-restore-best-on-lr-reduction \
  --resume-checkpoint "${OUT}/stage1_low_latest.pt" \
  --latest-checkpoint "${OUT}/stage1_low_latest.pt" \
  --best-checkpoint "${OUT}/stage1_low_best.pt" \
  --checkpoint-interval 5 \
  --metrics-path "${OUT}/stage1_low_resume_metrics.json" \
  --run-manifest-path "${OUT}/run_manifest_resume.json" \
  | tee -a "${OUT}/stage1_low_training.jsonl"
```

## 3. Paired zero-residual PN baseline

Run once for the independent holdout seed schedule. The baseline is retained at
`outputs/stage1_v3_batch64/pn_baseline/holdout_100_seed_20271000/`.

```bash
set -euo pipefail

HOLDOUT_SEED_START=20271000
BASELINE_OUT="outputs/stage1_v3_batch64/pn_baseline/holdout_100_seed_${HOLDOUT_SEED_START}"
mkdir -p "${BASELINE_OUT}"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src \
  conda run -n 612 --no-capture-output \
  python -m red_swarm_policy.validate_stage1_zero_pn \
    --trials-per-scenario 100 --red-counts 1,2,3,4 \
    --workers 12 --seed-start "${HOLDOUT_SEED_START}" \
    --output-dir "${BASELINE_OUT}"
```

## 4. Deterministic Stage 1 holdout gate

Every best checkpoint is evaluated against the same 400 paired trials. Exit 0
means the strict success/ineffective-loss gate passed; exit 3 means artifacts
were written but the quality gate failed; other nonzero codes indicate an
execution failure.

```bash
set -euo pipefail

HOLDOUT_SEED_START=20271000
BASELINE_OUT="outputs/stage1_v3_batch64/pn_baseline/holdout_100_seed_${HOLDOUT_SEED_START}"

for SEED in 20260703 20260713 20260723
do
  OUT="outputs/stage1_v3_batch64/A2/seed_${SEED}"
  HOLDOUT_OUT="${OUT}/holdout_100_seed_${HOLDOUT_SEED_START}"
  mkdir -p "${HOLDOUT_OUT}"

  set +e
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=src \
    conda run -n 612 --no-capture-output \
    python -m red_swarm_policy.validate_stage1_low_checkpoint \
      --checkpoint "${OUT}/stage1_low_best.pt" \
      --trials-per-scenario 100 --red-counts 1,2,3,4 \
      --workers 12 --seed-start "${HOLDOUT_SEED_START}" \
      --pn-baseline-summary "${BASELINE_OUT}/stage1_zero_pn_summary.json" \
      --output-dir "${HOLDOUT_OUT}" \
      2>&1 | tee "${HOLDOUT_OUT}/stage1_low_checkpoint_validation.log"
  STATUS=${PIPESTATUS[0]}
  set -e

  case "${STATUS}" in
    0) echo "Seed ${SEED}: holdout quality gate PASSED" ;;
    3) echo "Seed ${SEED}: quality gate FAILED" >&2; exit 3 ;;
    *) echo "Seed ${SEED}: evaluator failed with exit code ${STATUS}" >&2; exit "${STATUS}" ;;
  esac
done
```

## 5. Validate a retained checkpoint directly

For a lightweight trajectory/metrics check outside the formal Stage 1 gate:

```bash
set -euo pipefail

PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m red_swarm_policy.validate_checkpoint \
  --checkpoint outputs/stage1_v3_batch64/A2/seed_20260713/stage1_low_best.pt \
  --device cpu --trials 10 --style many_to_one --red-count 4 --blue-count 1 \
  --blue-policy rule \
  --metrics-path outputs/stage1_v3_batch64/checkpoint_validation_metrics.json \
  --trials-csv outputs/stage1_v3_batch64/checkpoint_validation_trials.csv
```

## 6. Required Stage 2 preflight

Stage 2 is bound to the pre-registered seed `20260713` Stage 1 best checkpoint
and its independent deterministic holdout gate. Do not reselect the Stage 1
seed from holdout results. Run this block from the repository root immediately
before starting training. It checks the exact checkpoint hash, the gate, the
current test suite, and the selected physical GPU.

```bash
set -euo pipefail

GPU=9
STAGE1_CHECKPOINT="outputs/stage1_v3_batch64/A2/seed_20260713/stage1_low_best.pt"
STAGE1_GATE="outputs/stage1_v3_batch64/A2/seed_20260713/holdout_100_seed_20271000/stage1_low_checkpoint_summary.json"
EXPECTED_CHECKPOINT_SHA256="29396d32fb9ed75a4531c56888e3653aa89a1ce52d7a8fe0423d7cd18272ac6a"

test -f "${STAGE1_CHECKPOINT}"
test -f "${STAGE1_GATE}"
test "$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')" = "${EXPECTED_CHECKPOINT_SHA256}"

jq -e --arg expected "${EXPECTED_CHECKPOINT_SHA256}" '
  .evaluation == "stage1_low_checkpoint_residual_guidance" and
  .stage1_quality_gate.schema_version == 1 and
  .stage1_quality_gate.policy_mode == "deterministic" and
  .stage1_quality_gate.passed == true and
  .stage1_quality_gate.checkpoint_sha256 == $expected and
  .stage1_quality_gate.required_red_counts == [1, 2, 3, 4] and
  .stage1_quality_gate.evaluated_red_counts == [1, 2, 3, 4] and
  (.stage1_quality_gate.runtime_validity | all(. == true)) and
  (.stage1_quality_gate.by_scenario | all(.passed == true))
' "${STAGE1_GATE}" >/dev/null

PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m pytest -q

nvidia-smi -i "${GPU}" \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

GPU_MEMORY_USED_MIB=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')
if [ "${GPU_MEMORY_USED_MIB}" -gt 1024 ]; then
  echo "Refusing to start: GPU ${GPU} already uses ${GPU_MEMORY_USED_MIB} MiB" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
conda run -n 612 --no-capture-output \
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The first Stage 2 process also repeats the full gate validation internally and
refuses a mismatched checkpoint. The transition must include both
`--stage1-quality-gate` and `--reset-best-on-resume`; these prevent an unverified
low-level policy or the Stage 1 best-score/scheduler state from entering Stage 2.

## 7. Final Stage 2 v2 P1 high-level PPO training

This is the authoritative Stage 2 command used for the retained model: one
pre-registered Stage 1 source, Stage 2 seed `20260810`, 48 parallel environments,
and at most 50 high-level updates. The frozen low-level actor and critic are
restored exactly and execution remains deterministic during high-level training.

The `low_only` to `high_only` transition starts a new stage-local timeline and
is the only full-resume path allowed to override the previously untrained
assignment reward scale and high-level terminal-success reward:
Stage 2 iteration, optimizer-update, and policy-update counters start at zero,
so the first completed update is `iteration=1` and the first periodic file is
`iteration_000005.pt`. The Stage 1 source counters remain recorded under
`stage_origin`; resuming from a Stage 2 latest checkpoint continues the existing
Stage 2 counters instead of resetting them again.

`rollout-steps=64` is an upper bound in high-level decision events, not seconds:
periodic 5 s decisions and immediate damage/failure redecisions both consume a
slot. The final run stopped at iteration 45 by validation patience; its retained
best checkpoint was selected at iteration 25. `GPU=9` records the GPU used by
this run; change it only after checking the selected physical GPU is idle.

```bash
cd /home/data/heyuxin/612/0810/v1.1
set -euo pipefail

GPU=9
SEED=20260810
STAGE2_UPDATES=50
STAGE1_CHECKPOINT="outputs/stage1_v3_batch64/A2/seed_20260713/stage1_low_best.pt"
STAGE1_GATE="outputs/stage1_v3_batch64/A2/seed_20260713/holdout_100_seed_20271000/stage1_low_checkpoint_summary.json"
STAGE2_ROOT="outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713"
OUT="${STAGE2_ROOT}/seed_${SEED}"

test -f "${STAGE1_CHECKPOINT}"
test -f "${STAGE1_GATE}"

GPU_USED_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MIB}" -gt 512 ]; then
  echo "GPU ${GPU} is not idle: ${GPU_USED_MIB} MiB already used" >&2
  exit 2
fi
if [ -e "${OUT}" ]; then
  echo "Refusing to overwrite existing Stage 2 output: ${OUT}" >&2
  exit 2
fi
mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m red_swarm_policy.train_env \
  --device cuda:0 --seed "${SEED}" \
  --training-mode high_only --iterations "${STAGE2_UPDATES}" \
  --resume-checkpoint "${STAGE1_CHECKPOINT}" \
  --stage1-quality-gate "${STAGE1_GATE}" \
  --reset-best-on-resume \
  --parallel-envs 48 --parallel-backend process \
  --env-worker-threads 1 --env-worker-timeout-s 600 \
  --rollout-steps 64 \
  --red-counts 24 --blue-counts 4,5,6 \
  --styles many_to_many --scenario-sampling random \
  --red-count-batch-mode stratified \
  --terminal-success-reward 512 \
  --assignment-reward-learning-scale 0.001953125 \
  --assignment-entropy-coef 0.001 \
  --assignment-stickiness-logit-bonus 1.0 \
  --high-potential-weight 512 \
  --assignment-lr-plateau-patience 2 \
  --assignment-lr-plateau-factor 0.5 \
  --assignment-min-actor-learning-rate 1e-5 \
  --assignment-restore-best-on-lr-reduction \
  --assignment-restore-best-on-early-stop \
  --early-stop-validation-patience 4 \
  --validation-interval 5 --validation-seed-start 20262000 \
  --validation-trials-per-blue-count 32 \
  --validation-parallel-envs 32 \
  --latest-checkpoint "${OUT}/stage2_high_latest.pt" \
  --best-checkpoint "${OUT}/stage2_high_best.pt" \
  --checkpoint-interval 5 \
  --metrics-path "${OUT}/stage2_high_metrics.json" \
  --run-manifest-path "${OUT}/run_manifest.json" \
  | tee "${OUT}/stage2_high_training.jsonl"
```

The transition retains the accepted low-level actor and scalar critic bitwise,
rebuilds the previously untrained high-level critic as a five-component
latent-sum critic, sets the high-level critic learning targets to reward/512,
and adds a 512 terminal bonus for full mission success. P1 also uses lower
assignment entropy, target stickiness, CPU-streamed rollout storage, and stronger
high-level potential shaping. Assignment LR reduction and early stop both restore
the validated-best assignment actor and Adam state; the frozen execution
actor/critic are never routed through this scheduler.

## 8. Resume an interrupted Stage 2 seed

Resume only from that seed's Stage 2 latest checkpoint. Do not use the Stage 1
checkpoint again, do not pass `--reset-best-on-resume`, and do not pass the
Stage 1 gate on a same-stage resume; the Stage 2 optimizer, RNG, best score, and
validation/scheduler state must continue intact. Reward, assignment learning
scale, LR scheduling, restore, and early-stop settings are inherited from the
Stage 2 checkpoint and therefore are not repeated below. Increment `RESUME_TAG`
for each attempt.

```bash
cd /home/data/heyuxin/612/0810/v1.1
set -euo pipefail

GPU=9
SEED=20260810
REMAINING_UPDATES=10
RESUME_TAG=resume_001
STAGE2_ROOT="outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713"
OUT="${STAGE2_ROOT}/seed_${SEED}"

test -f "${OUT}/stage2_high_latest.pt"
if [ -e "${OUT}/stage2_high_${RESUME_TAG}_metrics.json" ]; then
  echo "Refusing to overwrite resume record: ${RESUME_TAG}" >&2
  exit 2
fi

GPU_USED_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MIB}" -gt 512 ]; then
  echo "GPU ${GPU} is not idle: ${GPU_USED_MIB} MiB already used" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m red_swarm_policy.train_env \
  --device cuda:0 --seed "${SEED}" \
  --training-mode high_only --iterations "${REMAINING_UPDATES}" \
  --resume-checkpoint "${OUT}/stage2_high_latest.pt" \
  --parallel-envs 48 --parallel-backend process \
  --env-worker-threads 1 --env-worker-timeout-s 600 \
  --rollout-steps 64 \
  --red-counts 24 --blue-counts 4,5,6 \
  --styles many_to_many --scenario-sampling random \
  --red-count-batch-mode stratified \
  --validation-interval 5 --validation-seed-start 20262000 \
  --validation-trials-per-blue-count 32 \
  --validation-parallel-envs 32 \
  --latest-checkpoint "${OUT}/stage2_high_latest.pt" \
  --best-checkpoint "${OUT}/stage2_high_best.pt" \
  --checkpoint-interval 5 \
  --metrics-path "${OUT}/stage2_high_${RESUME_TAG}_metrics.json" \
  --run-manifest-path "${OUT}/run_manifest_${RESUME_TAG}.json" \
  | tee -a "${OUT}/stage2_high_training.jsonl"
```

## 9. Required first-update checks

Before leaving a new seed unattended, inspect its first `event=iteration` JSON
row. It must report `iteration=1`, `completed_optimizer_updates=1`,
`completed_policy_updates=1`, `completed_stage_policy_updates=1`, and
`training_mode=high_only`. Assignment actor/critic updates must be greater than
zero, execution actor/critic updates must equal zero, and losses and assignment
KL must be finite. If assignment KL reaches the configured `0.01` threshold,
`assignment_kl_stopped` must be true so further actor epochs are stopped. On
every fifth Stage 2 update, `fixed_validation` must contain separate 24v4, 24v5,
and 24v6 rows and `stage2_high_best.pt` must be selected only by the five-level
lexicographic task metric.

```bash
nvidia-smi -i 9
tail -f outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/seed_20260810/stage2_high_training.jsonl
```

## 10. Final deterministic Stage 2 validation

The deployable path requires both the assignment actor and the frozen execution
actor to be deterministic. Do not pass `--validation-assignment-stochastic` or
`--validation-policy-seed`. The accepted independent block uses environment seed
`20263000`; set `ENV_SEED=20262000` only when deliberately repeating the training
selection block. The command refuses to overwrite a retained validation result.

```bash
cd /home/data/heyuxin/612/0810/v1.1
set -euo pipefail

GPU=9
ENV_SEED=20263000
ROOT="outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713"
CKPT="${ROOT}/seed_20260810/stage2_high_best.pt"
OUT="${ROOT}/analysis/assignment_deterministic_env_${ENV_SEED}"
EXPECTED_SHA256="ac210d751f1ae226ca6681a201eed09cc867f2f6822f7d604c5f8d9f9ac85cbb"

test -f "${CKPT}"
test "$(sha256sum "${CKPT}" | awk '{print $1}')" = "${EXPECTED_SHA256}"
if [ -e "${OUT}" ]; then
  echo "Refusing to overwrite existing validation output: ${OUT}" >&2
  exit 2
fi

GPU_USED_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MIB}" -gt 512 ]; then
  echo "GPU ${GPU} is not idle: ${GPU_USED_MIB} MiB already used" >&2
  exit 2
fi
mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src \
conda run -n 612 --no-capture-output \
python -m red_swarm_policy.train_env \
  --device cuda:0 --seed 20260810 \
  --training-mode high_only --iterations 0 \
  --validation-only --resume-checkpoint "${CKPT}" \
  --parallel-envs 1 --parallel-backend process \
  --env-worker-threads 1 --env-worker-timeout-s 600 \
  --red-counts 24 --blue-counts 4,5,6 \
  --styles many_to_many \
  --validation-assignment-mode actor \
  --validation-seed-start "${ENV_SEED}" \
  --validation-trials-per-blue-count 32 \
  --validation-parallel-envs 32 \
  --checkpoint "" --latest-checkpoint "" --best-checkpoint "" \
  --checkpoint-interval 0 \
  --metrics-path "${OUT}/validation_metrics.json" \
  --run-manifest-path "${OUT}/run_manifest.json" \
  | tee "${OUT}/validation.jsonl"

jq -e --arg expected "${EXPECTED_SHA256}" --argjson env_seed "${ENV_SEED}" '
  .event == "validation_only" and
  .checkpoint_sha256 == $expected and
  .policy_mode == "deterministic" and
  .assignment_policy_mode == "deterministic" and
  .execution_policy_mode == "deterministic" and
  .validation_policy_seed == null and
  .validation_config.seed_start == $env_seed and
  .trial_count == 96
' "${OUT}/validation_metrics.json" >/dev/null

jq '{checkpoint_sha256, policy_mode, assignment_policy_mode,
     execution_policy_mode, validation_policy_seed, trial_count,
     full_success_rate, average_damage_rate, ineffective_loss_rate,
     successful_completion_time_s, control_effort, by_scenario}' \
  "${OUT}/validation_metrics.json"
```

## 11. Retained Stage 2 result

- Final inference checkpoint: `stage2_high_best.pt`, iteration 25, SHA256
  `ac210d751f1ae226ca6681a201eed09cc867f2f6822f7d604c5f8d9f9ac85cbb`.
- Deterministic actor validation: `92/96` on environment seed `20262000` and
  `95/96` on independent environment seed `20263000`; pooled `187/192`.
- `stage2_high_latest.pt` is retained only for training-state audit or a safe
  same-stage resume. Inference and final validation must load the best checkpoint.
- Stochastic assignment is diagnostic-only and is not an accepted runtime mode.
