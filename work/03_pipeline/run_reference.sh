#!/usr/bin/env bash
set -euo pipefail

# Reference reconstruction for the official 4.0003 package.
# This script documents the stage order; paths must be adapted to the host.

ROOT=${ROOT:-/workspace}
FREQ_ROOT=${FREQ_ROOT:-$ROOT/FreqOrtho-SR}
TADSR_ROOT=${TADSR_ROOT:-$ROOT/TADSR-main}
PY=${PY:-python}

# Stage 1: quarter-scale inputs for FreqOrtho-SR.
$PY pipeline/prep_freqortho_full100.py

# Stage 2: FreqOrtho-SR one-step teacher generation.
cd "$FREQ_ROOT"
$PY test_pisasr.py \
  --pretrained_model_path preset/models/SD21 \
  --pretrained_path "$ROOT/weights/freqortho_sr.pkl" \
  --process_size 512 \
  --upscale 4 \
  --input_image "$ROOT/FreqOrtho-SR/full100_in" \
  --output_dir "$ROOT/FreqOrtho-SR/full100_out" \
  --default \
  --prompt ""

# Stage 3: Gaussian low-pass teacher conditioning + TADSR a15 4K regeneration.
cd "$TADSR_ROOT"
$PY pipeline/run_teacher_freqortho_full100.py \
  --cases $(seq 1 100) \
  --out "$ROOT/FreqOrtho-SR/full100_teacher" \
  --teacher "$ROOT/FreqOrtho-SR/full100_out"

# Stage 4: 1024px paired comparison and top-5 selection.
cd "$ROOT"
$PY pipeline/compare_freq_teacher_1024.py
$PY pipeline/select_top5_from_compare.py \
  --compare "$ROOT/FreqOrtho-SR/freq_teacher_vs_plus3_full100_1024.json" \
  --top-k 5 \
  --out "$ROOT/FreqOrtho-SR/selected_top5.json"
CASES=$($PY -c "import json;print(','.join(map(str,json.load(open('$ROOT/FreqOrtho-SR/selected_top5.json'))['selected_cases'])))")

# Stage 5: replace only the selected cases in the official opt7 4.0002 package.
$PY pipeline/build_freq_teacher_package.py \
  --base "$ROOT/package/opt7_remote_base.zip" \
  --teacher-dir "$ROOT/FreqOrtho-SR/full100_teacher" \
  --cases "$CASES" \
  --out "$ROOT/package/freqortho_teacher_top5_1024_on_opt7_v1.zip" \
  --validation "$ROOT/package/freqortho_teacher_top5_1024_on_opt7_v1_validation.json"