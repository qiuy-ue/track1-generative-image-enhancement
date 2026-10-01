# 4.0003 Reconstruction Runbook

## 0. Directory layout expected at runtime

```text
ROOT/
  FreqOrtho-SR/
  TADSR-main/
  weights/
  pipeline/
  config/
  package/
```

Place `freqortho_sr.pkl` under `FreqOrtho-SR/preset/models/` or pass its absolute path with `--pretrained_path`.

Place the TADSR base download under `weights/TADSR` and point `run_teacher_freqortho_full100.py` to it by editing `cfg()`:

```python
a.pretrained_model_name_or_path = "/path/to/weights/TADSR"
a.tadsr_path = "/path/to/weights/tadsr_imag2006_a15.pkl"
```

## 1. Prepare quarter-resolution inputs

```bash
python pipeline/prep_freqortho_full100.py
```

The original script expects `/root/rivermind-data/full100_lq` and writes `/root/FreqOrtho-SR2/full100_in`. Change these paths at the top of the script for a new host.

## 2. Patch and run FreqOrtho-SR

```bash
python pipeline/patch_freqortho_prompt.py
python pipeline/patch_freqortho_skip.py

cd freqortho
python test_pisasr.py \
  --pretrained_model_path preset/models/SD21 \
  --pretrained_path /path/to/weights/freqortho_sr.pkl \
  --process_size 512 \
  --upscale 4 \
  --input_image /path/to/full100_in \
  --output_dir /path/to/full100_out \
  --default \
  --prompt ""
```

The final command used one-step default inference. The two patches add the `--prompt` argument and skip already generated files.

## 3. Regenerate native-4K teacher images with TADSR a15

```bash
cd ../tadsr
python ../pipeline/run_teacher_freqortho_full100.py \
  --cases 1 2 3 4 5 6 7 8 9 10 \
  --out /path/to/full100_teacher \
  --teacher /path/to/full100_out
```

The full run used `--cases 1 ... 100`, timestep `500` for normal cases and `800` for the scene list in `config/scene_4k.json`. It calls `tadsr_tilefix.TADSR_test`, loads `lora_svd_merge_fine/tadsr_imag2006_a15.pkl`, uses Gaussian blur radius `3.0` on the teacher, and applies ADain color correction.

## 4. Select top five cases

```bash
python pipeline/compare_freq_teacher_1024.py
python pipeline/select_top5_from_compare.py \
  --compare config/freq_teacher_vs_plus3_full100_1024.json \
  --top-k 5 \
  --out config/selected_top5.json
```

The selector computes:

```text
utility = 1.10286 * d_nr4 + 0.21747 * d_edge
```

The official top five were `44, 60, 97, 79, 51`.

## 5. Rebuild the submission ZIP

```bash
python pipeline/build_freq_teacher_package.py \
  --base /path/to/opt7_remote_base.zip \
  --teacher-dir /path/to/full100_teacher \
  --cases 44,60,97,79,51 \
  --out package/freqortho_teacher_top5_1024_on_opt7_v1.zip \
  --validation package/freqortho_teacher_top5_1024_on_opt7_v1_validation.json
```

Only the five selected entries are replaced. The other 95 entries come from the official opt7 4.0002 package and remain byte-identical.