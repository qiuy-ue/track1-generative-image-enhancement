# Clarity with Restraint, Enhancement under Control

**English** | [简体中文](README.md)

![Competition](https://img.shields.io/badge/2026_CSIG_Camera_Academic_Star-Track_1-2563EB)
![Stage](https://img.shields.io/badge/Stage-Finals_Presentation-7C3AED)
![Score](https://img.shields.io/badge/Prelim_Official_Record-4.0003-16A34A)
![Architecture](https://img.shields.io/badge/Architecture-Diffusion-DB2777)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9_/_2.0-EE4C2C)
![Hardware](https://img.shields.io/badge/End_to_end-Single_A100_40GB-0891B2)

> [!IMPORTANT]
> **In one sentence**: a two-stage generative enhancement pipeline built on **FreqOrtho-SR + TADSR** — FreqOrtho-SR preserves pixel structure at quarter resolution (FreqMoE frequency routing + OGP orthogonal gradient projection), then TADSR's time-aware one-step diffusion refines the result back to native 4K, and an interpretable weighted rule performs candidate selection. **Preliminary official record: 4.0003** (100/100 4K results packaged with strict naming and SHA-256), **end-to-end on a single A100 40GB**.
>
> **Where to look**: [Overview](#solution-overview) (two architecture diagrams) · [Evidence on five challenge scenarios](#how-controllability-lands-on-the-five-challenge-scenarios) · [End-to-end engineering loop](#end-to-end-engineering-loop)
>
> **How to run**: the full reproduction kit is archived in this repository under [`work/`](work/) — source, weights, inference scripts, pinned dual environments, and the official result package. The five-step reproduction lives in [work/RUNBOOK.md](work/RUNBOOK.md); environment reconstruction in [work/04_environment/ENVIRONMENT.md](work/04_environment/ENVIRONMENT.md). Only the two large base models (TADSR base and SD2.1-base) are not bundled — download notes in [work/00_README.md](work/00_README.md).

---

**2026 CSIG "Camera Academic Star" Imaging Algorithm Competition · Track 1: Controllability of Generative Image Enhancement · Finals solution**

## Results at a Glance

| Item | Result |
|------|--------|
| Preliminary official record | **4.0003** (100 / 100 4K results submitted and packaged, with SHA-256 checksums) |
| Official-record top-5 candidates | case **44 · 60 · 97 · 79 · 51** |
| FreqOrtho-SR · DRealSR benchmark | PSNR **29.08** / LPIPS **0.2799** (both improved simultaneously) |
| Hardware requirement | Entire pipeline on a **single A100 40GB** |

## The Challenge

The task requires enhancing 4K low-quality images with a **Diffusion-based** model, scored against hidden Ground Truth by a weighted combination of full-reference and no-reference metrics. The hard part is not upscaling — it is staying faithful while getting sharper.

**Four hard constraints:** clarity improvement · content consistency · resolution preservation · natural color and structure

**Five challenge scenarios:** small faces · text · dense vegetation · clocks · birds

## Solution Overview

```mermaid
flowchart LR
    A["01 Low-quality input<br/>4K test image"] --> B["02 Structure restoration<br/>FreqOrtho-SR<br/>quarter resolution"]
    B --> C["03 Detail teacher<br/>TADSR<br/>native-4K refinement"]
    C --> D["04 Compare & package<br/>candidate selection + strict naming<br/>case1…case100"]
```

One engineering throughline: **pixel structure is guarded by FreqMoE, semantic detail is supplied by TADSR, and the final output returns to a unified 4K geometry.**

### FreqOrtho-SR: Splitting "sharp" and "faithful" into controllable directions

Core idea: **learn pixel fidelity and semantic awareness separately, then use an orthogonality constraint to keep them from interfering with each other.**

- **FreqMoE** — routes FFT-based degradation features to pixel-level LoRA experts, first judging which kind of degradation the input resembles;
- **OGP (Orthogonal Gradient Projection)** — extracts the pixel subspace via SVD and projects the semantic-LoRA gradients onto its null space, so the two optimization directions never drag at each other.

![FreqOrtho-SR architecture: two-phase training with OGP](assets/freqortho_arch.png)

### TADSR: Guarding 4K detail with time-aware one-step diffusion

TADSR does not reinvent content — it **refines the coarsely restored result onto a more natural 4K surface**.

- Time-aware encoder + student model (UNet + LoRA);
- Frozen teacher model provides detail constraints (TAVSD distillation);
- High/low timesteps adapt to complex 4K scenes: **t=500 for ordinary cases, t=800 for complex cases flagged in scene_4k.json**.

![TADSR time-aware distillation framework](assets/tadsr_overview.png)

## End-to-End Engineering Loop

| Stage | Content |
|-------|---------|
| Prepare | Generate quarter-resolution inputs |
| Restore | FreqOrtho-SR one-step inference |
| Refine | TADSR teacher refinement to native 4K (t=500 / 800) |
| Compare | Candidate selection: `utility = 1.10286 × d_nr4 + 0.21747 × d_edge` |
| Package | Uniformly named output `output_dir/case1.jpg … case100.jpg` |

The selection weights come from a full-100 teacher-vs-candidate comparison experiment (`work/03_pipeline/config/freq_teacher_vs_plus3_full100_1024.json`) — an **interpretable, auditable** rule rather than a black-box score.

## Reproduction (work/)

The full reproduction kit is archived in this repository; directories mirror the original numbered bundles:

| Directory | Contents |
|-----------|----------|
| `work/01_freqortho_sr/` | Complete FreqOrtho-SR source + `freqortho_sr.pkl` weights (57.7MB) |
| `work/02_tadsr/` | TADSR source + LoRA weights (`tadsr_imag2006_a15.pkl` etc.) |
| `work/03_pipeline/` | Inference / comparison / selection / packaging scripts + config (`scene_4k.json` etc.) |
| `work/04_environment/` | Dual-environment dependency manifests (Conda + pip, incl. base-model download notes) |
| `work/05_official_package/` | Official 4.0003 submission package: `output_dir/case1.jpg … case100.jpg` |
| `work/06_defense/` | Defense materials |

**Five-step reproduction** (details in [work/RUNBOOK.md](work/RUNBOOK.md)):

1. Generate quarter-resolution inputs — `pipeline/prep_freqortho_full100.py`
2. Patch and run FreqOrtho-SR one-step inference (`--process_size 512 --upscale 4`)
3. TADSR teacher refinement back to native 4K — t=500 for ordinary cases, t=800 for complex cases in `scene_4k.json`; teacher Gaussian blur r=3.0 + ADain color correction
4. Candidate selection `utility = 1.10286 × d_nr4 + 0.21747 × d_edge` — official top-5: **44 / 60 / 97 / 79 / 51**
5. Rebuild the submission package — only the 5 selected entries are replaced; the other 95 remain byte-identical to the opt7 4.0002 baseline

**Environments** (two separate environments, pinned versions, see [work/04_environment/ENVIRONMENT.md](work/04_environment/ENVIRONMENT.md)):

| | FreqOrtho-SR | TADSR |
|---|---|---|
| Python | 3.10 | 3.10 |
| Torch / torchvision | 2.9.1 / 0.24.1 | 2.0.1 |
| transformers | 4.37.2 | 4.28.1 |
| diffusers | 0.25.0 | 0.25.0 |
| Others | xformers 0.0.33.post2 | xformers 0.0.20, peft 0.9.0 |

> Official package SHA-256: `8e24ea7a06dddeb7ab92322ee48bf4245d05023d9bb6ed629d81e791196d1a35` (matches the record in [work/SOURCE_FILES.md](work/SOURCE_FILES.md), re-verified by hash).

## How Controllability Lands on the Five Challenge Scenarios

The same generative pipeline keeps different detail boundaries for different degradation types:

| Scenario | Control principle |
|----------|-------------------|
| Small faces | Preserve identity structure; never invent a new face |
| Text | Guard stroke topology first, then sharpen edges |
| Dense vegetation | Suppress repeated textures and over-sharpening |
| Clocks | Preserve dial markers, hands, and circular structure |
| Birds | Recover feather boundaries without fabricating plumage |

Result-image evidence from the official package (case attribution verified against the test set by content matching):

| Text (case22 crop) | Dense vegetation (case56) | Clock (case43) | Bird (case60) |
|:---:|:---:|:---:|:---:|
| ![](assets/evidence_case22_text.jpg) | ![](assets/evidence_case56_plants.jpg) | ![](assets/evidence_case43_clock.jpg) | ![](assets/evidence_case60_bird.jpg) |

> In the text image, the "香樟" sign strokes, QR code, leaf veins, the clock's Roman markers, and the feather boundaries all stay natural — no repainting or artifact traces.

## Submission Compliance

- ✅ Only result images inside `output_dir` are submitted
- ✅ `case1.jpg … case100.jpg` strictly matched to test inputs
- ✅ No CJK punctuation in filenames; validation set excluded
- ✅ Outer `.zip` ≤ 10GB
- ✅ Result package ships with a SHA-256 manifest for audit

## Repository Structure

```
.
├── README.md                    # this document's Chinese original
├── README.en.md                 # English version
├── 赛题介绍.md                   # original competition brief (Chinese)
├── requirements.txt
├── data/
│   ├── 测试集/                  # case1 … case100 (4K inputs)
│   └── 验证集/                  # 5 aligned LQ/GT pairs
├── assets/                      # figures used by this document
├── work/                        # full reproduction kit (source / weights / scripts / env / official results)
│   ├── 00_README.md             # bundle index
│   ├── RUNBOOK.md               # five-step 4.0003 reproduction runbook
│   ├── SOURCE_FILES.md          # provenance declaration of original files
│   ├── 01_freqortho_sr/         # FreqOrtho-SR source + weights
│   ├── 02_tadsr/                # TADSR source + LoRA weights
│   ├── 03_pipeline/             # inference scripts & config
│   ├── 04_environment/          # environment manifests
│   ├── 05_official_package/     # official 4.0003 result package (100 images)
│   └── 06_defense/              # defense materials
└── 杂/                           # pre-competition notes & original zip bundles (zips not tracked)
```

> Official result package SHA-256: `8e24ea7a06dddeb7ab92322ee48bf4245d05023d9bb6ed629d81e791196d1a35` (matches the record in [work/SOURCE_FILES.md](work/SOURCE_FILES.md); re-verified by hash on this machine).

## Acknowledgements & Declaration

- The solution builds on **FreqOrtho-SR** (FreqMoE + orthogonal gradient projection) and **TADSR** (time-aware one-step diffusion distillation); third-party models and pretrained weights will be explicitly declared in the finals originality materials.
- Competition data is owned by the organizers, used here solely for this competition — no commercial or academic publication use.
- This repository is currently **private**.
