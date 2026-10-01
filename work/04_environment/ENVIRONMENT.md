# Environment reconstruction

Use two separate environments because the pinned versions differ.

## FreqOrtho-SR

- Python: 3.10
- Conda file: `env/environment-freqortho.yaml`
- Pip file: `env/requirements-freqortho.txt`
- Important pins from the reconstructed environment: Torch 2.9.1, torchvision 0.24.1, transformers 4.37.2, diffusers 0.25.0, xformers 0.0.33.post2.

```bash
conda env create -f env/environment-freqortho.yaml
conda activate PiSA-SR
```

## TADSR

- Python: 3.10
- Pip file: `env/requirements-tadsr.txt`
- Important pins: Torch 2.0.1, transformers 4.28.1, diffusers 0.25.0, xformers 0.0.20, peft 0.9.0.

```bash
python -m venv .venv-tadsr
source .venv-tadsr/bin/activate
pip install -r env/requirements-tadsr.txt
```

## Runtime notes

- The original runs used a CUDA GPU with tiled VAE/latent processing for 4K.
- `FreqOrtho-SR` performs its restoration at one quarter of the target size, then `upscale=4` restores the original geometry.
- TADSR inference uses `timesteps=500` for normal cases and `800` for cases listed in `scene_4k.json`.
- Do not install both environments' pinned Torch/xformers versions into one environment without revalidating behavior.