# 生成式图像增强可控性挑战

这是赛道一的项目骨架，已经包含：

- 验证集分析
- 验证集报告
- 推理入口
- 提交包打包

## 当前入口

- `scripts/analyze_dataset.py`
- `scripts/make_validation_report.py`
- `scripts/run_inference.py`
- `scripts/package_submission.py`

## 推理后端

当前默认是 `copy` 基线，保证流程能跑通。

更强的开源 diffusion 后端我先给你接了 DiffBIR v2，走官方 `inference.py`，不需要 prompt：

```yaml
inference:
  mode: diffbir_cli
```

对应配置在 [configs/diffbir.yaml](D:\AAA赛道一：生成式图像增强可控性挑战\configs\diffbir.yaml)。

你需要先把官方 DiffBIR 仓库 clone 到本地，再把 `repo_dir` 改成那个路径。

## 推荐流程

```bash
python scripts/make_validation_report.py
python scripts/run_inference.py --output_dir outputs/baseline
python scripts/package_submission.py --output_dir outputs/baseline --zip_path outputs/submission.zip
```

DiffBIR 版本的运行方式：

```bash
python scripts/run_inference.py --config configs/diffbir.yaml --output_dir outputs/diffbir
python scripts/package_submission.py --output_dir outputs/diffbir --zip_path outputs/diffbir_submission.zip
```

如果你后面想继续冲更强的 open-source diffusion，我建议下一步再切到 HYPIR。官方仓库自己也明确写了，它在他们的更新里“significantly outperforms DiffBIR”并且更快。
