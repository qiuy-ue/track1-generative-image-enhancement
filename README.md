# 清晰有度，增强可控

**简体中文** | [English](README.en.md)

![Competition](https://img.shields.io/badge/2026_CSIG_Camera学术之星-赛道一-2563EB)
![Stage](https://img.shields.io/badge/阶段-决赛方案-7C3AED)
![Score](https://img.shields.io/badge/初赛官方记录-4.0003-16A34A)
![Architecture](https://img.shields.io/badge/架构-Diffusion-DB2777)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9_/_2.0-EE4C2C)
![Hardware](https://img.shields.io/badge/全流程-单卡_A100_40GB-0891B2)

> [!IMPORTANT]
> **一句话**：基于 **FreqOrtho-SR + TADSR** 的两级生成式增强 —— FreqOrtho-SR 在四分之一分辨率守住像素结构（FreqMoE 频率路由 + OGP 正交梯度投影），TADSR 时间感知一步扩散把细节落回原生 4K，最后用可解释的加权规则完成候选筛选。**初赛官方记录 4.0003**（100/100 张 4K 结果严格命名封装，附 SHA-256），**全流程单卡 A100 40GB**。
>
> **看什么**：[方案总览](#方案总览)（两张架构图）· [五类挑战场景结果证据](#可控性如何落到五类挑战) · [端到端工程闭环](#端到端工程闭环)
>
> **怎么跑**：完整复现套件已随仓库归档于 [`work/`](work/) —— 源码、权重、推理脚本、双环境依赖与官方结果包齐备，五步复现见 [work/RUNBOOK.md](work/RUNBOOK.md)，环境重建见 [work/04_environment/ENVIRONMENT.md](work/04_environment/ENVIRONMENT.md)。仅 TADSR base 与 SD2.1-base 两个大底模不随库分发，下载说明见 [work/00_README.md](work/00_README.md)。

---

## 成果一览

| 项目 | 结果 |
|------|------|
| 初赛官方记录得分 | **4.0003**（100 / 100 张 4K 结果已提交封装，附 SHA-256 校验） |
| 候选筛选官方记录 top-5 | case **44 · 60 · 97 · 79 · 51** |
| FreqOrtho-SR · DRealSR 论文基准 | PSNR **29.08** / LPIPS **0.2799**（两项同时改善） |
| 硬件门槛 | 单卡 A100 40GB 完成全部推理与封装 |

## 赛题与挑战

赛题要求以 **Diffusion 架构**对 4K 低质量图像进行增强，评分对照隐藏 Ground Truth，综合多项有参考与无参考指标加权。真正难的不是放大，而是在放大后仍然保持内容真实。

**四个硬约束：** 清晰度提升 · 内容前后一致 · 分辨率保持不变 · 色彩与结构自然协调

**五类重点挑战场景：** 小人脸 · 文字 · 密集绿植 · 钟表 · 鸟类

## 方案总览

```mermaid
flowchart LR
    A["01 低质量输入<br/>4K test 图像"] --> B["02 结构恢复<br/>FreqOrtho-SR<br/>四分之一分辨率"]
    B --> C["03 细节 teacher<br/>TADSR<br/>原生 4K 细化"]
    C --> D["04 比较与提交<br/>候选筛选 + 严格命名封装<br/>case1…case100"]
```

一条工程主线：**像素结构由 FreqMoE 守住，语义细节由 TADSR 补足，最终输出回到统一的 4K 几何。**

### FreqOrtho-SR：把「清晰」和「真实」拆成可控方向

核心思想：**像素保真与语义感知分开学习，再用正交约束减少相互干扰。**

- **FreqMoE** —— FFT 降质特征路由到像素级 LoRA 专家，先判断输入更像哪一种退化；
- **OGP（正交梯度投影）** —— 用 SVD 提取像素子空间，把语义 LoRA 梯度投影到其零空间，两个优化方向互不拖累。

![FreqOrtho-SR 架构：两阶段训练与 OGP](assets/freqortho_arch.png)

### TADSR：用时间感知的一步扩散守住 4K 细节

TADSR 的作用不是重新发明内容，而是**把粗恢复结果细化到更自然的 4K 表面**。

- 时间感知编码器 + 学生模型（UNet + LoRA）；
- 冻结的教师模型提供细节约束（TAVSD 蒸馏）；
- 高低时间步适配复杂 4K 场景：**普通案例 t=500，复杂案例（scene_4k.json 标定）t=800**。

![TADSR 时间感知蒸馏框架](assets/tadsr_overview.png)

## 端到端工程闭环

| 阶段 | 内容 |
|------|------|
| 准备 | 生成 quarter-resolution 输入 |
| 恢复 | FreqOrtho-SR one-step 推理 |
| 细化 | TADSR teacher 细化至原生 4K（t=500 / 800） |
| 比较 | 候选筛选：`utility = 1.10286 × d_nr4 + 0.21747 × d_edge` |
| 封装 | 统一命名输出 `output_dir/case1.jpg … case100.jpg` |

筛选权重来自全量 100 张的 teacher-vs-候选对比实验（`work/03_pipeline/config/freq_teacher_vs_plus3_full100_1024.json`），是一条**可解释、可复核**的规则，而不是黑盒打分。

## 复现（work/）

完整复现套件已随仓库归档，目录对应原编号分包：

| 目录 | 内容 |
|------|------|
| `work/01_freqortho_sr/` | FreqOrtho-SR 完整源码 + `freqortho_sr.pkl` 权重（57.7MB） |
| `work/02_tadsr/` | TADSR 源码 + LoRA 权重（`tadsr_imag2006_a15.pkl` 等） |
| `work/03_pipeline/` | 推理 / 比较 / 筛选 / 封装脚本与 config（`scene_4k.json` 等） |
| `work/04_environment/` | 双环境依赖清单（Conda + pip，含底模下载说明） |
| `work/05_official_package/` | 官方 4.0003 提交包：`output_dir/case1.jpg … case100.jpg` |
| `work/06_defense/` | 答辩用素材 |

**五步复现**（细节见 [work/RUNBOOK.md](work/RUNBOOK.md)）：

1. 生成 quarter-resolution 输入 —— `pipeline/prep_freqortho_full100.py`
2. 打补丁并运行 FreqOrtho-SR one-step 推理（`--process_size 512 --upscale 4`）
3. TADSR teacher 细化回原生 4K —— 普通案例 t=500，`scene_4k.json` 复杂案例 t=800；teacher 高斯模糊 r=3.0 + ADain 色彩校正
4. 候选筛选 `utility = 1.10286 × d_nr4 + 0.21747 × d_edge` —— 官方 top-5：**44 / 60 / 97 / 79 / 51**
5. 重建提交包 —— 仅替换入选 5 张，其余 95 张与 opt7 4.0002 基线字节一致

**环境**（两套独立环境，版本钉死，详见 [work/04_environment/ENVIRONMENT.md](work/04_environment/ENVIRONMENT.md)）：

| | FreqOrtho-SR | TADSR |
|---|---|---|
| Python | 3.10 | 3.10 |
| Torch / torchvision | 2.9.1 / 0.24.1 | 2.0.1 |
| transformers | 4.37.2 | 4.28.1 |
| diffusers | 0.25.0 | 0.25.0 |
| 其他 | xformers 0.0.33.post2 | xformers 0.0.20, peft 0.9.0 |

> 官方结果包 SHA-256：`8e24ea7a06dddeb7ab92322ee48bf4245d05023d9bb6ed629d81e791196d1a35`（与 [work/SOURCE_FILES.md](work/SOURCE_FILES.md) 记录一致，已实测复核）。

## 可控性如何落到五类挑战

同一套生成式增强流程，针对不同退化类型保留不同的细节边界：

| 场景 | 控制原则 |
|------|----------|
| 小人脸 | 保留身份结构，避免凭空改脸 |
| 文字 | 先守住笔画拓扑，再提升边缘 |
| 密集绿植 | 抑制重复纹理与过度锐化 |
| 钟表 | 保持刻度、指针与圆形结构 |
| 鸟类 | 恢复羽毛边界，不制造伪羽纹 |

官方结果包中的结果图证据（按内容比对测试集确认归属）：

| 文字（case22 局部） | 密集绿植（case56） | 钟表（case43） | 鸟类（case60） |
|:---:|:---:|:---:|:---:|
| ![](assets/evidence_case22_text.jpg) | ![](assets/evidence_case56_plants.jpg) | ![](assets/evidence_case43_clock.jpg) | ![](assets/evidence_case60_bird.jpg) |

> 文字场景图中「香樟」铭牌笔画、二维码与叶脉纹理、钟面罗马刻度、羽毛边界均保持自然，无重绘或伪影迹象。

## 提交规范达成情况

- ✅ 仅提交 `output_dir` 内的结果图
- ✅ `case1.jpg … case100.jpg` 与测试图严格对应
- ✅ 文件名不含中文符号，不混入验证集
- ✅ 外层打包 `.zip` ≤ 10GB
- ✅ 结果包附带 SHA-256 清单，可复核

## 仓库结构

```
.
├── README.md                    # 本文档（简体中文）
├── README.en.md                 # English version
├── 赛题介绍.md                   # 赛题规则原文整理
├── requirements.txt
├── data/
│   ├── 测试集/                  # case1 … case100（4K 输入）
│   └── 验证集/                  # 5 组 LQ/GT 对齐数据
├── assets/                      # 本文档配图（PPT 内嵌图与结果证据）
├── work/                        # 完整复现套件（源码 / 权重 / 脚本 / 环境官方结果包）
│   ├── 00_README.md             # 分包索引
│   ├── RUNBOOK.md               # 4.0003 五步复现手册
│   ├── SOURCE_FILES.md          # 原始文件位置申明
│   ├── 01_freqortho_sr/         # FreqOrtho-SR 源码 + 权重
│   ├── 02_tadsr/                # TADSR 源码 + LoRA 权重
│   ├── 03_pipeline/             # 推理脚本与 config
│   ├── 04_environment/          # 环境依赖清单
│   ├── 05_official_package/     # 官方 4.0003 结果包（100 张）
│   └── 06_defense/              # 答辩素材
└── 杂/                           # 赛前笔记与原始压缩包留档（zip 不入库）
```

## 致谢与声明

- 方案基于 **FreqOrtho-SR**（FreqMoE + 正交梯度投影）与 **TADSR**（时间感知一步扩散蒸馏），引用的第三方模型与预训练权重将在决赛原创性证明材料中显著申明。
- 比赛数据所有权归主办方，仅用于本赛事，不作商用或学术发表。
- 本仓库当前为**私有**仓库。
