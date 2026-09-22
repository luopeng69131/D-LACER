# D-LACER

### 延迟反馈下的潜在状态自适应条件风险控制

**面向多步 Starlink 吞吐量预测的条件安全控制方法。**

[![论文](https://img.shields.io/badge/arXiv-2605.09508-b31b1b.svg)](https://arxiv.org/abs/2605.09508)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | **中文**

## 项目简介

Starlink 吞吐量会随网络状态和预测距离快速变化。一个预测器即使满足平均高估风险预算，仍可能在低容量时段、高不确定性窗口或较远预测距离上集中违约。对于多步预测，完整真实结果还会延迟到达，进一步增加在线风险控制的难度。

**D-LACER** 是一个与预测骨干模型解耦的多步预测安全层。它从预测发布时可获得的信息中识别潜在容量状态，用相互重叠的条件风险组表达安全要求，并在延迟结果到达后分别更新各组的单侧修正量。最大修正规则协调同一预测元素上的多个安全要求，同时避免简单累加带来的过度保守。

在三个公开 Starlink 数据集和四种风险预算的 12 个组合上，D-LACER 实现了 **12/12 的全组预算通过率**；本文实验中表现最好的在线基线为 **5/12**，同时 D-LACER 保持了相近的预测精度。

<p align="center">
  <img src="assets/method_overview.png" width="96%" alt="D-LACER 方法框架">
</p>

<p align="center"><em>D-LACER 将预测路由到多个相互重叠的风险组，并在延迟反馈到达后更新各组的安全修正量。</em></p>

## 方法概览

D-LACER 通过四个模块将常规多步预测器转化为条件安全预测系统：

1. **发布时表征。** 基于历史上下文、辅助预测形态和专家分歧构造不依赖未来标签的特征。
2. **潜在状态路由。** 两个学习式路由器识别潜在低容量和严重低容量窗口，并与高分歧组及近、中、远预测距离组形成重叠风险结构。
3. **延迟分组控制。** 使用历史残差初始化各组单侧修正量，仅在对应真实窗口完整可见后执行更新。
4. **最大修正聚合。** 每个预测元素采用其所有活跃风险组中的最大修正量，以单个标量修正协调多项安全要求。

该实现可以接入不同的预测骨干。仓库同时提供了按预测距离训练的 XGBoost 点预测/分位数预测库，用于运行端到端示例。

## 主要结果

下表汇总了 Chicago、Osnabrueck 和 Victoria 三个数据集、四种高估风险预算上的主要结果。**全组通过**要求七个条件风险组全部满足预算。

| 方法 | 全局通过 | 全组通过 | MAE |
|:--|--:|--:|--:|
| T3P Budget-Scale | 12/12 | 0/12 | 46.543 |
| One-sided CQR | 12/12 | 1/12 | 43.754 |
| Multi-step ACI | 12/12 | 4/12 | **42.493** |
| ConformalOpt-SQT | 12/12 | 5/12 | 42.965 |
| **D-LACER** | **12/12** | **12/12** | 42.541 |

<p align="center">
  <img src="assets/main_compliance.png" width="96%" alt="边际风险与条件风险预算通过结果">
</p>

<p align="center"><em>仅满足边际风险预算仍可能掩盖条件违约；D-LACER 在全部数据集与预算组合上均实现全组通过。</em></p>

<details>
<summary><strong>各条件风险组结果</strong></summary>

<br>

<p align="center">
  <img src="assets/group_compliance.png" width="96%" alt="重叠条件风险组的预算通过情况">
</p>

七个风险组包括全局预测流、两个潜在容量状态、高预测分歧，以及近、中、远三个预测距离范围。

</details>

## 代码结构

```text
D-LACER/
├── assets/                  # 方法图与实验结果图
├── configs/paper.json       # 与论文对应的默认配置
├── data/README.md           # 数据来源与处理说明
├── scripts/
│   ├── prepare_starnet.py   # 数据清洗与窗口构造
│   ├── run_demo.py          # 合成数据快速演示
│   └── run_experiment.py    # Starlink 端到端实验
├── src/dlacer/
│   ├── controller.py        # 延迟投影分组控制器
│   ├── groups.py            # 潜在状态路由与重叠分组
│   ├── features.py          # 发布时无标签特征
│   ├── forecast.py          # XGBoost 点/分位数预测库
│   ├── data.py              # 数据处理与时间划分
│   └── metrics.py           # 精度与单侧安全指标
└── tests/                   # 控制器和数据处理测试
```

### 论文方法与代码对应关系

| 方法模块 | 代码位置 |
|:--|:--|
| 发布时表征 | `src/dlacer/features.py` |
| 潜在容量状态路由 | `src/dlacer/groups.py::build_latent_risk_groups` |
| 残差初始化与延迟投影更新 | `src/dlacer/controller.py::DLACERController` |
| 最小保守性的最大修正聚合 | `src/dlacer/controller.py::DLACERController.run` |
| Starlink 窗口化与带隔离区的时间划分 | `src/dlacer/data.py` |

## 环境安装

```bash
git clone https://github.com/luopeng69131/D-LACER.git
cd D-LACER

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

运行合成数据上的延迟反馈示例：

```bash
python scripts/run_demo.py
```

程序将输出整体安全指标、各风险组的高估率，以及每个条件风险组是否满足给定预算。

## Starlink 实验

公开数据的官方下载地址和引用信息见 [`data/README.md`](data/README.md)。按照 [StarNet 数据处理流程](https://github.com/ConnectedSystemsLab/StarNet) 获得某一地区的 `dataset_tp_sat.pkl` 后，构造论文使用的预测窗口：

```bash
python scripts/prepare_starnet.py \
  --input /path/to/dataset_tp_sat.pkl \
  --output data/processed/chi.npz \
  --name chi \
  --step-len 46
```

随后训练参考预测器并运行 D-LACER：

```bash
python scripts/run_experiment.py \
  --data data/processed/chi.npz \
  --budget 0.35 \
  --output outputs/chi_b35.npz
```

实验会将预测结果和修正轨迹保存为 `.npz`，同时在同目录生成包含整体指标与分组指标的 JSON 文件。三个地区与四种预算的论文配置见 [`configs/paper.json`](configs/paper.json)。

## 接入其他预测骨干

```python
from dlacer import DLACERController, build_latent_risk_groups

groups = build_latent_risk_groups(
    context=issue_time_context,                    # (窗口数, 特征数)
    point_prediction=backbone_prediction,          # (窗口数, 预测长度)
    candidate_predictions=auxiliary_predictions,  # (专家数, 窗口数, 预测长度)
    target=observed_throughput,
    initialization_end=initialization_end,
)

result = DLACERController().run(
    base_prediction=backbone_prediction,
    target=observed_throughput,
    groups=groups,
    budget=0.35,
    initialization_end=initialization_end,
    evaluation_start=evaluation_start,
    feedback_delay=feedback_delay,
)

print(result.metrics)
print(result.group_passes(0.35))
```

## 数据集准备

数据处理代码位于 `src/dlacer/data.py`，包含时间排序、缺失值清理、卫星编号编码、时间特征构造、75 步输入到 15 步输出的窗口生成、路由上下文特征生成，以及 55/15/15/15 的带隔离区时间划分。各地区窗口步长与论文实验配置记录在 `configs/paper.json`。

## 引用

如果本项目对你的研究有所帮助，请引用：

> Xie H, Zhang C, Luo P, et al. Risk-aware safe throughput forecasting for Starlink networks. *arXiv preprint arXiv:2605.09508*, 2026.

```bibtex
@article{xie2026risk,
  title   = {Risk-Aware Safe Throughput Forecasting for Starlink Networks},
  author  = {Xie, Hongjun and Zhang, Chao and Luo, Pengcheng and others},
  journal = {arXiv preprint arXiv:2605.09508},
  year    = {2026}
}
```

## 致谢

本项目实验使用了 [StarNet](https://github.com/ConnectedSystemsLab/StarNet) 发布的 Starlink 实测数据与数据处理工具，感谢原作者向研究社区开放相关成果。

## License

本项目采用 [MIT License](LICENSE) 开源。
