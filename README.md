# SafeInvest

2026 MCM A 题：设计一个能解释推荐理由、控制风险和交易成本的自动投资顾问。

[题目](2026_MCM_SafeInvest.pdf) · [数据](data/SafeInvest_Synthetic_Data.csv)

## 目的

1. 定义 Conservative、Balanced、Growth 三类组合，给客户 A–E 推荐组合或转人工。
2. 为 Balanced 设计年度调整策略，与每年恢复三等分的基准比较。
3. 检验成本变化，确定人工复核条件，完成英文报告。

增长用实际 CAGR，普通波动用实际收益标准差，大损失用实际最大回撤。资产顺序统一为 `equity, bond, bill`；成本率为 `0.0025`，建仓也收费。情景 **1–120 开发，121–180 最终测试**。

## 文档

| 文档 | 内容 |
| --- | --- |
| [建模方案](docs/FORMULATION.md) | 公式、组合与策略思路 |
| [统一接口](docs/INTERFACES.md) | 模块签名、核算口径和输出格式 |
| [任务清单](docs/ISSUES.md) |  自由增减 Issue, 仅作参考 |
| [协作约定](CONTRIBUTING.md) | 认领、提交和接口变更 |
| [工作记录](LOG.md) | 共同维护 |

建议先认领数据、引擎、指标和基准；策略与论文可以同步开始。共享类型和数据加载器已完成，接下来实现回测和策略。

建议人和 AI 共用 [LOG.md](LOG.md)：开始前阅读，完成一段工作后记录改动、验证和下一步，方便接手。

## 项目结构

```text
MCM2026/
├── README.md                    项目入口
├── CONTRIBUTING.md              团队协作
├── AGENTS.md                    AI 协作约定
├── LOG.md                       人和 AI 的工作记录
├── 2026_MCM_SafeInvest.pdf       题目
├── pyproject.toml               Python 项目与依赖
├── configs/
│   ├── default.json             权重、成本和实验配置
│   └── profiles.json            客户 A–E
├── data/
│   └── SafeInvest_Synthetic_Data.csv
├── docs/
│   ├── FORMULATION.md           建模方案
│   ├── INTERFACES.md            统一接口
│   ├── ISSUES.md                任务清单
│   └── DATA.md                  数据说明
├── src/safeinvest/
│   ├── contracts.py             公共类型
│   ├── data.py                  加载与划分（已实现）
│   ├── engine.py                交易与回测
│   ├── metrics.py               增长与风险指标
│   ├── suitability.py           客户适配
│   ├── policies/                基准、固定与动态策略
│   ├── evaluation.py            实验入口
│   └── reporting.py             图表与导出
├── experiments/                 探索笔记
├── tests/                       模块与集成验证
├── outputs/                     生成的运行结果
├── report/                      英文报告及图表
└── .github/ISSUE_TEMPLATE/       Issue 模板
```

目录树包含后续模块规划：引擎、指标、适配、策略、实验入口、图表及报告待实现。公共类型：[contracts.py](src/safeinvest/contracts.py)。

## 运行

Python 3.10+，在仓库根目录运行现有测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

需要分析库时安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
```

依赖配置：[pyproject.toml](pyproject.toml)。数据说明：[DATA.md](docs/DATA.md)。

# DDL: 2026-10-20 23:59