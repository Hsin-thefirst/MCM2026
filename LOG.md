# LOG
## Template

```text
### YYYY-MM-DD — 执行者，任务或 Issue
- 完成：做了什么，涉及哪些文件。
- 验证：运行了什么，结果如何；未验证的部分写明。
- 下一步：待完成事项、需要谁接手。
```

## 2026-10-14

### 项目初始化

- 完成：建模方案、模块接口、11 项任务、Python 公共类型和数据加载器。
- 文件：`docs/`、`configs/`、`src/safeinvest/`、`tests/`、`pyproject.toml`。
- 验证：11 项测试通过；数据为完整的 180×30 网格，开发 / 测试划分为 120 / 60。
- 下一步：认领数据探索、回测引擎、指标和基准策略。

### 文档整理

- 完成：精简文档，统一标题和“名称：内容”的写法；README 补充项目目录树。
- 文件：`README.md`、`CONTRIBUTING.md`、`docs/`、`LOG.md`、`AGENTS.md`。
- 协作：人和 AI 开始前读 LOG，完成后记录改动、验证和下一步。
- 验证：文档链接、LOG 引用和 11 项任务检查通过，`git diff --check` 通过。
- 下一步：将任务清单转成 Issue，认领后在这里记录实现进展。

### 基准框架
- 完成：阅读权威题目 PDF、共享接口、数据加载器和项目记录；实现先隔离分区的开发加载器、可核验的年度核算、17 种基准策略、嵌套情景交叉验证、指标、配对区间、客户适配及受控冻结/最终评价流程。
- 文件：`safebench/`、`configs/development.json`、`run.py`、`tests/`、`docs/`、`requirements.lock.txt` 和 `README.md`。
- 研究：查阅 1/N 分散配置、协方差收缩、CVaR 和模型选择偏差的原始论文；来源链接与本项目的改编记录在 `docs/METHODOLOGY.md`。
- 验证：解析核算、因果前缀、复核状态、分区保护和合成数据最终评价流程通过；未使用题目提供的最终留出情景。
- v1 暴露出新增 Balanced 权重区间内 8% 年度 CVaR 上限不可行。保留并标记该探索运行，移除静默回退；防御性 CVaR 策略改在官方单纯形内搜索，不可行时明确报错，上限不变。
- v2 执行修正模型；v3 仅优化完整账本核验，全部 18,360 行财务指标与 v2 完全一致。v3 是当时与代码匹配的权威运行：120 条开发情景、17 种策略、五个期限及五个 H=30 成本情形。
- 结果：成本感知策略 30 年中位实际 CAGR 为 5.45%，官方基准为 3.58%；但 P90 最大回撤为 52.03%，官方为 34.84%。没有门控候选同时满足增长/下行风险晋级筛选。高复核率和等待期间持有不能证明风险保护有效。
- 文件：`outputs/development_v1/`、`outputs/development_v2/`、`outputs/development_v3/`、`docs/FINDINGS.md` 及实验登记。
- 验证：28 项测试通过；核验全部 10,200 条组合路径（306,000 次年度决策、18,360 行指标），最大指标差异 1.37e-12。独立市场重放核对全部 306,000 次收益/通胀变化和 1,080 次标量参考计算，最大持仓差异 1.82e-12。各外层训练 CVaR 约束可行，交叉验证分区互斥。
- 交付：离线 HTML 报告、五组 PNG/SVG 图、原始及逐情景指标、压缩完整决策轨迹、数学定义、选择协议和运行说明。已检查图表和 HTML 资源链接；当时代码 hash 与 v3 manifest 匹配，git 状态仅新增独立目录。
- 最终评价流程只用生成的零收益数据测试；未创建真实冻结包或最终评价锁，本阶段未触及真实情景 121–180。
- 下一步：团队审查风险偏好/客户适配假设及预注册的路径风险约束开发实验；候选选择完成后冻结，只执行一次真实最终评价。共享包合并需先获授权并确认接口。

### Benchmark

- 完成：独立目录改名为 `benchmark/`；历史日志并入根日志，目录内日志改为索引。统一根 README、协作说明、接口、建模方案、数据说明和任务进展；原历史实验 manifest 保持原样。
- 依赖与入口：虚拟环境统一到根 `.venv/` 并修复迁移后的启动脚本；精确版本移到根 `requirements.lock.txt`，`requirements.txt` 引用 `pyproject.toml` 的 analysis 依赖；新增 Makefile 的 test/benchmark/report/audit/replay。
- 数据隔离：原完整网格单测改用生成的 180×30 数据；日常测试不读取真实最终留出集。
- 验证：11 项共享测试 + 28 项 benchmark 测试通过；依赖检查、文档链接、CLI 和报告入口、`git diff --check` 通过。新 `development_integrated` 的 18,360 行指标与 v3 完全一致，306,000 条年度记录完成账本核验和独立市场重放，代码 hash 匹配。真实最终评估未执行。
- 下一步：用根 README/Makefile 作为统一入口；新模型使用新运行目录，按协议选择方案后再冻结并执行一次真实最终评估。
- 完成：仅保留根 README/LOG；benchmark 的 docs、configs、tests 分别并入根目录，独立重放工具移到 `scripts/verify_benchmark.py`。合并 FORMULATION 与 METHODOLOGY、两份接口说明；CONTRIBUTING 内容并入根入口，删除冗余副本、无用默认配置和空 experiments 目录。
- 依赖与语言：依赖范围仅由 `pyproject.toml` 维护，保留精确版本快照 `requirements.lock.txt`，删除转发用的 requirements.txt 和重复忽略文件。人读文档、历史英文日志译文及新 HTML 报告使用中文；AGENTS、代码标识符和机器接口使用英文。图表保留英文以便提交引用。
- 文件：`README.md`、`LOG.md`、`AGENTS.md`、`docs/`、`configs/benchmark.json`、`tests/test_benchmark.py`、`scripts/`、`Makefile`、`benchmark/run.py`、报告生成器及实验登记。
- 验证：统一入口 39 项测试通过；新 `development_consolidated` 的 18,360 行指标与 integrated 完全一致。核验 10,200 条路径，独立重放 306,000 次年度变化及 1,080 项标量参考计算；代码 hash 匹配，中文报告资源和文档链接有效。历史实验快照保持原样，真实最终留出集未使用。
- 下一步：从根入口运行新实验，继续按统一协议研究/选择模型；不再创建平行的文档、配置或依赖清单。
