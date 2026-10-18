# 提交版本与复现

本文维护源码交付与复现入口，不制作论文或数据报告。实际提交策略以冻结包的`selected_policy`及开发运行的`submission_decision.json`为准，不手工挑选外层赢家。

2026-10-18的最终收敛工作位于`feat/final-submission-release`。最后预算惩罚批次完成后，以该运行的`release_decision.json`记录唯一金融程序、客户核验和来源；新候选必须同时通过预先登记的原金融/可行性/费用条件。没有通过者保留原冻结A，不能把新版源码包解释为新模型已经通过最终评价。

当前版本已冻结`plan_a_balanced`，全120选择的训练比例0.95、gamma4，客户A/D人工、B/C保守、E均衡。真实最终评价已经完成一次，结果位于`benchmark/outputs/final_submission_v1`；所含最终锁不得删除。后续只复现固定程序、核验保存结果，不再使用最终表现修改金融模型。

## 环境及开发复现

项目根目录安装精确运行环境并执行统一测试：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -e '.[analysis]'
make test
```

开发重现使用全新目录，只处理情景1–120。运行前在实验登记追加复现记录，不覆盖原结果：

```bash
.venv/bin/python benchmark/run.py development --config configs/submission.json --out benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_benchmark.py benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_plan_a.py benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_submission.py benchmark/outputs/development_submission_reproduction
```

默认不生成报告或论文。重现耗时受机器影响，金融结果应相同，推理计时不要求相同。原25策略在共同评价键上保持不变；双倍费用新增加四个短期限，因此不能直接用总行数比较历史运行。

## 冻结及最终评价

严格遵守[实验协议](PROTOCOL.md)。开发核验结束、停止金融搜索后，先读取`submission_decision.json`确定唯一策略ID，再执行freeze。冻结会以120条开发路径执行同一选择和重拟合，不能直接把某折参数当作最终参数。

真实最终121–180只能评价一次。最终锁和冻结包随源码包保留；源码包已有最终结果时，直接读取保存文件，不运行final命令、不删锁、不依据最终结果改模型。生成数据测试使用独立临时锁，不消费真实最终评价机会。

`scripts/verify_saved_final.py <final_run>`只读取已保存最终结果，检查所有策略账本之间的市场一致性、独立标量等权核算及冻结客户映射；不会重开市场CSV或重新运行策略。

## 源码包

在开发与最终核验都完成后运行：

```bash
.venv/bin/python scripts/package_submission.py --development-run benchmark/outputs/development_submission_v1 --bundle benchmark/submission_frozen_v1.json --final-run benchmark/outputs/final_submission_v1 --out dist/SafeInvest_source.zip
```

打包工具核验源码hash、冻结包、最终锁和数值审计的一致性，保存源码、配置、原始赛题数据、测试、复现说明、紧凑开发证据与完整最终账本。原开发年度账本保留在工作区，不放入源码ZIP；可用同配置重现。压缩包严格小于50,000,000字节，逐文件SHA256与CRC核验，清单为`PACKAGE_MANIFEST.json`，包外另存verification JSON。

独立核验需要的历史参照也随包保留：原Benchmark、校准A、C筛查v3三个目录的原始`scenario_metrics.csv`与`manifest.json`。只复制已有数值快照，不修改其内容或hash。正式源码包补齐历史复现依赖与题目PDF；早期打包草稿不是交付版本，金融模型、冻结包及最终评价仍为同一次v1。

源码包不含虚拟环境、Git历史、缓存、密钥或论文。比赛所需团队信息、来源与AI使用披露须在最终提交材料中如实填写；本轮按用户要求继续暂停论文制作。

## 最终收敛源码包

新版交付使用`dist/SafeInvest_final_source.zip`，不覆盖原45.5MB快照。默认金融程序的完整复现依赖和完整原最终账本仍保留；同时附带后续研究的源码、配置、汇总、失败约束与核验JSON。研究大账本和训练模型缓存不放入50MB包，保留在原工作区，所以紧凑研究证据不支持在解压目录直接独立重放全部研究路径。`make test`及`configs/submission.json`默认提交程序的复现依赖完整。

在发布决定为ready且对应原冻结A时，使用：

```bash
.venv/bin/python scripts/package_submission.py --development-run benchmark/outputs/development_submission_v1 --bundle benchmark/submission_frozen_v1.json --final-run benchmark/outputs/final_submission_v1 --research-run benchmark/outputs/development_fusion_calibration_v1 --research-run benchmark/outputs/development_fusion_value_extensions_v1 --research-run benchmark/outputs/development_final_budget_v1 --release-decision benchmark/outputs/development_final_budget_v1/release_decision.json --out dist/SafeInvest_final_source.zip
```

新包生成后独立解压，在精确已验证环境执行`make test`；核对金融源码、冻结包、客户分配及全部文件hash/CRC。在解压目录按同一显式输入重打包（输出改为新文件名），应得到完全相同字节。包外verification JSON记录最终大小及hash，另附独立复现JSON；不要在源码包内嵌该包自身hash，避免自引用。已使用的最终集只保留保存证据，不再运行final入口。
