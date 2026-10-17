# 提交版本与复现

本文维护源码交付与复现入口，不制作论文或数据报告。实际提交策略以冻结包的`selected_policy`及开发运行的`submission_decision.json`为准，不手工挑选外层赢家。

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
.venv/bin/python scripts/package_submission.py --development-run benchmark/outputs/development_submission_v1 --bundle benchmark/submission_frozen_v1.json --final-run benchmark/outputs/final_submission_v1 --out dist/SafeInvest_source_v1.zip
```

打包工具核验源码hash、冻结包、最终锁和数值审计的一致性，保存源码、配置、原始赛题数据、测试、复现说明、紧凑开发证据与完整最终账本。原开发年度账本保留在工作区，不放入源码ZIP；可用同配置重现。压缩包严格小于50,000,000字节，逐文件SHA256与CRC核验，清单为`PACKAGE_MANIFEST.json`，包外另存verification JSON。

源码包不含虚拟环境、Git历史、缓存、密钥或论文。比赛所需团队信息、来源与AI使用披露须在最终提交材料中如实填写；本轮按用户要求继续暂停论文制作。
