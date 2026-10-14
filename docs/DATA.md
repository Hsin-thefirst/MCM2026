# 数据

[原始 CSV](../data/SafeInvest_Synthetic_Data.csv) 共 5,400 行：180 个情景，每个 30 年。主键为 `(scenario_id, period)`，已核验完整性、唯一性和数值有效性。

| 字段 | 含义 |
| --- | --- |
| `scenario_id` / `period` | 情景 ID / 年度 |
| `equity_return` | 权益名义总收益 |
| `bond_return` | 政府债券名义总收益 |
| `bill_proxy_return` | 短期 bill 代理收益 |
| `inflation` | 年度通胀 |

收益和通胀用小数，均有限且大于 −1；负值保留。公共资产顺序为 `equity, bond, bill`。

情景 **1–120 用于开发，121–180 用于最终评估**。加载器 `safeinvest.data.load_scenarios` 校验后排序，`split_scenarios` 按完整情景划分。

文件 SHA256：`def0177704bf6cb98cec6da325b34208750d988f6ad97ec4a706a0d39e6f82c0`。

数据是基于 JST Macrohistory Database Release 6 部分美国序列生成的合成情景。题目及数据采用 CC BY-NC-SA 4.0；再分发时保留来源、许可和修改说明。详见 [题目 PDF](../2026_MCM_SafeInvest.pdf)。
