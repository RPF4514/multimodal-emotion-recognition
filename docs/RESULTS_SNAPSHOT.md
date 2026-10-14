# 最终版本结果快照

这是已有实验日志的摘录，整理于2026年10月14日，未重新训练或评价数据。

## 问题一

全部100条样本保留，已有人工观察21条，其余79条明确标为未人工复核。正式输出是outputs/problem1/audited_20260924，文件放在Release运行材料中。

## 问题二：锁定seed-43模型

| 数据 | 完整Accuracy | 完整Macro-F1 | 完整MAE | 完整Pearson | 18场景平均Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 验证集728条 | 0.6346 | 0.6192 | 0.6125 | 0.6511 | 0.5873 |
| 测试集727条 | 0.6534 | 0.6143 | 0.6519 | 0.6695 | 0.5818 |

来源：outputs/logs/bertmask_validation_20260924.json → scenario_metrics → retrained_teacher → 43；outputs/logs/final_test_bertmask_seed43_20260925.json → scenario_metrics。

测试集在旧模型探索阶段已曾被查看，不能描述成整个研发过程未接触过的盲测。最终测试日志保留此限制与原始文件哈希，原权重和打包权重的哈希对应见SOURCE_WEIGHT_AUDIT.json。

附件3无标签，30条预测分布为负向7、中性10、正向13；这不是准确率。

## 问题三

附件4无标签，20条预测解释分布为负向8、中性5、正向7，既有文本一致性核对20/20。验证方法记录见outputs/logs/explanation_validation_bertmask_20260924.json；20条专项窗口的事后比较不能冒充独立验证或因果效果。

三问的完整实验与复现记录由仓库outputs/logs和docs/submission_reproduction.md给出，实际定位可信度随解释记录保留。
