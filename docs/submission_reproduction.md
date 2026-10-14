# E题支撑材料使用说明

## 内容与正式结果
问题一正式交付在 outputs/problem1/audited_20260924/：features_100.npz（100条三模态特征）、coverage_100.csv、time_map_100.jsonl和run_summary.json。
上一级features_100.npz与time_map.jsonl为审计前流程中间结果，仅供复核与重建，论文问题一以audited_20260924为准。既有21条审计表随包保留；不需要再次人工标注。
问题二选中检查点为 outputs/checkpoints/gated_bigru_teacher_bertmask24_seed43.pt。另一text_student.pt是公共训练入口的构造依赖，不是最终文本输入路线。
附件3正式文件：outputs/predictions/attachment3_predictions_20260924.csv（30条）。
附件4正式文件：outputs/predictions/attachment4_predictions_explanations_20260924.csv（20条），完整窗口分数、时间映射和关键帧随包保留。
类别0/1/2依次为Negative/Neutral/Positive；无标签专项附件不计算准确率。

## 环境与输入
主路线aligned_50，数据由官方赛题附件提供，未重复打包。设置环境变量MM_EMOTION_DATA_ROOT为实际“E题数据/E题数据”目录。原始视频、文本、标签和配对保持题给版本。
本包保留训练集拟合的outputs/normalizer.npz。环境详见ENVIRONMENT.txt，字段、掩码规则见DATA_INTERFACE.md。
冻结bert-base-uncased使用公开通用权重；包内含配置、词表、获取与SHA-256验证脚本，不含约440MB公开权重。本队自训的模型权重已包含。公开资源的来源与许可见SOURCES_AND_LICENSES.md。

## 复现专项预测（PowerShell，在解压目录运行）
```powershell
$env:MM_EMOTION_DATA_ROOT = 'D:\赛题数据\E题数据'
python -m src.fetch_verified_bert
python -m unittest discover -s tests -p 'test_*.py' -v
$s = 'outputs/logs/model_selection_bertmask_20260924.json'
python -m src.explain --selection $s --output-tag rerun
python -m src.predict_with_bert_text --selection $s --output-tag rerun
```
先解释附件4再推理附件3，是为了核对附件4的题给文本向量与重建文本向量预测一致性。以上使用新标签rerun，原正式CSV保留。再次执行时换一个未使用的标签。
src/explain.py读取已有时间映射；不要求重新运行ASR。FFmpeg需可用。时间映射是估计定位，低可信状态保留。

## 问题一从原素材重建（可选，在副本中运行）
先准备公开Systran/faster-whisper-tiny.en、Systran/faster-whisper-base.en模型到outputs/problem1/pretrained缓存。本包含已有结果，重新提取不是读取结果的前提。
```powershell
python -c "from faster_whisper import WhisperModel; [WhisperModel(n, device='cpu', compute_type='int8', download_root='outputs/problem1/pretrained') for n in ('tiny.en','base.en')]"
python -m src.problem1 --alignment whisper
python -m src.finalize_problem1
```
finalize_problem1使用已有审计表处理21条候选；词级不可信处保留独立音视频窗口，文字词时标留空。79条自动词锚不冒充人工确认。需要重建附件4定位时运行python -m src.align_attachment4。

## 训练与日志
```powershell
python -m src.train --model gated_bigru --text-source teacher --seed 43 --hidden 64 --batch-size 32 --epochs 12 --patience 4 --lr 0.0005 --regression-weight 0.4 --consistency-weight 0.1 --mask-protocol bert_reencoded_20260924 --run-tag reproduction
```
同口径实验种子42、43、44；原训练日志随包保留。此命令用于核查过程，不能据新结果重新挑选已锁定模型。
9月24日选型JSON的test_evaluated=false是当时状态；9月25日已经完成后续测试，见outputs/logs/final_test_bertmask_seed43_20260925.json。打包只补齐既有记录，不再次评测测试集。
原日志中的历史权重哈希对应原检查点；打包只清理路径字符串，权重张量未改。SOURCE_WEIGHT_AUDIT.json记录原文件与打包文件的哈希对应。
2026年9月29日修包：BERT配置文件保留上游原始字节，获取脚本按原始SHA-256核验；测试审计脚本按当前包内检查点的SHA-256核验，校验失败仍拒绝加载。已存在的最终测试日志继续受防覆盖保护，修包不重新运行测试集或改写成绩。

## 复现依赖与边界
BERT与Whisper仅为公开冻结特征提取依赖；不引入外部情感数据训练。附件内不含原始赛题数据、完整Conda环境或论文PDF。
支持材料不改变已提交论文。MANIFEST.json列出文件和SHA-256，PACKAGE_CHECK.json记录本次打包核查；所有数值来自已有实验或明确标注的复现检查。
