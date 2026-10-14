# 多模态情感识别：时序对齐、缺失感知预测与反事实解释

基于 CMU-MOSEI 赛题附件的数学建模项目，处理文字、语音、视觉三模态信息。本文档整理的是2026年9月29日修复并核验过的最终代码和实验记录。

## 三个问题

1. **时序组织与质量审计**：保留全部100条原始样本，记录文本、音频、画面可追溯关系。对于已有人工审计中无法可靠词级对应的样本，文本独立编码，音视频按自身时间窗口处理，文字词时标留空。
2. **情感预测**：冻结 BERT 重建768维文本表示，三模态分别编码，经观测掩码约束的门控 BiGRU 融合，输出负向/中性/正向类别和[-3,3]强度。文字遮挡在词元端发生，随后重算全部上下文表示。
3. **预测解释**：复用锁定预测器，通过整模态与连续窗口遮挡计算目标类概率变化，输出模型依赖的证据与定位可信度。

## 目录

```text
src/                   数据接口、预测模型、训练、评估、解释及特征提取
configs/               三个候选模型的配置
tests/                 文字遮挡、路径解析和文件完整性检查
outputs/logs/          已有训练、验证、测试和复现记录
outputs/predictions/   附件3的30条预测、附件4的20条预测解释
outputs/bert/          BERT配置与词表；公开权重通过脚本获取
outputs/normalizer.npz 训练集拟合的标准化参数
docs/                  原提交复现说明、来源清单、打包检查与结果摘要
```

运行所需的自训检查点、问题一特征与时间映射、人工审计表、解释关键帧在单独的 `runtime-assets-v1.0.0.zip` 中。上传 GitHub 后，把此ZIP挂到同一仓库的 Release，下载者将ZIP内的 `outputs/` 合并至仓库根目录即可恢复既有输入路径。ZIP内的 `ASSET_MANIFEST.json` 给出逐文件SHA-256；不要把整个ZIP解压到 `outputs/` 内。

## 已有结果

完整结果表由 `docs/RESULTS_SNAPSHOT.md` 给出。数据来自随仓库保留的实验日志，验证集728条，测试集727条。附件3和附件4无标签，只报告预测与解释，不报告准确率。

测试结果是锁定seed-43模型的后续评价。该数据划分此前曾用于查看旧模型，因此不把它称为整个研发过程从未接触过的盲测。详细范围见最终测试JSON中的 `scope_note`。

## 环境与运行

实际环境为Python 3.10、PyTorch 2.6.0、CUDA 12.4；GPU为RTX 3060 Laptop GPU。完整版本见 `ENVIRONMENT.txt`。问题一重建及关键帧定位需要FFmpeg加入PATH。

```powershell
# 在仓库根目录执行；按PyTorch官方安装适合本机的2.6.0版本
python -m pip install -r requirements.txt

# 先解压runtime-assets-v1.0.0.zip，将其中outputs合并进仓库根目录
# 设置实际赛题数据路径
$env:MM_EMOTION_DATA_ROOT = 'D:\赛题数据\E题数据'

# 下载并严格校验公开冻结BERT权重（约440MB）
python -m src.fetch_verified_bert

# 含检查点核验的完整9项检查需要已经解压runtime-assets
python -m unittest discover -s tests -p 'test_*.py' -v

# 保留原正式结果，使用新的输出标签
$s = 'outputs/logs/model_selection_bertmask_20260924.json'
python -m src.explain --selection $s --output-tag github_rerun
python -m src.predict_with_bert_text --selection $s --output-tag github_rerun
```

问题一重建及训练命令见 `docs/submission_reproduction.md`。仓库内已经包含既有结果，阅读这些记录不要求先下载数据或运行模型。重新推理需要合法取得的赛题数据及上述权重；原数据目录在9月29日修包时已缺失，当次只完成文件核验、9项检查和合成输入前向，未重新跑30/20条真实样本。

## 来源与使用

冻结BERT及其他第三方依赖的来源见 `SOURCES_AND_LICENSES.md`。官方原始数据、440MB公开BERT权重和完整Conda环境不随此仓库分发。整理过程没有修改原模型张量、正式预测或实验成绩。

本仓库尚未为团队自有代码指定开源许可证。先用于私有备份；公开及选定许可证前，由团队共同确定代码与派生材料的发布范围。第三方内容仍适用其原许可。
