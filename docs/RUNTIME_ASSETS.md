# 固定版本运行材料

位置：[artifacts/runtime-assets-v1.0.0.zip](../artifacts/runtime-assets-v1.0.0.zip)。

本次通过代码仓库正常推送这一个34.88MB固定归档，未创建Release。仓库保持私有。

- 文件大小：34,882,474字节。
- SHA-256：`3183640381b884423f25afab687d0334f4323f10b2d75552571707e37761ad04`。
- 归档内56个原始材料文件及1份ASSET_MANIFEST.json。
- 包含自训检查点、问题一派生特征、时间映射、已有21条人工审计表、关键帧及解释图。
- 56个文件逐项校验与9月29日核验版本一致。

## 下载与恢复

克隆本仓库或使用Code → Download ZIP，然后在仓库根目录执行：

```powershell
Get-FileHash -LiteralPath '.\artifacts\runtime-assets-v1.0.0.zip' -Algorithm SHA256
Expand-Archive -LiteralPath '.\artifacts\runtime-assets-v1.0.0.zip' -DestinationPath '.'
```

确认哈希与上文一致，再按README安装依赖、取得赛题数据及公开冻结BERT。含自训检查点的完整9项检查需要先完成上述解压。

模型代码、配置、标准化参数、正式日志和专项预测保存在仓库其他目录。论文PDF、官方原始数据、完整Conda环境及约440MB公开BERT权重不包含在此次备份中；BERT提供下载和校验脚本。

本归档用于固定版本保存。后续若反复更新大权重，应改用Release或专门的模型存储，避免Git历史持续膨胀。
