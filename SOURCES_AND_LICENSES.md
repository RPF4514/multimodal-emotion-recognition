# 外部依赖来源
- BERT: google-bert/bert-base-uncased，Apache-2.0。https://huggingface.co/google-bert/bert-base-uncased
- 权重获取与哈希：src/fetch_verified_bert.py。权重SHA-256：097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a。
- BERT固定字节由哈希约束；上游文件改变时脚本拒绝使用。
- ASR: Systran/faster-whisper-tiny.en、Systran/faster-whisper-base.en（公开模型，faster-whisper 1.2.1）；https://huggingface.co/Systran/faster-whisper-tiny.en 及 https://huggingface.co/Systran/faster-whisper-base.en
- 代码框架：PyTorch、NumPy、pandas、OpenCV、FFmpeg、tokenizers，按各项目许可证使用。
- 赛题数据仅来自官方CMU-MOSEI附件。
- 公共预训练特征权重可提供版本、链接或脚本，不必随50MB附件上传：E题专家答疑 https://www.shumo.com/forum/forum.php?mod=viewthread&tid=251903 。
- 上传时段、命名与大小依据2026开赛公告：https://www.cmathc.org.cn/cpmcm/news/642.html 。
- 使用AI辅助编程、排查、文稿排版与材料整理；AI工具与使用范围的正式披露以已提交论文为准。
