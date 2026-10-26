---
status: accepted
---

# Apple Silicon 优先采用 MLX 推理

2026-10-23：macOS 首版要求转录和 Qwen TTS 声音克隆使用 Apple GPU，对齐和说话人区分允许使用 CPU。现有 CTranslate2 转录不能直接使用 MPS，因此选择 MLX Whisper 和社区维护的 MLX Audio Qwen3-TTS 作为 Mac 的优先路线，接受独立适配及转换后的模型权重；Windows 和 Linux 保持原有路线。

这项决策确定技术方向，不代表依赖组合已经可安装或实机效果已经验证。现有 Qwen、WhisperX 与所检查 MLX Audio 版本存在依赖冲突，因此由 podtran 管理独立 MLX TTS 环境和子进程，主环境保留原有 CPU 后端，用户继续使用同一套 CLI。实现固定使用 mlx-whisper 0.4.3 和 mlx-audio 0.6.1；主环境锁文件与独立 TTS 依赖清单均已按 macOS 14、Python 3.11 解析通过。原生 Qwen MPS 的现有资料不足以保证完整克隆流程，因此暂不作为首选。

显式指定 GPU 时，不可用就报错；默认自动选择模式须说明原因并回退 CPU。按用户优先沿用现有逻辑的要求，设备选择限于启动阶段，不新增运行中 GPU 故障自动切换 CPU。默认模型规模沿用 Whisper medium 和 Qwen3-TTS 0.6B，不自动降低规模或切换量化权重。交付先通过自动化测试，真实音频效果由用户人工确认。

一手证据与版本约束见 [GPU 调研记录](../macos-gpu-research.md)。

实现与自动化验证结果见 [实施验证记录](../macos-validation.md)。真实模型效果仍待人工验收。
