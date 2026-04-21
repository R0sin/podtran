# podtran

`podtran` 是一个面向长播客的分阶段翻译 CLI，目标是让你在一台普通笔记本上，也能把英文播客转成可听的中文版本。

它默认走这条流水线：

`transcribe -> translate -> synthesize -> compose`

特点：

- 本地用 `WhisperX` 做转写、对齐和说话人区分
- 翻译和 TTS 默认走 DashScope
- 默认支持 `clone` 音色克隆，也支持 `preset` 预置音色
- 每次运行都会创建独立 task，避免旧结果污染新结果
- 共享缓存会自动复用已完成的转写、翻译、声纹和逐段 TTS 结果
- 中断后可用 `podtran resume` 从断点继续，已完成的翻译不会丢失
- 支持 `--preview`，先用前 5 分钟低成本试跑

## 适合什么场景

- 你想把英文播客、访谈或对话音频翻成中文音频
- 你不想一次性跑一个黑盒脚本，而是希望看到每个阶段的结果
- 你希望失败后可以从某个阶段继续，而不是全部重来

## 运行前准备

- Python `3.11` 推荐，支持 `>=3.10,<3.13`
- `ffmpeg` 和 `ffprobe` 需要在 `PATH` 中可执行
- 需要一个 Hugging Face token 给 WhisperX diarization 使用
- 翻译默认需要 DashScope API key
- TTS 可选 `dashscope`、`openai-compatible` 或 `vllm-omni`，所需配置取决于 provider

## 安装

标准方式是直接从 Git 仓库安装 CLI，然后使用 `podtran` 命令：

```powershell
uv tool install git+https://github.com/R0sin/podtran
```

安装完成后可以先确认命令已可用：

```powershell
podtran --help
```

提示：CLI help 会把 `podtran run AUDIO` 作为正式入口展示，日常使用仍可直接写 `podtran AUDIO`。

升级：

```powershell
uv tool install git+https://github.com/R0sin/podtran
```

卸载：

```powershell
uv tool uninstall podtran
```

如果你希望优先使用 CPU-only PyTorch，可以先准备对应依赖环境，再安装 CLI。

## 快速开始

1. 初始化配置

```powershell
podtran init
```

`init` 现在会进入交互式向导，默认帮你生成一份完整配置。向导会提示你：

- 先去接受 Hugging Face 的 `speaker-diarization-community-1` 协议
- 填写 `hf_token`
- 填写 DashScope API key
- 确认翻译模型
- 选择 TTS provider，并按提示填写对应的 `base_url`、API key、mode 和 model

默认配置会写到 `~/.podtran/podtran.toml`。如果传 `--workdir <path>`，则会写到 `<path>/podtran.toml`，同时任务和缓存也会放到这个目录下。

TTS provider 说明：

- `dashscope`：支持 `preset` 和 `clone`
- `openai-compatible`：支持 `preset`
- `vllm-omni`：支持 `preset` 和 `clone`，需要配置 `tts.base_url`

如果你准备自己部署 `vllm-omni` 的 `Qwen3-TTS` 服务，可先看这些官方资料：

- `vLLM-Omni` 文档：[Installation / Quickstart](https://vllm-omni.readthedocs.io/)
- `vLLM-Omni` 仓库：[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)
- `Qwen3-TTS` 官方说明：[QwenLM/Qwen3-TTS 的 vLLM Usage](https://github.com/QwenLM/Qwen3-TTS)

对 `podtran` 来说，只需要一个可访问的 `vllm-omni` TTS 服务，并把 `tts.base_url` 指向它；默认示例地址是 `http://localhost:8091/v1`。

转录相关设置默认来自 `podtran.toml` 里的 `[asr]` 配置：

```toml
[asr]
model = "medium"
compute_type = "int8"
device = "cpu"
batch_size = 4
```

`medium + cpu + int8` 是面向普通笔记本的默认组合。如果有实力或想尝试不同效果，可以手动调整 `model`、`device` 和 `compute_type`。

可选值参考：

- `model`：`base`、`small`、`medium`、`large-v2`、`large-v3`、`turbo`、`distil-large-v3`
- `compute_type`：`int8`、`float16`

一般建议：

- CPU 环境优先用 `int8`
- CUDA 环境优先用 `float16`

以下是单卡 3090 的示例：

```toml
[asr]
model = "distil-large-v3"
compute_type = "float16"
device = "cuda"
batch_size = 16
```

2. 先跑一个 5 分钟预览

```powershell
podtran path\to\podcast.mp3 --preview
```

这一步就是首选验证方式。对长播客也一样，先用预览模式确认配置、说话人区分、翻译质量和 TTS 效果，再决定是否跑完整音频。

3. 预览效果没问题后，跑完整音频

```powershell
podtran path\to\podcast.mp3
```

4. 如果运行中断（Ctrl+C 或意外退出），用 `resume` 从断点继续

```powershell
podtran resume
```

`resume` 默认恢复最近一个 task。已完成的阶段会自动跳过，翻译阶段会从上次保存的进度继续翻译。也可以指定 task id：

```powershell
podtran resume 20260415-083242-50ed61
```

提示：如果翻译已经**完成**过，重新运行 `podtran <audio>` 也会通过共享缓存自动复用。但如果翻译**中途中断**，只有 `resume` 能恢复未完成的进度。

5. 查看最近任务状态

```powershell
podtran tasks
podtran status
```

## 你会得到什么

默认输出会放在工作目录下的 `artifacts/tasks/<task_id>/final/` 中：

- 完整任务通常生成 `<原文件名>.interleave.mp3`
- 预览任务通常生成 `<原文件名>.preview.interleave.mp3`

`interleave` 是默认模式，表示保留英文原声并穿插中文配音。

## License

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
