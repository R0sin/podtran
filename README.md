# podtran

把英文播客转成中文音频的命令行工具。默认保留英文原声并穿插中文配音，也可输出纯中文版本。

- **先试听**：用前 5 分钟音频预览翻译和配音效果。
- **可恢复**：转写、翻译、配音、合成分阶段执行，中断后从兼容的断点继续。
- **复用结果**：自动缓存转写、翻译、音色和逐段配音，减少重复处理。
- **灵活选择**：默认本地 WhisperX 转写、免费网页翻译、本地 Qwen TTS，也支持云端和自托管服务。

## 安装

需要先准备：

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 [Git](https://git-scm.com/downloads)；推荐 Python 3.11，支持 3.10–3.12。
- [FFmpeg](https://ffmpeg.org/download.html)，确保终端可执行 `ffmpeg` 和 `ffprobe`。
- [Hugging Face token](https://huggingface.co/settings/tokens)，并接受 [说话人区分模型的使用条款](https://huggingface.co/pyannote/speaker-diarization-community-1)。

安装默认的本地 TTS 版本：

```powershell
uv tool install --python 3.11 --torch-backend auto "podtran[qwen-local] @ git+https://github.com/R0sin/podtran"
podtran --help
```

`--torch-backend auto` 让 uv 根据设备选择 PyTorch 后端，详见 [uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。转写默认使用 CPU；首次运行需要下载模型，处理时间取决于硬件与音频长度。

**使用云端或自托管 TTS？** 可只安装基础依赖，再在初始化时选择对应服务：

```powershell
uv tool install --python 3.11 --torch-backend auto git+https://github.com/R0sin/podtran
```

远程 TTS 按服务要求配置地址和密钥；转写仍在本地运行。

## 快速开始

### 1. 初始化配置

```powershell
podtran init
```

跟随向导填写 Hugging Face token，选择翻译和 TTS 服务。默认使用 `google-free` 翻译和 `qwen-local` 配音，无需翻译或 TTS API key。配置保存在 `~/.podtran/config.toml`。

### 2. 先试听前 5 分钟

将 `podcast.mp3` 替换为你的音频文件路径：

```powershell
podtran run "podcast.mp3" --preview
```

预览只处理前 300 秒音频。确认说话人区分、翻译和配音效果后，再运行完整版：

```powershell
podtran run "podcast.mp3"
```

也可简写为 `podtran "podcast.mp3"`。默认按 2–5 位说话人识别；已知人数时可[指定说话人数量](docs/usage.md#说话人数量)。

也可以直接粘贴 YouTube 等网站的单集链接，自动下载并翻译：

```powershell
podtran "https://www.youtube.com/watch?v=VIDEO_ID" --background
podtran run "https://example.com/episode.mp3" --preview
```

下载工具会随 podtran 自动安装。支持的网站取决于 yt-dlp，请使用公开可访问的单集链接；暂不支持播放列表、频道页和直播。

链接预览会先下载完整音频，再处理前五分钟。下载中断后用 `podtran resume` 继续，无需重新提交链接。

支持赞助广告跳过参数 `--sponsorblock`，详见[使用说明](docs/configuration.md#跳过赞助口播)。

### 3. 找到输出

完成后，终端会显示输出文件路径。默认保存在：

```text
~/.podtran/artifacts/tasks/<task_id>/final/
```

| 任务 | 默认输出文件 |
| --- | --- |
| 完整音频 | `<原文件名>.interleave.mp3` |
| 5 分钟预览 | `<原文件名>.preview.interleave.mp3` |

`interleave` 表示英文原声与中文配音交替播放。需要纯中文或调整倍速，见[输出模式与倍速](docs/configuration.md#输出模式与倍速)。

## 查看与恢复任务

```powershell
podtran tasks         # 列出任务
podtran status        # 查看最近任务
podtran resume        # 继续最近任务
```

每次 `run` 都会创建新任务。**中断后请用 `resume`**，它会跳过仍有效的已完成阶段，并恢复兼容的翻译和配音进度。`status` 和 `resume` 也可接任务 ID 或唯一前缀。

长任务可加 `--background` 在后台运行，用 `podtran stop` 停止最近的后台任务；前台运行时使用 `Ctrl+C`。更多用法见[任务与维护](docs/usage.md)。

## 选择翻译与配音服务

在 `podtran init` 中选择，或按[配置指南](docs/configuration.md)修改配置文件。

| 用途 | 服务 | 说明 |
| --- | --- | --- |
| 翻译 | `google-free`（默认）、`bing-free` | 无需 API key，非官方网页接口可能受网络及频率限制影响 |
| 翻译 | `openai-compatible` | 兼容 Chat Completions 的服务，包括 DashScope |
| 配音 | `qwen-local`（默认） | 本地 Qwen3-TTS，默认 0.6B 模型，支持音色克隆和预置音色 |
| 配音 | `dashscope`、`mimo` | 云端服务，支持音色克隆和预置音色 |
| 配音 | `openai-compatible` | 兼容语音接口，仅支持预置音色 |
| 配音 | `vllm-omni` | 自托管服务，支持音色克隆和预置音色 |

## 更多文档

- [配置指南](docs/configuration.md)：网络代理、服务商配置、GPU 与批量大小、输出模式与倍速。
- [任务与维护](docs/usage.md)：后台运行、切换翻译服务、工作目录、单阶段执行、升级与开发。
- [常见问题](docs/troubleshooting.md)：安装、模型下载、权限、内存及翻译失败排查。
- [更新日志](CHANGELOG.md) · [问题反馈](https://github.com/R0sin/podtran/issues)

## 许可证

[MIT](LICENSE)
