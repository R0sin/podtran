# 常见问题

[返回 README](../README.md) · [配置指南](configuration.md) · [任务与维护](usage.md)

## 找不到 podtran 命令

确认 `uv tool install` 成功完成。若工具目录未加入 PATH，运行 `uv tool update-shell`，重新打开终端后执行 `podtran --help`。

## 找不到 ffmpeg 或 ffprobe

安装 [FFmpeg](https://ffmpeg.org/download.html)，将包含 `ffmpeg` 和 `ffprobe` 的目录加入 PATH，重新打开终端并验证：

```powershell
ffmpeg -version
ffprobe -version
```

## 说话人区分提示无权限

确认已使用 token 所属账户接受 [speaker-diarization-community-1 的模型使用条款](https://huggingface.co/pyannote/speaker-diarization-community-1)，并在配置中填写有模型读取权限的 `hf_token`。

## 模型下载失败

如果新环境首次运行时报错类似：

```text
An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local disk.
```

通常表示本地还没有 Hugging Face 模型缓存，同时当前网络无法直接访问 Hugging Face Hub。这个错误发生在 WhisperX 加载 ASR 模型阶段，通常不是 `hf_token` 配错导致的。

先检查网络连接。如果需要第三方镜像，在 PowerShell 中可临时设置 Hugging Face Hub endpoint（镜像可用性取决于网络和服务状态）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
podtran resume
```

如果希望长期生效：

```powershell
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

重新打开 PowerShell 后确认变量已生效：

```powershell
echo $env:HF_ENDPOINT
```

## 运行慢或内存不足

首次运行需要下载模型；CPU 转写和本地 TTS 的耗时取决于硬件与音频长度。`--preview` 只截取前 300 秒，不能保证 5 分钟内处理完。

可先调小 `[asr].batch_size` 和 `[tts].batch_size`，或选择更小的转写模型。长音频仍可能在 WhisperX 加载时占用大量内存，此时可先拆分源音频。也可切换远程 TTS，但转写仍在本地运行。配置示例见[转写与硬件](configuration.md#转写与硬件)。

## 免费翻译接口失败

`google-free` 和 `bing-free` 使用非官方网页接口，可能受地区、网络和频率限制影响。可在配置好服务商后，使用 `podtran resume --translation-provider bing-free` 或 `openai-compatible` 继续处理剩余片段，详见[恢复任务](usage.md#恢复任务)。

## 旧配置无法加载

旧版配置字段会被拒绝。运行 `podtran init` 按向导重建；迁移旧 TTS 配置时会生成 `config.toml.bak`，并保留 Hugging Face token 和服务商 API key。

仍无法解决时，可在 [GitHub Issues](https://github.com/R0sin/podtran/issues) 提供 `podtran version` 输出、操作系统、运行命令和错误日志；请移除 token、API key 等敏感内容。
