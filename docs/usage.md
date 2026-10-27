# 任务与维护

[返回 README](../README.md) · [配置指南](configuration.md) · [常见问题](troubleshooting.md)

## 恢复任务

`podtran resume` 默认恢复最近一个任务，也可以指定任务 ID 或唯一前缀：

```powershell
podtran tasks
podtran status
podtran resume 20260415-083242-50ed61
```

上面的 ID 仅为示例，请换成 `podtran tasks` 列出的实际值。已完成且仍有效的阶段会跳过，翻译和 TTS 会继续使用兼容的部分结果。改变模型或其他影响输出的配置，可能使旧结果失效。

新运行 `podtran run AUDIO` 会创建新任务，并尝试复用共享缓存；要继续中途中断的进度，请使用 `resume`。

### 音频内容与缓存

音频缓存按解码后的采样内容、采样率、声道和时间信息识别。仅文件封装或元数据变化不会让转写缓存失效；重新编码后的有损音频即使听起来相同，也不保证命中。实际音频变化、静音长度变化或时间间隙变化会产生不同指纹。

首次读取或文件变化后，需要 FFmpeg 完整解码一次以计算指纹。计算按流进行，不把整段音频加载到内存；文件未变化时复用本地指纹索引。

每次运行 URL 仍会重新下载并创建独立任务，不复用其他任务的下载文件。下载后，音频内容相同且阶段配置相同即可复用共享缓存。

旧版文件哈希不迁移到新内容指纹。恢复旧任务时会刷新源音频和处理音频的指纹，相关阶段可能重新执行。刷新指纹本身不会删除旧缓存或成品。

开发时可运行 `uv run python scripts/check_audio_fingerprints.py`，用真实 FFmpeg 验证换封装、元数据、时间信息和音频内容变化。脚本也接受多个预期内容相同的音频路径，所有临时产物和索引均在临时目录内。

### 临时切换翻译服务

如果原翻译渠道无法处理剩余片段，可以只为本次恢复切换翻译 provider。已成功的片段会保留，新 provider 只处理失败或缺失的片段；model、endpoint 和密钥仍从当前配置读取：

```powershell
podtran resume 20260415-083242-50ed61 --translation-provider openai-compatible
```

## 后台运行与停止

```powershell
podtran run "podcast.mp3" --background
podtran status
podtran stop
podtran resume --background
```

`status`、`stop` 和 `resume` 省略任务 ID 时都选择最近任务，也支持指定 ID。`stop` 会验证后台进程归属，终止进程树并将任务标记为中断；前台任务使用 `Ctrl+C`。日志保存在任务目录的 `run.log` 中。

恢复时可以组合使用 `--background` 和 `--translation-provider`。

## 说话人数量

默认按 2–5 人进行说话人区分。已知人数时可缩小范围，例如单人播客：

```powershell
podtran run "podcast.mp3" --min_speakers 1 --max_speakers 1
```

## 自定义工作目录

`--workdir` 会一起移动配置、任务和共享缓存的位置。后续命令应使用同一个目录：

```powershell
podtran init --workdir "D:\podtran-data"
podtran run "podcast.mp3" --workdir "D:\podtran-data"
podtran resume --workdir "D:\podtran-data"
```

默认目录布局：

```text
~/.podtran/
  config.toml
  artifacts/
    tasks/<task_id>/
      transcript.json     # 转写
      segments.json       # 翻译前重建的分段
      translated.json     # 翻译与逐段进度
      voices.json         # 音色信息
      refs/               # 克隆参考音频
      tts/                # 逐段配音
      final/              # 最终 MP3
      manifests/          # 阶段状态
      run.log             # 运行日志
    cache/                # 跨任务共享缓存
```

## 单独运行一个阶段

以下 `TASK` 需替换为已有任务 ID 或唯一前缀。单阶段命令不会自动运行前置阶段。

| 命令 | 前提 |
| --- | --- |
| `podtran transcribe TASK` | 已有任务及其源音频 |
| `podtran translate TASK` | 已有 `transcript.json` |
| `podtran synthesize TASK` | 翻译已完成 |
| `podtran compose TASK` | 所需逐段配音已完成 |

例如，修改输出倍速后只需运行 `podtran compose TASK`。完整流程为 `transcribe → translate → synthesize → compose`。

## 升级与卸载

通过 uv 安装后，可使用 [uv 的工具升级命令](https://docs.astral.sh/uv/concepts/tools/#upgrading-tools)：

```powershell
uv tool upgrade podtran
```

如需重建安装环境，在 [README 中对应的安装命令](../README.md#安装)加上 `--force`。例如本地 TTS 版本：

```powershell
uv tool install --force --python 3.11 --torch-backend auto "podtran[qwen-local] @ git+https://github.com/R0sin/podtran"
```

卸载 CLI：

```powershell
uv tool uninstall podtran
```

## 清理缓存

`podtran cache clean` 清理共享缓存；按时间筛选等参数见 `podtran cache clean --help`。

## 从源码开发

```powershell
git clone https://github.com/R0sin/podtran
cd podtran
uv sync
uv run podtran --help
uv run ruff check src tests
uv run pytest -q
```

实际运行本地 Qwen TTS 时，改用 `uv sync --extra qwen-local` 安装可选依赖。测试使用模拟的外部服务，无需下载或运行真实模型。
