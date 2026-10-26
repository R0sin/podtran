# macOS 实施验证

日期：2026-10-23。

## 已完成

- Apple Silicon 自动检测 MLX Metal 能力，显式 GPU 不可用时报错，自动模式提示并选择 CPU。运行中不切换设备。
- MLX Whisper 转录，CPU 词对齐和说话人区分；转录结束或失败时清理上游全局模型引用。
- 独立 MLX TTS Python 3.11 环境，固定依赖、安装锁、未完成安装重建及持久子进程。
- 克隆和预设声音适配、完整音频片段收集、原子发布、取消和进程退出清理。
- 实际后端参与缓存指纹，更新相关阶段版本。
- macOS CI、安装说明、依赖检查脚本和安装包资源检查。

## 验证结果

- `.venv/bin/pytest -q`：344 项通过。后台进程测试在沙箱外执行，因为沙箱限制 psutil 的进程查询。
- `.venv/bin/ruff check src tests scripts`：通过。
- `.venv/bin/ruff format --check src tests scripts`：通过。
- `python scripts/check_macos_dependencies.py`：主环境锁文件 160 个包、独立 TTS 环境 41 个包，均按 macOS 14 / Python 3.11 解析成功。
- `uv build --no-sources` 和 `python scripts/check_distribution.py dist/podtran-0.4.1-py3-none-any.whl`：构建、临时环境安装、CLI 和 MLX worker 资源检查通过。
- CLI 导入未加载 torch、whisperx、qwen_tts、mlx.core、mlx_audio 或 soundfile。
- 固定发行包 mlx-whisper 0.4.3 和 mlx-audio 0.6.1 的源码已核对。

CI 使用 macOS 15 runner，仍以 macOS 14 为依赖解析目标。GitHub 宣布 macOS 14 runner 于 2026-11-02 停用，见 https://github.com/actions/runner-images/issues/13518 。GitHub 上的工作流尚未实际运行。

## 人工验收

按 README 的 macOS 安装步骤，在现有配置的 `[asr]` 和 `[providers.qwen_local]` 中设置 `device = "gpu"`，运行 `podtran run "podcast.mp3" --preview`。确认首次环境准备、真实 GPU 使用、音色与完整音频输出，再试后台停止及 resume。

本次未下载模型或执行真实模型推理，不对音质、峰值内存或处理速度作实测结论。
