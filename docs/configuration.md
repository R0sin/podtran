# 配置指南

[返回 README](../README.md) · [任务与维护](usage.md) · [常见问题](troubleshooting.md)

先运行 `podtran init` 生成配置，再按需修改 `~/.podtran/config.toml`。以下代码都是配置片段，请修改已有的对应表，不要重复添加同名 TOML 表。

## 跳过赞助口播

处理 YouTube 节目时，加上 `--sponsorblock` 可以去掉其他用户已标记的赞助口播，保留片头、片尾和求订阅等内容。这个功能默认关闭：

```powershell
podtran run "https://www.youtube.com/watch?v=VIDEO_ID" --sponsorblock
```

如果希望以后自动去掉赞助口播，在 `config.toml` 中设置：

```toml
[download]
sponsorblock = true
```

临时想保留完整节目，可以加 `--no-sponsorblock`。这两个命令行选项只影响本次任务，不会修改配置文件。

没有人标记过的广告仍会保留；没有找到标记时，会继续翻译完整节目。其他网站和本地文件不会去广告。如果连接 SponsorBlock 多次失败，或剪切音频失败，任务会停止并显示错误，之后可用 `podtran resume` 重试。

加上 `--preview` 时，会试听去掉赞助口播后的前五分钟。`podtran resume` 会沿用该任务原先的选择，已经处理好的音频也不会重新剪切。如果想改变是否去广告，或使用后来新增的广告标记，请重新运行 `podtran run`。

## 网络代理

在 `config.toml` 顶层（任何 `[section]` 之前）配置无认证 HTTP 代理：

```toml
proxy = "http://192.168.1.10:7890"
no_proxy = ["localhost", "127.0.0.1", "::1", "tts.internal"]
```

HTTP 代理也能用于 HTTPS 网站。代理覆盖音频 URL 下载、翻译、远程配音、声音克隆和模型下载；不支持在该配置中填写 SOCKS 或用户名密码。

```powershell
podtran run "https://www.youtube.com/watch?v=VIDEO_ID" --proxy http://192.168.1.10:7890
podtran resume --proxy http://192.168.1.20:7890
podtran resume --no-proxy
```

`run`、音频快捷入口、`resume` 及四个单阶段命令均支持互斥的 `--proxy` / `--no-proxy`。优先级为：本次参数 > 任务保存的临时设置 > 当前配置 > 标准代理环境变量。显式参数会保存到任务，后续恢复沿用；`--no-proxy` 保存强制直连设置。后台进程继承本次设置。

配置中的 `proxy = ""` 表示继承环境变量，不表示强制直连。未显式配置代理时保留依赖库原有的环境代理行为。绕过规则每次读取当前配置，并追加环境变量 `NO_PROXY`；默认绕过本机地址。代理失败按现有规则重试并报错，不自动切换直连。修改代理不使已有内容缓存失效。

使用显式代理或强制直连时，Hugging Face 模型下载采用普通 HTTP 路径，禁用 Xet / hf_transfer 下载加速，以统一网络行为。

## 翻译服务

- `google-free`：默认选项，免费，无需 API key；走 Google 非公开网页接口，可能受地区、风控、请求频率影响
- `bing-free`：免费，无需 API key；走 Bing 中国站网页接口，单个片段超过 1000 字符时会自动拆分并重组
- `openai-compatible`：适合自建、DashScope compatible-mode 或其他兼容 OpenAI Chat Completions 的翻译端点；需要设置 `providers.openai_compatible.translation_base_url`

如果你手动编辑 `config.toml`，最常见的翻译配置是：

```toml
[translation]
provider = "google-free"  # 默认，无需 API key
```

国内网络可以先尝试 Bing 网页渠道：

```toml
[translation]
provider = "bing-free"
```

如果你想切到 DashScope compatible-mode，可改成：

```toml
[translation]
provider = "openai-compatible"

[providers.openai_compatible]
translation_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
translation_api_key = "sk-..."
translation_model = "qwen-flash"
```

## 语音合成（TTS）

`mode = "auto"` 时，`openai-compatible` 使用预置音色，其余后端使用音色克隆。也可显式设置 `preset` 或 `clone`；`openai-compatible` 仅支持 `preset`。

### 本地 Qwen（默认）

默认 TTS 后端会直接在 podtran 进程内运行 Qwen3-TTS，需要安装 `qwen-local` 可选依赖。使用 `uv tool install` 安装 CLI 时，请按 [README 的安装命令](../README.md#安装)启用 `podtran[qwen-local]`。

如果是在源码 checkout 中开发或运行，可同步 extra：

```powershell
uv sync --extra qwen-local
```

然后配置：

```toml
[tts]
provider = "qwen-local"
mode = "auto"
batch_size = 4
max_concurrency = 1

[providers.qwen_local]
clone_model_size = "0.6B"
preset_model_size = "0.6B"
device = "auto"
language = "Chinese"
```

`batch_size` 可以根据设备资源微调；显存或内存紧张时调小，资源更充足时可以适当调大。

### DashScope

```toml
[tts]
provider = "dashscope"
mode = "auto"

[providers.dashscope]
api_key = "YOUR_DASHSCOPE_API_KEY"
```

支持预置音色和服务端音色克隆；具体模型可在初始化生成的配置中调整。

### OpenAI-compatible

使用兼容 OpenAI 的语音接口时，需填写服务商提供的地址、模型和支持的音色。TTS 与翻译的连接配置分别保存。

```toml
[tts]
provider = "openai-compatible"
mode = "preset"

[providers.openai_compatible]
tts_base_url = "https://YOUR_HOST/v1"
tts_api_key = "YOUR_API_KEY"
tts_model = "YOUR_TTS_MODEL"

[tts.preset]
fallback_voices = ["YOUR_VOICE"]
```

### vLLM-Omni

如果你准备自己部署 `vllm-omni` 的 `Qwen3-TTS` 服务，可先看这些官方资料：

- `vLLM-Omni` 文档：[Installation / Quickstart](https://vllm-omni.readthedocs.io/)
- `vLLM-Omni` 仓库：[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)
- `Qwen3-TTS` 官方说明：[QwenLM/Qwen3-TTS 的 vLLM Usage](https://github.com/QwenLM/Qwen3-TTS)

对 `podtran` 来说，只需要一个可访问的 `vllm-omni` TTS 服务，并把 `providers.vllm_omni.base_url` 指向它；默认示例地址是 `http://localhost:8091/v1`。

```toml
[tts]
provider = "vllm-omni"
mode = "auto"

[providers.vllm_omni]
base_url = "http://localhost:8091/v1"
```

### MiMo

如果你想使用 MiMo-V2.5-TTS 或 MiMo-V2.5-TTS-VoiceClone，可配置：

```toml
[tts]
provider = "mimo"
mode = "auto"

[providers.mimo]
api_key = "..."
base_url = "https://api.xiaomimimo.com/v1"
preset_model = "mimo-v2.5-tts"
clone_model = "mimo-v2.5-tts-voiceclone"
preset_voice = "mimo_default"
audio_format = "wav"
instructions = ""
```

API key 也可通过 `MIMO_API_KEY` 环境变量提供；如使用其他服务入口，请以账户对应的地址为准。

## 转写与硬件

转写相关设置默认来自 `config.toml` 里的 `[asr]` 配置：

```toml
[asr]
model = "medium"
compute_type = "int8"
device = "auto"
batch_size = 4
```

默认组合为 `medium + auto + int8`。`auto` 在 PyTorch 和 CTranslate2 均可使用 CUDA 时选择第一张可见 GPU，否则使用 CPU；ASR 不自动选择 XPU。已有配置中显式设置的 `cpu` 继续生效。ASR 与本地 TTS 分别通过 `[asr].device` 和 `[providers.qwen_local].device` 配置。

可选值参考：

- `model`：`base`、`small`、`medium`、`large-v2`、`large-v3`、`turbo`、`distil-large-v3`
- `compute_type`：`int8`、`float16`
- `device`：`auto`、`cpu`、`cuda`、`cuda:N`；`cuda` 使用第一张可见 GPU，`cuda:N` 指定可见设备编号（从 0 开始，受 `CUDA_VISIBLE_DEVICES` 影响）。转录、对齐和说话人分离使用同一设备。

转写进度会显示最终设备。选定设备后，模型加载失败或显存不足会直接报错，不自动切换 CPU 重试；需要时可手动设置 `device = "cpu"`。`auto` 不改变 `compute_type`，也不按实际设备拆分转录缓存。

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

安装时选择 PyTorch 后端与运行时选择 ASR 设备是两个步骤，使用 CUDA 还需要兼容的驱动及运行库。上述 GPU 示例中的批量大小应按可用显存调整。

## 输出模式与倍速

默认 `mode = "interleave"`，保留英文原声并穿插中文配音。

如果在 `[compose]` 里设置 `mode = "replace"`，完整任务会生成 `<原文件名>.replace.mp3`，表示只保留中文配音；预览任务的文件名另带 `.preview` 标记。

可在配置文件的 `[compose]` 中分别调整英文原声和中文译声的播放倍速：

```toml
[compose]
english_speed = 1.0
chinese_speed = 1.0
```

两项默认均为 `1.0`，允许范围为 `0.5–2.0`（含边界），保持音高；例如 `1.25` 表示加快到 1.25 倍速。
按原声和译声区分，中文译声中夹杂的英文词语仍采用中文倍速。`replace` 模式只使用中文倍速。
原声中的停顿、音乐和片尾一起变速，程序额外插入的切换停顿、段间静音及缺失配音替代静音保持原时长。
修改倍速后可运行 `podtran compose TASK` 重新合成；已有转录、翻译和 TTS 音频可继续复用。

其中 `TASK` 替换为实际任务 ID 或唯一前缀。
