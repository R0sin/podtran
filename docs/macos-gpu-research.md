# macOS Apple GPU 可行性调研

调研日期：2026-10-23。范围：Apple Silicon、macOS 14+、原生 arm64 Python 3.11；转录和 Qwen TTS 必须使用 Apple GPU，词对齐及说话人分离允许 CPU。此处只做资料和源码核验，未安装依赖、下载模型或声称完成实机验证。

## 决策结论

存在有上游实现的技术路线：MLX Whisper 做 GPU 转录，WhisperX 继续做 CPU 词对齐和说话人分离，MLX Audio 的 Qwen3-TTS Base 做 GPU 声音克隆。它满足功能方向，但目前不能称为已确认可直接安装的整体方案：现有依赖与最新 MLX Audio 有明确冲突，必须先完成依赖版本选择或隔离设计。不要承诺仅增加 `mps` 设备选项就能实现全部 GPU 加速。

建议将这条 MLX 路线作为优先设计方案，并先解决安装组合；原生 `qwen-tts` 的 PyTorch MPS 路线保留为候选，证据不足以直接承诺首版质量或性能。上述排序是基于以下来源的工程判断。

## 已确认的上游能力

| 组件 | 证据与判断 |
|---|---|
| 现有 WhisperX 转录 | 依赖 faster-whisper/CTranslate2。CTranslate2 预编译版本支持 ARM64 CPU，但 GPU 清单仅 NVIDIA/CUDA；不能把 `device=mps` 传给现有转录器来获得 Apple GPU。[硬件支持](https://opennmt.net/CTranslate2/hardware_support.html) |
| MLX Whisper | Apple 的 MLX Examples 仓库提供 `mlx-whisper`；API 支持模型路径/Hub 仓库、转录结果及词级时间戳。是已实现的 Apple Silicon 转录候选。[README](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md)、[转录源码](https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py) |
| 保留 CPU 对齐/分离 | WhisperX `align` 接收时间段文本、音频及独立 device；`DiarizationPipeline` 独立接受 device，并有 `assign_word_speakers`。因此把 MLX 转录段适配为 WhisperX 输入、后续显式 CPU，有源码支持，但 podtran 适配及结果质量仍需验证。[alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)、[diarize.py](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py) |
| MLX Qwen 声音克隆 | MLX Audio 维护者文档明确列出 Qwen3-TTS Base 0.6B/1.7B，通过 `ref_audio` + `ref_text` 克隆；CustomVoice 用于预设声音。这是社区维护的 MLX 实现，不能称为 Qwen 团队保证的 macOS 后端。[模型指南](https://blaizzy.github.io/mlx-audio/models/tts/qwen3-tts/) |
| MLX 安装基线 | 当前官方文档列出 Apple Silicon、原生 Python >=3.10、macOS >=14.0；发布工作流仍构建 macOS 14.0 wheel。用户选定的 14+/3.11/arm64 与 MLX 基线一致；这不等于所有音频依赖都已通过安装验证。[安装文档](https://ml-explore.github.io/mlx/build/html/install.html)、[发布工作流](https://github.com/ml-explore/mlx/blob/main/.github/workflows/release.yml) |

## 原生 Qwen MPS 的证据边界

Qwen 官方模型源码有针对 `mps` 的分支，因此不能断言原生 Qwen 完全不支持 MPS。但官方 demo 的 device 帮助只列 CPU/CUDA，默认 CUDA、bfloat16、FlashAttention；当前找到的官方 README 没有明确 macOS 支持矩阵或端到端克隆验收。源码出现 MPS 分支不是声音克隆全链路通过证明。推断：值得作为低改动候选，但不宜在未补齐证据时直接作为唯一保证路线。

来源：[Qwen 模型源码](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)、[官方 demo](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/cli/demo.py)、[官方 README](https://github.com/QwenLM/Qwen3-TTS)。

## 已发现的安装阻碍

本仓库当前锁定 `qwen-tts==0.1.1`、`transformers==4.57.3`、`whisperx==3.8.5`（`uv.lock`）。

- Qwen 0.1.1 上游明确要求 `transformers==4.57.3`。[依赖声明](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml)
- MLX Audio 当前 main 要求 `transformers>=5.14.0`、`huggingface_hub>=1.0`；较早的 v0.3.1 也要求 `transformers==5.0.0rc3`。这两个已核验版本均不能直接与现有 Qwen 依赖同装。[main 依赖](https://github.com/Blaizzy/mlx-audio/blob/main/pyproject.toml)、[v0.3.1 依赖](https://github.com/Blaizzy/mlx-audio/blob/v0.3.1/pyproject.toml)
- WhisperX 3.8.5 要求 `huggingface-hub<1.0.0`，也与 MLX Audio main 冲突。仅对 macOS 排除 `qwen-tts` 尚不足以解决全部冲突。[WhisperX 3.8.5 依赖](https://github.com/m-bain/whisperX/blob/v3.8.5/pyproject.toml)

下一步应核验可用的已发布 MLX Audio/WhisperX 版本组合，或将 MLX TTS 放入独立运行环境。不要强行忽略依赖约束。main 分支资料用于能力调查，不是建议直接安装未发布版本。尚未执行依赖解析，因此没有给出未经验证的精确安装命令。

## podtran 适配与验收边界

本地源码显示 `asr.py` 把一个 device 同时传给转录、对齐和分离；需要拆开转录后端与后处理 device。`tts.py` 的自动设备检测只有 CUDA/XPU/CPU，现有 Qwen 接口与 MLX Audio 的生成接口不同；不能仅改字符串。

推荐在自动化测试中验证：平台/后端路由、GPU 不可用行为、MLX 段转换和 CPU 后处理调用、参考音频及参考文本传递、音频结果及采样率、模型释放、后台停止/恢复、fingerprint 隔离。变更应把实际后端和模型标识纳入输出语义，按 AGENTS.md 更新阶段版本；不可让 MLX 与原生后端误复用不兼容结果。

MLX Whisper 当前源码使用全局 ModelHolder 缓存模型；只 `del` 调用方局部变量不会保证释放模型。应明确运行边界或清理机制，避免转录模型与 TTS 模型同时常驻。该点来自[上游转录源码](https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py)，需在实现时按固定版本复核。

按用户要求，交付先以自动化测试通过为目标；音质、克隆相似度、真实 GPU 使用与耗时由用户人工确认。测试通过不能写成实机性能已经验证。

## 仍需决定

1. 是否接受 macOS 使用社区维护的 MLX Qwen 后端及转换后的模型权重；保持 Windows/Linux 原路线。
2. 若找不到兼容的单环境版本组合，是否接受独立 TTS 环境；先由实现调查给出具体方案，避免把内部依赖细节转嫁用户。
3. GPU 不可用时是否明确失败并给出操作建议（建议），还是用户显式选择 CPU；不应自动静默回退而仍称 GPU 运行成功。
4. Mac 默认 Qwen 模型大小及量化方案：首版可沿用现有 0.6B/1.7B 选择语义，但量化会影响输出和缓存，需明确而不能自动混用。

## 追加核验：已发布版本与 CPU 回退（2026-10-23）

用户后续已接受 macOS 优先使用 MLX；设备策略已确定：显式要求 GPU 而不可用时失败，`auto` 检测不可用时警告并回退 CPU。上文“仍需决定”第 1、3 项由此关闭。

### 稳定发行版依赖结论

本轮未找到可推荐的、保留稳定 WhisperX 且同环境运行 MLX Qwen 的版本组合。证据进一步收窄如下：

- PyPI 当前稳定 MLX Audio 为 **0.6.1，2026-10-20 发布**；其 tag 的依赖与前述 main 相同，包括 MLX 0.32.4、Transformers >=5.14.0、Hub >=1.0。[PyPI](https://pypi.org/project/mlx-audio/)、[v0.6.1 元数据](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/pyproject.toml)
- PyPI 当前稳定 WhisperX 为 **3.8.6，2026-05-25 发布**，仍要求 Hub <1.0。PyPI 有 2026-06-26 发布的 **3.8.7rc1 预发行版**；当前 main 的版本号也为 3.8.7rc1，已放宽 Hub 下限为 >=0.28.1，但本轮未验证该发布 wheel 是否与 main 完全相同，也未验证运行兼容性。不能将这个候选说成稳定版解决方案。[PyPI](https://pypi.org/project/whisperx/)、[v3.8.6 元数据](https://github.com/m-bain/whisperX/blob/v3.8.6/pyproject.toml)、[当前 main](https://github.com/m-bain/whisperX/blob/main/pyproject.toml)
- 较早稳定 MLX Audio **0.3.0、0.3.1、0.4.0** 都固定 Transformers 5.0.0rc3；后者自身要求 Hub >=1.3.0,<2.0，因此不能仅凭 MLX Audio 表面的 Hub >=0.27 下限认为它能与 WhisperX 3.8.6 共存。[v0.3.0](https://github.com/Blaizzy/mlx-audio/blob/v0.3.0/pyproject.toml)、[v0.4.0](https://github.com/Blaizzy/mlx-audio/blob/v0.4.0/pyproject.toml)、[Transformers rc3](https://github.com/huggingface/transformers/blob/v5.0.0rc3/setup.py)
- MLX Audio 0.2.9/0.2.10 声明 Transformers <5，但 0.2.10 的发布日期为 2026-01-06，早于 Qwen3-TTS 发布日 2026-01-22，且该 tag 的 Qwen3-TTS 模型文件查询为 404，故不能把回退 0.2.10 当作已具备 Qwen3 克隆的解决办法。[版本历史](https://pypi.org/project/mlx-audio/)、[v0.2.10 元数据](https://github.com/Blaizzy/mlx-audio/blob/v0.2.10/pyproject.toml)、[Qwen 发布说明](https://github.com/QwenLM/Qwen3-TTS)

未逐个遍历所有中间版本，以上结论是“未找到可信组合”，不是数学意义证明不存在。直接 Python 请求 PyPI JSON 因当前工具网络 DNS 限制失败；通过网页读取补齐关键 tag 元数据，没有进行 solver 验证或安装。

### 建议把隔离缩到 MLX TTS

这是架构建议，不是已落实实现：主环境继续保留现有稳定 WhisperX、原生 qwen-tts 和 CPU 回退；MLX Whisper 可作为主环境的 macOS 转录依赖候选；仅 MLX Qwen TTS 放入由 podtran 管理、固定版本的独立 Python 环境/子进程。这样隔离 Transformers/Hub 冲突，也保留已经存在的原生 Qwen CPU 路线。仍需验证完整依赖锁、MLX Whisper 的具体发布版依赖及所有 arm64 wheel。

另一条候选是采用 WhisperX 3.8.7rc1、Mac 排除原生 Qwen，再将 MLX Audio 与转录放同环境；代价是引入预发行依赖，并且 CPU TTS 回退仍未证实。因此当前更建议稳定依赖隔离方案，除非用户明确希望先评估预发行路线。

### CPU 回退不能只假定切换 mx.cpu

MLX 官方提供 CPU/GPU stream 选择，但框架支持 CPU 不等于 MLX Audio Qwen 的完整声音克隆链路已验证 CPU。Qwen 的 talker、codec 等是多层调用，还使用编译和 fast 运算；本轮未找到上游对完整 Qwen CPU 克隆的明确保证。[MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)、[Qwen talker](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/mlx_audio/tts/models/qwen3_tts/talker.py)、[Qwen codec](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py)

因此按用户已确认的回退行为设计时，推荐 GPU 不可用的 `auto` 分支转到现有 CPU WhisperX/原生 Qwen，并明确记录实际执行后端；不要把尚未证明的 MLX CPU 克隆当作交付保证。需要区分“启动前检测 GPU 不可用”与“已开始生成后 OOM/模型错误”：后者是否回退不应擅自扩展为吞掉任何异常，且后端变化必须对应缓存隔离。

## 实现接口复核（2026-10-23）

通过 PyPI 精确版本页复核 `mlx-audio==0.6.1`：wheel 和 sdist 均标记 2026-10-20 上传。wheel SHA256 为 `4c5e4ef3f2a453dcc1c84c9aaf1e43701f62f5a32632aff50bfa6be9047d0f10`。`mlx-whisper==0.4.3` 的 PyPI 发布日期为 2025-08-29。版本存在已核实，完整依赖解析仍未完成。[MLX Audio 精确版本页](https://pypi.org/project/mlx-audio/0.6.1/)、[MLX Whisper](https://pypi.org/project/mlx-whisper/)

以下基于 [MLX Audio v0.6.1 Qwen 源码](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py)：

- 加载：`from mlx_audio.tts.utils import load_model; model = load_model(repo_id)`。[加载源码](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/mlx_audio/tts/utils.py)
- 公共生成：`model.generate(text=..., voice=..., instruct=..., lang_code=..., ref_audio=..., ref_text=..., stream=False)`，返回生成器。`lang_code` 对应 `chinese`/`english`/`auto` 等语言名。
- Base 克隆：提供 `ref_audio` 和 `ref_text`；codec 必须有 encoder 才进入 ICL。引用文件路径可以直接传字符串，公共 generate 内部按 `self.sample_rate` 调用 `load_audio`。
- 没有命名为 `x_vector_only_mode` 的参数。根据源码，单条生成传 `ref_text=None` 会跳过 ICL，并从 ref_audio 提取 speaker embedding；这是 x-vector-only 的可实现映射，仍需针对固定版本测试。不能把该原生 Qwen 参数原样透传后假定生效。
- `batch_generate` 的 reference clone 强制同时存在 ref_audio/ref_text，因此 x-vector-only 不能直接套批量路径。
- CustomVoice 可经公共 generate 的 `voice`/`instruct`，或 `generate_custom_voice(text,speaker,language,instruct)`。0.6B（内部 `tts_model_size=='0b6'`）明确丢弃 instruct；1.7B 路径保留。
- Base ICL 不转发 instruct；不要承诺 clone 可用情绪指令。`speed` 也被文档标为尚未直接支持。
- `GenerationResult` 提供 `audio: mx.array`、`sample_rate: int` 等；应处理完整迭代结果，避免仅保存第一段。[结果模型](https://github.com/Blaizzy/mlx-audio/blob/v0.6.1/mlx_audio/tts/models/base.py)

四个 BF16 模型仓库逐个访问确认，均为 MLX 转换权重：

- [mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16)
- [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16)
- [mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16)
- [mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16)
