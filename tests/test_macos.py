from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from podtran import macos
from podtran.asr import _transcribe_mlx
from podtran.config import AppConfig, ASRConfig
from podtran.fingerprints import (
    FingerprintService,
    SYNTHESIZE_CONFIG_KEYS,
    TRANSCRIBE_CONFIG_KEYS,
    VOICE_CLONE_CONFIG_KEYS,
)


@pytest.mark.parametrize("requested", ["auto", "gpu", "mps", "mlx"])
def test_apple_gpu_selection(monkeypatch, requested):
    monkeypatch.setattr(macos, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(macos, "apple_gpu_available", lambda: True)
    assert macos.apple_device(requested) == "mlx"


def test_auto_falls_back_only_before_execution(monkeypatch):
    monkeypatch.setattr(macos, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(macos, "apple_gpu_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="using CPU"):
        assert macos.apple_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="Apple GPU"):
        macos.apple_device("gpu")


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "xpu"])
def test_explicit_existing_devices_do_not_probe_apple_gpu(monkeypatch, device):
    probe = Mock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr(macos, "apple_gpu_available", probe)
    assert macos.apple_device(device) == device


def test_effective_backends_separate_stage_and_clone_fingerprints(
    monkeypatch, tmp_path
):
    config = AppConfig()
    fingerprints = FingerprintService(tmp_path)
    monkeypatch.setattr(macos, "is_apple_silicon", lambda: True)
    keys = [TRANSCRIBE_CONFIG_KEYS, SYNTHESIZE_CONFIG_KEYS, VOICE_CLONE_CONFIG_KEYS]
    monkeypatch.setattr(macos, "apple_gpu_available", lambda: True)
    gpu = [fingerprints.hash_config_subset(config, item) for item in keys]
    resolved = macos.resolve_tts_config(config)
    assert resolved.tts_clone_model() == "qwen-local:mlx:base:0.6B"
    assert config.providers.qwen_local.device == "auto"
    monkeypatch.setattr(macos, "apple_gpu_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        cpu = [fingerprints.hash_config_subset(config, item) for item in keys]
    assert all(left != right for left, right in zip(gpu, cpu))


@pytest.mark.parametrize("fails", [False, True])
def test_mlx_transcription_releases_global_model_on_success_and_failure(
    monkeypatch, fails
):
    core = ModuleType("mlx.core")
    core.gpu = object()
    core.stream = Mock(return_value=nullcontext())
    core.clear_cache = Mock()
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    holder = SimpleNamespace(model=object(), model_path="old-model")
    module = ModuleType("mlx_whisper")
    module.transcribe = Mock(return_value={"segments": []})
    if fails:
        module.transcribe.side_effect = RuntimeError("GPU ran out of memory")
    child = ModuleType("mlx_whisper.transcribe")
    child.ModelHolder = holder
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", child)
    if fails:
        with pytest.raises(RuntimeError, match="out of memory"):
            _transcribe_mlx("audio", ASRConfig())
    else:
        assert _transcribe_mlx("audio", ASRConfig()) == {"segments": []}
    assert holder.model is None
    assert holder.model_path is None
    core.clear_cache.assert_called_once()
    assert (
        module.transcribe.call_args.kwargs["path_or_hf_repo"]
        == "mlx-community/whisper-medium-mlx"
    )


def test_mlx_transcription_keeps_alignment_and_diarization_on_cpu(monkeypatch):
    from podtran.asr import transcribe_audio

    monkeypatch.setattr(macos, "apple_gpu_available", lambda: True)
    whisperx = SimpleNamespace(
        load_audio=Mock(return_value="audio"),
        load_model=Mock(side_effect=AssertionError("must use MLX")),
        load_align_model=Mock(return_value=("aligner", {})),
        align=Mock(return_value={"segments": []}),
        DiarizationPipeline=Mock(return_value=Mock(return_value=[])),
        assign_word_speakers=Mock(
            return_value={
                "segments": [
                    {"start": 0, "end": 1, "text": "hello", "speaker": "SPEAKER_00"}
                ]
            }
        ),
    )
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    transcribe = Mock(
        return_value={
            "language": "en",
            "segments": [{"text": "hello", "start": 0, "end": 1}],
        }
    )
    monkeypatch.setattr("podtran.asr._transcribe_mlx", transcribe)
    result = transcribe_audio(Path("audio.wav"), ASRConfig(device="gpu"), "token", 2, 3)
    assert result[0].speaker == "SPEAKER_00"
    assert whisperx.load_align_model.call_args.kwargs["device"] == "cpu"
    assert whisperx.align.call_args.args[4] == "cpu"
    assert whisperx.DiarizationPipeline.call_args.kwargs["device"] == "cpu"
    assert whisperx.DiarizationPipeline.return_value.call_args.kwargs == {
        "min_speakers": 2,
        "max_speakers": 3,
    }


def test_gpu_probe_checks_mlx_metal_not_torch(monkeypatch):
    monkeypatch.setattr(macos, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(macos.platform, "mac_ver", lambda: ("14.0", (), "arm64"))
    core = ModuleType("mlx.core")
    available = Mock(return_value=False)
    core.metal = SimpleNamespace(is_available=available)
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    assert macos.apple_gpu_available() is False
    available.return_value = True
    assert macos.apple_gpu_available() is True
