"""Standalone worker executed by the isolated MLX TTS Python interpreter."""

from __future__ import annotations

from contextlib import redirect_stdout
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["preset", "reference_clone"]
    size: Literal["0.6B", "1.7B"]
    text: str = Field(min_length=1)
    output_path: str
    language: str = "Chinese"
    instructions: str = ""
    voice: str = ""
    reference_audio: str = ""
    reference_text: str | None = None

    @model_validator(mode="after")
    def check_voice(self) -> SpeechRequest:
        if self.kind == "preset" and not self.voice:
            raise ValueError("Preset synthesis requires a voice.")
        if self.kind == "reference_clone" and not self.reference_audio:
            raise ValueError("Voice cloning requires reference audio.")
        return self

    @property
    def model_repo(self) -> str:
        kind = "Base" if self.kind == "reference_clone" else "CustomVoice"
        return f"mlx-community/Qwen3-TTS-12Hz-{self.size}-{kind}-bf16"


class SpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str | None = None


class SpeechEngine:
    def __init__(self) -> None:
        self.model = None
        self.repo = ""

    def generate(self, request: SpeechRequest) -> None:
        import gc

        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.tts.utils import load_model

        if not mx.metal.is_available():
            raise RuntimeError("Apple GPU is unavailable in the MLX TTS worker.")
        if self.repo != request.model_repo:
            self.model = None
            gc.collect()
            mx.clear_cache()
            self.model = load_model(request.model_repo)
            self.repo = request.model_repo
        kwargs = {"voice": request.voice, "instruct": request.instructions or None}
        if request.kind == "reference_clone":
            kwargs = {
                "ref_audio": request.reference_audio,
                "ref_text": request.reference_text,
            }
        chunks = []
        sample_rate = None
        with mx.stream(mx.gpu):
            for result in self.model.generate(
                text=request.text,
                lang_code=request.language,
                stream=False,
                **kwargs,
            ):
                if sample_rate is not None and sample_rate != result.sample_rate:
                    raise RuntimeError("MLX TTS returned inconsistent sample rates.")
                sample_rate = result.sample_rate
                chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not chunks or not any(chunk.size for chunk in chunks):
            raise RuntimeError("MLX TTS returned no audio.")
        # This is a temporary file. The parent publishes it atomically.
        sf.write(request.output_path, np.concatenate(chunks), sample_rate, format="WAV")


def main() -> None:
    engine = SpeechEngine()
    for line in sys.stdin:
        try:
            request = SpeechRequest.model_validate_json(line)
            with redirect_stdout(sys.stderr):
                engine.generate(request)
            response = SpeechResponse()
        except Exception as exc:
            response = SpeechResponse(error=f"{type(exc).__name__}: {exc}")
        print(response.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
