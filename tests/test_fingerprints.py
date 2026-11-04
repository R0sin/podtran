from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from podtran.fingerprints import FingerprintService
from podtran.models import AudioFormat


PCM = b"\x01\x02\x03\x04" * 1200


@pytest.fixture(autouse=True)
def audio_format(monkeypatch):
    monkeypatch.setattr(
        "podtran.audio.probe_audio_format",
        lambda path: AudioFormat(sample_rate=24000, channels=1),
    )


@pytest.fixture
def decoder(monkeypatch):
    calls = []
    processes = []

    def start(command, **kwargs):
        calls.append(command)
        path = Path(command[command.index("-i") + 1])
        process = FakeProcess(path.read_bytes())
        processes.append(process)
        return process

    monkeypatch.setattr("podtran.audio.subprocess.Popen", start)
    return calls, processes


class FakeProcess:
    def __init__(self, output):
        self.stdout = io.BytesIO(output)
        self.returncode = 0
        self.killed = False

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


def test_audio_content_ignores_container_and_packet_framing(tmp_path, monkeypatch):
    first = tmp_path / "first.opus"
    second = tmp_path / "second.mka"
    first.write_bytes(b"container-one")
    second.write_bytes(b"different-container")
    commands = []

    def start(command, **kwargs):
        commands.append(command)
        return FakeProcess(PCM)

    monkeypatch.setattr("podtran.audio.subprocess.Popen", start)
    service = FingerprintService(tmp_path / "indexes")
    assert service.hash_audio(first) == service.hash_audio(second)
    assert "-copyts" not in commands[0]
    assert "pcm_s16le" in commands[0]
    assert "-ar" not in commands[0] and "-ac" not in commands[0]
    assert service.hash_file(first) != service.hash_file(second)


def test_sample_and_silence_changes_invalidate_fingerprint(tmp_path, decoder):
    original = tmp_path / "original.flac"
    changed = tmp_path / "changed.flac"
    original.write_bytes(PCM)
    service = FingerprintService(tmp_path / "indexes")
    baseline = service.hash_audio(original)
    for content in [PCM[:-2], PCM + b"\x00\x00", b"\x00\x00" + PCM[2:]]:
        changed.write_bytes(content)
        assert service.hash_audio(changed) != baseline


@pytest.mark.parametrize(
    "format",
    [
        AudioFormat(sample_rate=48000, channels=1),
        AudioFormat(sample_rate=24000, channels=2),
    ],
)
def test_sample_format_participates_in_identity(tmp_path, decoder, monkeypatch, format):
    audio = tmp_path / "input.flac"
    audio.write_bytes(PCM)
    before = FingerprintService(tmp_path / "before").hash_audio(audio)
    monkeypatch.setattr("podtran.audio.probe_audio_format", lambda path: format)
    assert FingerprintService(tmp_path / "after").hash_audio(audio) != before


def test_index_ignores_legacy_entries_and_reuses_only_unchanged_files(
    tmp_path, decoder
):
    audio = tmp_path / "audio.opus"
    audio.write_bytes(PCM)
    service = FingerprintService(tmp_path / "indexes")
    stat = audio.stat()
    service.index_path.write_text(
        json.dumps(
            {f"{audio.resolve()}|{stat.st_size}|{stat.st_mtime_ns}": "legacy-file-hash"}
        )
    )
    first = service.hash_audio(audio)
    assert first != "legacy-file-hash"
    assert service.hash_audio(audio) == first
    assert len(decoder[0]) == 1
    audio.write_bytes(PCM + b"changed")
    assert service.hash_audio(audio) != first
    assert len(decoder[0]) == 2


def test_empty_decode_is_not_cached(tmp_path, decoder):
    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"")
    service = FingerprintService(tmp_path / "indexes")
    with pytest.raises(RuntimeError, match="audio"):
        service.hash_audio(audio)
    assert not service.index_path.exists()


def test_decoder_failure_does_not_publish_partial_fingerprint(tmp_path, monkeypatch):
    audio = tmp_path / "broken.opus"
    audio.write_bytes(b"broken")
    process = FakeProcess(PCM)
    process.returncode = 1

    def start(*args, **kwargs):
        kwargs["stderr"].write(b"corrupt audio packet")
        return process

    monkeypatch.setattr("podtran.audio.subprocess.Popen", start)
    service = FingerprintService(tmp_path / "indexes")
    with pytest.raises(RuntimeError, match="corrupt audio packet"):
        service.hash_audio(audio)
    assert not service.index_path.exists()
    assert process.stdout.closed


def test_interruption_reaps_decoder_without_publishing_index(tmp_path, monkeypatch):
    audio = tmp_path / "audio.opus"
    audio.write_bytes(b"audio")
    process = FakeProcess(PCM)

    class InterruptedOutput(io.BytesIO):
        def read(self, size=-1):
            raise KeyboardInterrupt

    process.stdout = InterruptedOutput()
    monkeypatch.setattr("podtran.audio.subprocess.Popen", lambda *a, **kw: process)
    service = FingerprintService(tmp_path / "indexes")
    with pytest.raises(KeyboardInterrupt):
        service.hash_audio(audio)
    assert process.killed
    assert process.stdout.closed
    assert not service.index_path.exists()


def test_file_changed_during_decode_does_not_publish_index(tmp_path, monkeypatch):
    audio = tmp_path / "audio.opus"
    audio.write_bytes(b"before")

    def changing_decode(path):
        path.write_bytes(b"after decoding started")
        return "unreliable"

    monkeypatch.setattr("podtran.fingerprints.hash_audio_content", changing_decode)
    service = FingerprintService(tmp_path / "indexes")
    with pytest.raises(RuntimeError, match="Audio changed"):
        service.hash_audio(audio)
    assert not service.index_path.exists()
