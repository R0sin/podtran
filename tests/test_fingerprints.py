from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from podtran.fingerprints import FingerprintService
from podtran.audio import _hash_audio_frames


FRAMEHASH = b"""#format: frame checksums
#version: 2
#hash: SHA256
#software: Lavf61
#tb 0: 1/24000
#media_type 0: audio
#codec_id 0: pcm_f64le
#sample_rate 0: 24000
#channel_layout_name 0: mono
#stream#, dts, pts, duration, size, hash
0, 0, 0, 2400, 19200, abcdef
1, 0, 0, 2400, 19200, abcdef
"""


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


def test_audio_content_ignores_container_bytes_and_encoder_version(tmp_path, decoder):
    first = tmp_path / "first.opus"
    second = tmp_path / "second.mka"
    first.write_bytes(FRAMEHASH)
    second.write_bytes(FRAMEHASH.replace(b"Lavf61", b"Lavf99"))
    service = FingerprintService(tmp_path / "indexes")
    assert service.hash_audio(first) == service.hash_audio(second)
    assert len(decoder[0]) == 2
    command = decoder[0][0]
    assert "-copyts" in command
    assert "pcm_f64le" in command
    assert "asetnsamples=n=4096:p=0" in command[command.index("-filter_complex") + 1]
    assert "-ar" not in command and "-ac" not in command


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"abcdef", b"fedcba"),
        (b"24000", b"48000"),
        (b"mono", b"stereo"),
        (b"0, 0, 0,", b"0, 2400, 2400,"),
        (b"2400, 19200", b"2401, 19208"),
    ],
)
def test_audio_content_changes_invalidate_fingerprint(tmp_path, decoder, old, new):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(FRAMEHASH)
    second.write_bytes(FRAMEHASH.replace(old, new))
    service = FingerprintService(tmp_path / "indexes")
    assert service.hash_audio(first) != service.hash_audio(second)


def test_index_ignores_legacy_entries_and_reuses_only_unchanged_files(
    tmp_path, decoder
):
    audio = tmp_path / "audio.opus"
    audio.write_bytes(FRAMEHASH)
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
    audio.write_bytes(FRAMEHASH.replace(b"abcdef", b"changed"))
    assert service.hash_audio(audio) != first
    assert len(decoder[0]) == 2


def test_empty_decode_is_not_cached(tmp_path, decoder):
    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"")
    service = FingerprintService(tmp_path / "indexes")
    with pytest.raises(RuntimeError, match="audio"):
        service.hash_audio(audio)
    assert not service.index_path.exists()


def test_timing_ignores_packet_boundaries_but_preserves_gaps_and_overlaps():
    split = FRAMEHASH.replace(
        b"0, 0, 0, 2400, 19200, abcdef\n",
        b"0, 0, 0, 1024, 8192, first\n0, 1024, 1024, 1376, 11008, last\n",
    )
    baseline = _hash_audio_frames(FRAMEHASH.splitlines())
    assert _hash_audio_frames(split.splitlines()) == baseline
    for offset in [100, -100]:
        changed = split.replace(
            b"0, 1024, 1024,", f"0, 1024, {1024 + offset},".encode()
        )
        assert _hash_audio_frames(changed.splitlines()) != baseline


def test_decoder_failure_does_not_publish_partial_fingerprint(tmp_path, monkeypatch):
    audio = tmp_path / "broken.opus"
    audio.write_bytes(b"broken")
    process = FakeProcess(FRAMEHASH)
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
    process = FakeProcess(FRAMEHASH)

    class InterruptedOutput(io.BytesIO):
        def __next__(self):
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
