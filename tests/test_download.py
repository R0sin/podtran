from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from podtran.download import download_audio


@pytest.mark.parametrize("sponsorblock", [False, True])
@pytest.mark.parametrize("preparation_fails", [False, True])
def test_download_uses_postprocessed_audio_and_single_episode_options(
    tmp_path, monkeypatch, sponsorblock, preparation_fails
):
    audio = tmp_path / "episode.opus"
    audio.write_bytes(b"audio")

    class Downloader:
        def __init__(self, options):
            assert options["noplaylist"] is True
            assert options["extract_flat"] == "in_playlist"
            assert options["format"] == "bestaudio/best"
            processors = options["postprocessors"]
            if sponsorblock:
                assert processors == [
                    {
                        "key": "SponsorBlock",
                        "categories": ["sponsor"],
                        "when": "after_filter",
                    },
                    {"key": "FFmpegExtractAudio", "preferredcodec": "best"},
                ]
            else:
                assert processors == [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "best"}
                ]
            assert not options.get("ignoreerrors")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            assert not download
            return {"id": "episode"}

        def process_ie_result(self, info, download):
            assert download
            return {
                "requested_downloads": [{"filepath": str(audio)}],
                "sponsorblock_chapters": [
                    {"category": "sponsor", "start_time": 1.25, "end_time": 2.75}
                ],
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=Downloader))

    def prepare(source, output, *, cuts):
        assert source == audio
        assert [(cut.start_time, cut.end_time) for cut in cuts] == (
            [(1.25, 2.75)] if sponsorblock else []
        )
        if preparation_fails:
            raise RuntimeError("preparation failed")
        output.parent.mkdir(parents=True)
        output.write_bytes(b"prepared")
        return output

    monkeypatch.setattr("podtran.download.prepare_audio", prepare)
    if preparation_fails:
        with pytest.raises(RuntimeError, match="preparation failed"):
            download_audio("https://example.com/episode", tmp_path, sponsorblock)
        assert audio.read_bytes() == b"audio"
        return
    result = download_audio("https://example.com/episode", tmp_path, sponsorblock)
    assert result.processing_audio == tmp_path / "processed" / "episode.flac"
    assert result.processing_audio.read_bytes() == b"prepared"
    assert result.source_audio_name == "episode.opus"
    assert result.source_audio_sha256 == hashlib.sha256(b"audio").hexdigest()
    assert not audio.exists()


@pytest.mark.parametrize(
    "info", [None, {"_type": "playlist", "entries": []}, {"is_live": True}]
)
def test_rejects_lists_and_live_before_downloading(tmp_path, monkeypatch, info):
    class Downloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            return info

        def process_ie_result(self, *args, **kwargs):
            pytest.fail("Must not download a playlist or live stream")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=Downloader))
    with pytest.raises(ValueError):
        download_audio("https://example.com/list", tmp_path)


def test_sponsor_cuts_change_processing_identity_but_not_source_provenance(
    tmp_path, monkeypatch, fake_audio_decoder
):
    from pathlib import Path

    from podtran.fingerprints import FingerprintService

    class Downloader:
        def __init__(self, options):
            self.directory = Path(options["outtmpl"]).parent

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            return {"id": "episode"}

        def process_ie_result(self, *args, **kwargs):
            audio = self.directory / "episode.opus"
            audio.write_bytes(b"speech sponsor speech")
            return {
                "requested_downloads": [{"filepath": str(audio)}],
                "sponsorblock_chapters": [
                    {"category": "sponsor", "start_time": 1, "end_time": 2}
                ],
            }

    def prepare(source, output, *, cuts):
        content = source.read_bytes()
        output.parent.mkdir(parents=True)
        output.write_bytes(content.replace(b" sponsor", b"") if cuts else content)
        return output

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=Downloader))
    monkeypatch.setattr("podtran.download.prepare_audio", prepare)
    full = download_audio("https://example.com/episode", tmp_path / "full", False)
    cut = download_audio("https://example.com/episode", tmp_path / "cut", True)
    assert full.source_audio_sha256 == cut.source_audio_sha256
    assert (
        cut.source_audio_sha256 == hashlib.sha256(b"speech sponsor speech").hexdigest()
    )
    fingerprints = FingerprintService(tmp_path / "indexes")
    assert fingerprints.hash_audio(full.processing_audio) != fingerprints.hash_audio(
        cut.processing_audio
    )
    assert not (tmp_path / "full/episode.opus").exists()
    assert not (tmp_path / "cut/episode.opus").exists()
