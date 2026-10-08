from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from podtran.download import download_audio


def test_download_uses_postprocessed_audio_and_single_episode_options(
    tmp_path, monkeypatch
):
    audio = tmp_path / "episode.opus"
    audio.write_bytes(b"audio")

    class Downloader:
        def __init__(self, options):
            assert options["noplaylist"] is True
            assert options["extract_flat"] == "in_playlist"
            assert options["format"] == "bestaudio/best"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            assert not download
            return {"id": "episode"}

        def process_ie_result(self, info, download):
            assert download
            return {"requested_downloads": [{"filepath": str(audio)}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=Downloader))
    assert download_audio("https://example.com/episode", tmp_path) == audio


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
