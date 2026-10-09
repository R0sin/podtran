from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit


def is_http_url(value: str) -> bool:
    return urlsplit(value).scheme.lower() in {"http", "https"}


def download_audio(url: str, directory: Path, sponsorblock: bool = False) -> Path:
    from yt_dlp import YoutubeDL

    directory.mkdir(parents=True, exist_ok=True)
    postprocessors = [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}]
    if sponsorblock:
        postprocessors.insert(
            0,
            {"key": "SponsorBlock", "categories": ["sponsor"], "when": "after_filter"},
        )
        postprocessors.append(
            {"key": "ModifyChapters", "remove_sponsor_segments": ["sponsor"]}
        )
    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "extract_flat": "in_playlist",
        "lazy_playlist": True,
        "outtmpl": str(directory / "%(title).100B [%(id)s].%(ext)s"),
        "restrictfilenames": True,
        "postprocessors": postprocessors,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        if not info or info.get("_type", "video") != "video" or "entries" in info:
            raise ValueError(
                "Please provide a single episode URL, not a playlist or channel."
            )
        if info.get("is_live"):
            raise ValueError(
                "Live streams are not supported. Use a completed episode URL."
            )
        result = downloader.process_ie_result(info, download=True)
        audio = Path(result["requested_downloads"][0]["filepath"])
        if not audio.is_file():
            raise FileNotFoundError(f"Downloaded audio not found: {audio}")
        return audio
