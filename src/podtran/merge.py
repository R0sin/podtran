from __future__ import annotations

from podtran.models import SegmentRecord, TranscriptSegment, WordAlignment


def merge_transcript_segments(
    transcript: list[TranscriptSegment],
    pause_threshold: float,
    max_block_duration: float,
    configured_voice_map: dict[str, str],
    fallback_voices: list[str],
) -> list[SegmentRecord]:
    voice_map = dict(configured_voice_map)
    next_voice_index = 0

    def resolve_voice(speaker: str) -> str:
        nonlocal next_voice_index
        if speaker in voice_map:
            return voice_map[speaker]
        if not fallback_voices:
            return "default"
        voice = fallback_voices[next_voice_index % len(fallback_voices)]
        voice_map[speaker] = voice
        next_voice_index += 1
        return voice

    blocks: list[SegmentRecord] = []
    current: list[TranscriptSegment] = []
    block_start = 0.0

    for segment in sorted(transcript, key=lambda item: item.start):
        speaker = segment.speaker or "UNKNOWN"
        if not current:
            current = [segment]
            block_start = segment.start
            continue

        last_segment = current[-1]
        last_speaker = last_segment.speaker or "UNKNOWN"
        gap = segment.start - last_segment.end
        duration = segment.end - block_start
        if speaker != last_speaker or gap > pause_threshold or duration > max_block_duration:
            blocks.append(build_block(current, resolve_voice(current[0].speaker or "UNKNOWN"), len(blocks)))
            current = [segment]
            block_start = segment.start
        else:
            current.append(segment)

    if current:
        blocks.append(build_block(current, resolve_voice(current[0].speaker or "UNKNOWN"), len(blocks)))

    return blocks


def build_block(entries: list[TranscriptSegment], voice: str, index: int) -> SegmentRecord:
    words: list[WordAlignment] = []
    for entry in entries:
        words.extend(entry.words)
    text = " ".join(item.text.strip() for item in entries).strip()
    speaker = entries[0].speaker or "UNKNOWN"
    return SegmentRecord(
        segment_id=f"seg_{index:05d}",
        block_id=f"block_{index:05d}",
        start=entries[0].start,
        end=entries[-1].end,
        text=text,
        speaker=speaker,
        voice=voice,
        words=words,
    )
