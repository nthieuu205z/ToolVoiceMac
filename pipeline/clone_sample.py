"""Choose a quiet reference boundary instead of truncating active speech."""
from __future__ import annotations

import numpy as np


def reference_end(samples: np.ndarray, sample_rate: int, *, max_seconds: float = 12.0) -> int:
    """Return an exclusive PCM16 endpoint; refuse an unsafe long recording.

    This is a conservative energy heuristic, not speech recognition. Short
    uploaded clips are preserved. At the size limit, require a quiet tail or
    a >=150 ms pause after three seconds, leaving up to 150 ms of that pause.
    The transcript must be generated from the resulting file, never the source.
    """
    limit = int(max_seconds * sample_rate)
    if len(samples) < limit:
        return len(samples)
    audio = samples[:limit].astype(np.float32) / 32768.0
    frame = max(1, round(sample_rate * .01))
    frames = audio[:len(audio) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(.001, min(.01, float(np.sqrt(np.mean(audio * audio))) * .1))
    quiet = rms <= threshold
    if quiet[-10:].all():
        return limit
    candidates = []
    start = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = index
        if start is not None and (not is_quiet or index == len(quiet) - 1):
            end = index if not is_quiet else index + 1
            if start * frame >= 3 * sample_rate and end - start >= 15:
                candidates.append(start + min(15, (end - start) // 2))
            start = None
    if not candidates:
        raise ValueError("Mẫu quá dài và không có khoảng nghỉ an toàn trước 12 giây. Hãy chọn đoạn 3–10 giây, kết thúc trọn câu hoặc cụm từ.")
    return candidates[-1] * frame
