"""
transcribe_video.py — Whisper transcription for MP4 video files (Bahasa Indonesia)
Usage: python transcribe_video.py <video_file.mp4>
"""

import whisper
import sys
import os
import time
import json
import torch
from pathlib import Path


def transcribe(video_path: str):
    path = Path(video_path)

    if not path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    # ── Model selection ─────────────────────────────────────────────────────
    # For Bahasa Indonesia accuracy, "medium" is the sweet spot.
    # Options: tiny | base | small | medium | large | large-v2 | large-v3
    #   tiny/base  → fast, less accurate
    #   small      → decent accuracy
    #   medium     → recommended for Bahasa Indonesia
    #   large-v3   → best accuracy, needs ~10GB VRAM or is slower on CPU
    MODEL_SIZE = "medium"

    print(f"\n[1/4] Loading Whisper model: {MODEL_SIZE}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model(MODEL_SIZE, device=device)

    print(f"[2/4] Processing file: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    start = time.time()
    print(f"[3/4] Transcribing... (this may take a few minutes depending on the video length)")

    # Whisper can ingest .mp4 directly because it uses FFmpeg under the hood to extract the audio!
    result = model.transcribe(
        video_path,
        language="id",          # "id" = Bahasa Indonesia
        task="transcribe",      # use "translate" to get English output instead
        verbose=False,          # set True to stream segments in real-time
        fp16=False,             # set True if you have a CUDA GPU
        condition_on_previous_text=True,  # helps with long coherence
    )

    elapsed = time.time() - start
    print(f"[4/4] Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Output files ────────────────────────────────────────────────────────
    stem = path.stem
    out_dir = path.parent / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plain text transcript
    txt_path = out_dir / f"{stem}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
    print(f"\n✓ Transcript saved: {txt_path}")

    # Timestamped segments (readable)
    srt_path = out_dir / f"{stem}_segments.txt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            start_ts = format_time(seg["start"])
            end_ts   = format_time(seg["end"])
            f.write(f"[{start_ts} → {end_ts}]\n{seg['text'].strip()}\n\n")
    print(f"✓ Timestamped segments saved: {srt_path}")

    # SRT subtitle file (useful for video)
    srt_file = out_dir / f"{stem}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt(seg['start'])} --> {format_srt(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")
    print(f"✓ SRT subtitle file saved: {srt_file}")

    # JSON with full metadata
    json_path = out_dir / f"{stem}_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "language": result.get("language"),
            "model": MODEL_SIZE,
            "duration_seconds": result["segments"][-1]["end"] if result["segments"] else 0,
            "segments": [
                {
                    "id": s["id"],
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip(),
                }
                for s in result["segments"]
            ],
            "text": result["text"].strip(),
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Full JSON saved: {json_path}")

    # Move original video to the folder
    new_video_path = out_dir / path.name
    if path.resolve() != new_video_path.resolve():
        path.rename(new_video_path)
        print(f"✓ Original video moved to: {new_video_path}")

    # Print preview
    preview = result["text"].strip()[:500]
    print(f"\n── Transcript preview ──────────────────────────────────")
    print(preview + ("..." if len(result["text"]) > 500 else ""))
    print(f"────────────────────────────────────────────────────────")
    print(f"\nDetected language: {result.get('language', 'unknown')}")
    print(f"Total segments:    {len(result['segments'])}")


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_video.py <video_file.mp4>")
        print("Example: python transcribe_video.py meeting.mp4")
        sys.exit(1)

    transcribe(sys.argv[1])
