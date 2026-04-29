"""
transcribe.py — Whisper transcription for M4A audio (Bahasa Indonesia)
Usage: python transcribe.py <audio_file.m4a>
"""

import whisper
import sys
import os
import time
import json
from pathlib import Path


def transcribe(audio_path: str):
    path = Path(audio_path)

    if not path.exists():
        print(f"[ERROR] File not found: {audio_path}")
        sys.exit(1)

    # ── Model selection ─────────────────────────────────────────────────────
    # For Bahasa Indonesia accuracy, "medium" is the sweet spot.
    # Options: tiny | base | small | medium | large | large-v2 | large-v3
    #   tiny/base  → fast, less accurate (OK for simple speech)
    #   small      → decent, decent accuracy
    #   medium     → recommended for Bahasa Indonesia
    #   large-v3   → best accuracy, needs ~10GB VRAM or is slower on CPU
    MODEL_SIZE = "medium"

    print(f"\n[1/4] Loading Whisper model: {MODEL_SIZE}")
    model = whisper.load_model(MODEL_SIZE)

    print(f"[2/4] Processing file: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    start = time.time()
    print(f"[3/4] Transcribing... (this may take a few minutes for a 45-min file)")

    result = model.transcribe(
        audio_path,
        language="id",          # "id" = Bahasa Indonesia
        task="transcribe",      # use "translate" to get English output instead
        verbose=False,          # set True to stream segments in real-time
        fp16=False,             # set True if you have a CUDA GPU
        condition_on_previous_text=True,  # helps with long audio coherence
        # Optional: uncomment for better timestamp accuracy on long audio
        # word_timestamps=True,
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

    # SRT subtitle file (optional — useful for video)
    srt_file = out_dir / f"{stem}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt(seg['start'])} --> {format_srt(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")
    print(f"✓ SRT file saved: {srt_file}")

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

    # Move original file to the folder
    new_audio_path = out_dir / path.name
    if path.resolve() != new_audio_path.resolve():
        path.rename(new_audio_path)
        print(f"✓ Original audio moved to: {new_audio_path}")

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
        print("Usage: python transcribe.py <audio_file.m4a>")
        print("Example: python transcribe.py rekaman.m4a")
        sys.exit(1)

    transcribe(sys.argv[1])