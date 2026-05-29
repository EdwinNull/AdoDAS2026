#!/usr/bin/env python3
"""
Extract per-session linguistic features from clean_transcript.txt and segments.csv.

Features (12-dim):
  0. word_count          — total jieba-segmented words
  1. unique_ratio        — unique / total words
  2. mean_word_len       — avg chars per word
  3. segment_count       — number of speech segments
  4. total_duration_sec  — total speech duration
  5. speech_rate         — chars per second
  6. first_person_ratio  — 我/我的/我自己 / total
  7. negation_ratio      — 不/没/没有/无/别 / total
  8. cognitive_ratio     — 觉得/想/知道/可能/因为/应该 / total
  9. neg_emotion_ratio   — 难过/伤心/累/烦/怕/痛苦 / total
  10. pos_emotion_ratio  — 开心/好/喜欢/快乐/高兴/爱 / total
  11. filler_ratio       — 嗯/呃/那个/就是/然后 / total

Output: <output_dir>/<split>/<pid>/<session>/linguistic.npy  (12, float32)
        <output_dir>/<split>/<pid>/linguistic_participant.npy  (12, float32, mean over sessions)

Usage:
  python scripts/extract_linguistic_features.py --split train
  python scripts/extract_linguistic_features.py --split val
  python scripts/extract_linguistic_features.py --split test_hidden
  python scripts/extract_linguistic_features.py --split all   # all three
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_ling")

# ── word category dictionaries ──────────────────────────────────────────
# Multi-char entries only to avoid ambiguous single-char matches
FIRST_PERSON = {"我", "我的", "我自己", "我们", "我们的"}
NEGATIONS = {"不", "没", "没有", "不是", "不会", "不能"}
COGNITIVE = {"觉得", "知道", "可能", "因为", "应该", "如果", "所以", "但是", "认为", "也许", "或许"}
NEG_EMOTION = {"难过", "伤心", "害怕", "痛苦", "紧张", "担心", "生气", "讨厌", "孤独", "失望", "烦躁"}
POS_EMOTION = {"开心", "快乐", "高兴", "幸福", "美好", "有趣", "喜欢", "很棒", "不错", "挺好"}
FILLERS = {"嗯", "呃", "那个", "就是", "然后", "然后呢"}

# ── feature dimension ───────────────────────────────────────────────────
N_FEATURES = 12


def _safe_read_txt(path: Path) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def extract_session_features(
    transcript_dir: Path, pid: str, session: str
) -> np.ndarray | None:
    """Extract 12-dim linguistic feature vector for one session.
    Returns None if transcript is missing or empty.
    """
    sess_dir = transcript_dir / pid / session
    txt_path = sess_dir / "clean_transcript.txt"
    seg_path = sess_dir / "segments.csv"

    text = _safe_read_txt(txt_path)
    if text is None or len(text) == 0:
        return None

    import jieba
    words = list(jieba.cut(text))
    words = [w.strip() for w in words if len(w.strip()) > 0]
    n_words = len(words)
    n_chars = len(text.replace(" ", ""))
    n_unique = len(set(words))

    # Word-level features
    unique_ratio = n_unique / max(n_words, 1)
    mean_word_len = n_chars / max(n_words, 1)

    # Segment-level features (from segments.csv)
    n_segments = 0
    total_dur = 0.0
    if seg_path.exists():
        try:
            seg = pd.read_csv(seg_path)
            n_segments = len(seg)
            total_dur = float(seg["duration"].sum())
        except Exception:
            pass

    speech_rate = n_chars / max(total_dur, 0.1)

    # Category ratios
    def _cat_ratio(cat_set: set[str]) -> float:
        cnt = sum(1 for w in words if w in cat_set)
        return cnt / max(n_words, 1)

    features = np.array([
        float(n_words),
        unique_ratio,
        mean_word_len,
        float(n_segments),
        total_dur,
        speech_rate,
        _cat_ratio(FIRST_PERSON),
        _cat_ratio(NEGATIONS),
        _cat_ratio(COGNITIVE),
        _cat_ratio(NEG_EMOTION),
        _cat_ratio(POS_EMOTION),
        _cat_ratio(FILLERS),
    ], dtype=np.float32)

    return features


def process_split(
    transcript_root: Path,
    output_root: Path,
    split: str,
) -> None:
    """Process all participants in a split."""
    transcript_dir = transcript_root / split

    # Discover participants with sessions
    pid_sessions: dict[str, set[str]] = {}
    for sch in sorted(transcript_dir.iterdir()):
        if not sch.is_dir():
            continue
        for cls in sorted(sch.iterdir()):
            if not cls.is_dir():
                continue
            for pid_dir in sorted(cls.iterdir()):
                if not pid_dir.is_dir():
                    continue
                pid = pid_dir.name
                sessions = set()
                for sess in sorted(pid_dir.iterdir()):
                    if sess.is_dir() and sess in {"A01", "B01", "B02", "B03"}:
                        txt = pid_dir / sess / "clean_transcript.txt"
                        if txt.exists():
                            sessions.add(sess)
                if sessions:
                    pid_sessions[pid] = sessions

    log.info(f"Processing {split}: {len(pid_sessions)} participants")

    errors = 0
    for pid, sessions in tqdm(pid_sessions.items(), desc=f"Extract {split}", dynamic_ncols=True):
        # Compute path using same structure as feature data
        # Find the participant's school/class from transcript tree
        pid_dir = None
        for root, dirs, _ in os.walk(str(transcript_dir)):
            if os.path.basename(root) == pid:
                pid_dir = Path(root)
                break

        if pid_dir is None:
            errors += 1
            continue

        rel_parts = pid_dir.relative_to(transcript_dir).parts
        school, cls = rel_parts[0], rel_parts[1]

        session_feats = {}
        for sess in sessions:
            feats = extract_session_features(transcript_dir, f"{school}/{cls}/{pid}", sess)
            if feats is not None:
                session_feats[sess] = feats
                # Save per-session
                out_dir = output_root / split / school / cls / pid / sess
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(out_dir / "linguistic.npy"), feats)

        # Pool to participant level (mean over available sessions)
        if session_feats:
            participant_feat = np.mean(list(session_feats.values()), axis=0)
            out_dir = output_root / split / school / cls / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / "linguistic_participant.npy"), participant_feat.astype(np.float32))

    if errors > 0:
        log.warning(f"{split}: {errors} participants had errors")


def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features from transcripts")
    parser.add_argument("--transcript-root", default="/data1/AdoDas",
                        help="Root dir containing {train,val,test}_transcript/")
    parser.add_argument("--output-root", default="/data1/AdoDas/linguistic_features",
                        help="Output root for .npy files")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test_hidden", "all"])
    args = parser.parse_args()

    transcript_root = Path(args.transcript_root)
    output_root = Path(args.output_root)

    split_map = {
        "train": "train_transcript",
        "val": "val_transcript",
        "test_hidden": "test_transcript",
    }

    splits = list(split_map.keys()) if args.split == "all" else [args.split]

    for split in splits:
        tdir = transcript_root / split_map[split]
        if not tdir.exists():
            log.warning(f"Transcript dir not found: {tdir}, skipping {split}")
            continue
        process_split(tdir, output_root, split)

    log.info("Done.")


if __name__ == "__main__":
    main()
