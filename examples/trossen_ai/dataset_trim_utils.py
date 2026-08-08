#!/usr/bin/env python
"""
dataset_trim_utils.py
=====================

Headless "black box" used by record_gr00t_trossen.py to turn a full
intervention episode into a trimmed clip = [first_teleop_start - tpad, first_teleop_end],
then hand the trimmed episode back to LeRobot's own machinery for appending
into the growing multi-episode dataset.

Design (Option 2 — maximally robust, minimal record_ changes):

    1. record_ records ONE full intervention episode into a TEMP dataset using
       LeRobot's normal save_episode (pure intervention behavior).
    2. trim_and_readd_episode():
         a. reads that temp episode's parquet,
         b. slices rows to [start, end], renumbers frame_index/timestamp/index,
         c. trims each camera video with ffmpeg -ss/-t,
         d. writes a clean single-episode TEMP dataset (the "trimmed temp"),
         e. loads the trimmed temp as a LeRobotDataset,
         f. re-adds its frames into the REAL target dataset via add_frame +
            save_episode (LeRobot's proven append path),
         g. deletes both temps.

    Everything risky is confined to this file. If the trim is wrong, only a
    throwaway temp is affected; the real dataset only ever receives episodes
    through LeRobot's own save_episode.

The parquet/video/stat conventions here are copied from the user's
dataset_splitter.py so the intermediate trimmed dataset is a valid LeRobot v2.1
dataset that LeRobotDataset can load.

Requirements: pandas, pyarrow, numpy, ffmpeg on PATH, lerobot.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

log = logging.getLogger("dataset_trim_utils")


# ---------------------------------------------------------------------------
# Small JSON helpers (match dataset_splitter conventions)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    entries.append(json.loads(s))
    return entries


def _save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Path helpers (LeRobot v2.1 layout)
# ---------------------------------------------------------------------------

def _chunk(ep: int, chunks_size: int) -> int:
    return ep // chunks_size


def _parquet_path(root: Path, ep: int, chunks_size: int) -> Path:
    return root / "data" / f"chunk-{_chunk(ep, chunks_size):03d}" / f"episode_{ep:06d}.parquet"


def _video_path(root: Path, ep: int, key: str, chunks_size: int) -> Path:
    return root / "videos" / f"chunk-{_chunk(ep, chunks_size):03d}" / key / f"episode_{ep:06d}.mp4"


# ---------------------------------------------------------------------------
# Stats (LeRobot v2.1 episodes_stats format) — copied from dataset_splitter
# ---------------------------------------------------------------------------

def _compute_stats(df: "pd.DataFrame") -> dict:
    out: dict = {}
    for col in df.columns:
        try:
            vals = df[col].values
            if hasattr(vals[0], "__len__"):
                s = np.stack(vals)
                if s.ndim == 2:
                    out[col] = dict(
                        min=np.min(s, axis=0).tolist(),
                        max=np.max(s, axis=0).tolist(),
                        mean=np.mean(s, axis=0).tolist(),
                        std=np.std(s, axis=0).tolist(),
                        count=[len(s)],
                    )
                elif s.ndim >= 3:
                    out[col] = dict(
                        min=np.min(s, axis=0).tolist(),
                        max=np.max(s, axis=0).tolist(),
                        mean=np.mean(s, axis=0).tolist(),
                        std=np.std(s, axis=0).tolist(),
                        count=[s.shape[0]],
                    )
                else:
                    out[col] = dict(
                        min=[float(np.min(s))], max=[float(np.max(s))],
                        mean=[float(np.mean(s))], std=[float(np.std(s))],
                        count=[len(s)],
                    )
            elif np.issubdtype(type(vals[0]), np.number):
                out[col] = dict(
                    min=[float(np.min(vals))], max=[float(np.max(vals))],
                    mean=[float(np.mean(vals))], std=[float(np.std(vals))],
                    count=[len(vals)],
                )
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Core trim: write a clean single-episode dataset from a slice of episode 0
# of the source (temp) dataset.
# ---------------------------------------------------------------------------

def _trim_to_new_dataset(src_root: Path, dst_root: Path,
                         start: int, end: int,
                         ffmpeg_path: str = "/home/trossen-ai/miniconda3/envs/lerobot/bin/ffmpeg") -> int:
    """Read episode 0 of src_root, keep frames [start, end), write a fresh
    single-episode dataset at dst_root. Returns the trimmed length.
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas + pyarrow required (pip install pandas pyarrow)")

    meta = src_root / "meta"
    info = _load_json(meta / "info.json")
    fps = info.get("fps", 30)
    chunks_size = info.get("chunks_size", 1000)
    video_keys = [k for k, v in info.get("features", {}).items()
                  if v.get("dtype") == "video"]
    src_eps = {e["episode_index"]: e for e in _load_jsonl(meta / "episodes.jsonl")}
    src_tasks = _load_jsonl(meta / "tasks.jsonl")

    if 0 not in src_eps:
        raise ValueError("Temp dataset has no episode 0 to trim.")
    ep_len = src_eps[0]["length"]
    ep_tasks = src_eps[0].get("tasks", ["unknown"])

    # Clamp the requested window to the real episode bounds.
    start = max(0, int(start))
    end = min(int(end), ep_len)
    if end <= start:
        raise ValueError(f"Empty trim window after clamping: [{start}, {end}) "
                         f"for episode length {ep_len}")

    # ── parquet slice + renumber ──
    src_pq = _parquet_path(src_root, 0, chunks_size)
    dst_pq = _parquet_path(dst_root, 0, chunks_size)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src_pq).iloc[start:end].copy()
    df["episode_index"] = 0
    df["frame_index"] = np.arange(len(df))
    df["index"] = np.arange(len(df))
    df["timestamp"] = df["frame_index"] / fps
    # task_index left as-is (single task carried through); tasks.jsonl copied below
    df.to_parquet(dst_pq, index=False)
    trimmed_len = len(df)

    # ── video trim (ffmpeg -ss/-t), copied convention from dataset_splitter ──
    t0 = start / fps
    dur = trimmed_len / fps
    for vk in video_keys:
        src_vid = _video_path(src_root, 0, vk, chunks_size)
        if not src_vid.exists():
            continue
        dst_vid = _video_path(dst_root, 0, vk, chunks_size)
        dst_vid.parent.mkdir(parents=True, exist_ok=True)
        _r=subprocess.run([
            ffmpeg_path, "-y",
            "-i", str(src_vid),
            "-ss", f"{t0:.6f}",
            "-t", f"{dur:.6f}",
            "-vf", "setpts=PTS-STARTPTS",
            "-c:v", "libsvtav1",
            "-pix_fmt", "yuv420p",
            "-g", "2",
            "-crf", "30",
            "-svtav1-params", "fast-decode=0",
            "-r", str(fps),
            "-frames:v", str(trimmed_len),
            "-an",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(dst_vid),
        ], capture_output=True, check=False, text=True) #capture_output=True, check=True)
        if _r.returncode != 0:
            log.error("ffmpeg failed:\n" + _r.stderr[-2000:])
            raise RuntimeError("ffmpeg trim failed; see stderr above")
        
    # ── metadata for a valid single-episode dataset ──
    out_meta = dst_root / "meta"
    new_info = info.copy()
    new_info.update(
        total_episodes=1,
        total_frames=trimmed_len,
        total_videos=len(video_keys),
        total_chunks=1,
        splits={"train": "0:1"},
    )
    _save_json(new_info, out_meta / "info.json")
    _save_jsonl(src_tasks, out_meta / "tasks.jsonl")
    _save_jsonl([{"episode_index": 0, "tasks": ep_tasks, "length": trimmed_len}],
                out_meta / "episodes.jsonl")
    stats = _compute_stats(df)
    _save_jsonl([{"episode_index": 0, "stats": stats}],
                out_meta / "episodes_stats.jsonl")

    return trimmed_len


# ---------------------------------------------------------------------------
# Public entry point called by record_gr00t_trossen.py
# ---------------------------------------------------------------------------

def trim_and_add_episode(temp_dataset_root,
                         target_dataset,
                         teleop_start: int,
                         teleop_end: int,
                         tpad_s: float,
                         task_prompt: str,
                         ffmpeg_path: str = "/home/trossen-ai/miniconda3/envs/lerobot/bin/ffmpeg"):
    """Trim the single full episode sitting in `temp_dataset_root` down to
    [teleop_start - tpad, teleop_end] and ADD its frames into `target_dataset`
    via LeRobot's own add_frame. Does NOT call save_episode — the caller (the
    record loop) does that, so the save stays visible in the main loop and the
    normal/clip paths share one save call.

    Returns `target_dataset` (so the caller can write
    `dataset = trim_and_add_episode(...)` then `dataset.save_episode()`).
    Cleans up its own intermediate temp.

    Args:
        temp_dataset_root: path to the temp dataset holding ONE full episode.
        target_dataset: the real, growing multi-episode LeRobotDataset.
        teleop_start, teleop_end: frame indices of the FIRST teleop segment.
        tpad_s: seconds of pre-teleop rollout to keep as lead-in context.
        task_prompt: task string for the re-added frames.
        ffmpeg_path: ffmpeg binary.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    temp_dataset_root = Path(temp_dataset_root).resolve()
    meta = temp_dataset_root / "meta"
    info = _load_json(meta / "info.json")
    fps = info.get("fps", 30)

    pad = int(round(tpad_s * fps))
    start = teleop_start - pad
    end = teleop_end
    log.info(f"trim: teleop=[{teleop_start},{teleop_end}] pad={pad} "
             f"-> window=[{start},{end}]")

    trimmed_root = Path(tempfile.mkdtemp(prefix="clip_trim_"))
    try:
        # 1) produce a clean single-episode trimmed dataset on disk
        n = _trim_to_new_dataset(temp_dataset_root, trimmed_root, start, end,
                                 ffmpeg_path=ffmpeg_path)
        log.info(f"trim: wrote {n} trimmed frames to {trimmed_root}")

        # 2) load it back and ADD its frames into the target (no save here)
        trimmed_ds = LeRobotDataset(info.get("repo_id", "clip_trim"),
                                    root=str(trimmed_root))
        n_frames = trimmed_ds.num_frames if hasattr(trimmed_ds, "num_frames") else n
        added = 0
        for i in range(n_frames):
            item = trimmed_ds[i]
            frame = {}
            for k, v in item.items():
                if k in ("episode_index", "frame_index", "timestamp", "index",
                         "task_index", "task"):
                    continue
                # Images come back channels-first (C,H,W) from the dataset;
                # add_frame expects channels-last (H,W,C). Permute image tensors.
                if k.startswith("observation.images.") and hasattr(v, "ndim") and v.ndim == 3:
                    if v.shape[0] == 3:              # (C, H, W) -> (H, W, C)
                        v = v.permute(1, 2, 0) if hasattr(v, "permute") else v.transpose(1, 2, 0)
                frame[k] = v
            frame["task"] = task_prompt
            target_dataset.add_frame(frame)
            added += 1
        log.info(f"trim: added {added} frames into target dataset (caller saves)")
        return target_dataset
    finally:
        shutil.rmtree(trimmed_root, ignore_errors=True)
