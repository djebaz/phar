/env python3
"""
Multimodal HAR demo (RGB / Skeleton / Audio) — cleaned & refactored
-------------------------------------------------------------------
Key improvements vs original:
- Clear structure (Pipeline class), no mutable global state
- Safer tempfile handling & cleanup, robust ffmpeg audio muxing
- Better logging with Rich, consistent progress bars (tqdm)
- Parametrized thresholds (incl. --min-correct-rate)
- Defensive checks (no audio/video, empty detections, etc.)
- Type hints, docstrings, smaller helpers, early returns
- Deterministic label handling & weighted fusion

NOTE: Keeps dependency on mmaction2 / mmdet / mmpose APIs used by the project.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2
import torch
import yaml
import soundfile as sf
import pyloudnorm as pyln
import moviepy.editor as mpy
from rich.console import Console
from tqdm import tqdm

# project-local imports
sys.path.append('./mmaction2')
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.core.evaluation import get_weighted_score
from demo.demo_skeleton import frame_extraction

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError("mmdet is required: install mmdetection and its deps") from e

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError("mmpose is required: install mmpose and its deps") from e

# ---------------------------- UI constants
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.85
FONTCOLOR = (255, 255, 0)           # BGR
FONTCOLOR_SCORE = (0, 165, 255)
THICKNESS = 1
LINETYPE = 1

# ---------------------------- Defaults
AUDIO_FEATURE_SCRIPT = 'mmaction2/tools/data/build_audio_features.py'
TEMP_DIR = 'temp'
NUM_KEYPOINTS = 17  # COCO-17

# placeholder scores when a modality is skipped
PLACEHOLDER = {
    'kissing': 0.0,
    'fondling': 0.0,
    'handjob': 0.0,
    'fingering': 0.0,
    'titjob': 0.0,
}

console = Console()


# ============================ Small utilities

def verbose(console_on: bool):
    return console.print if console_on else (lambda *a, **k: None)


def prepare_audio_input(x: Any) -> np.ndarray:
    """Prepare audio features for inference_recognizer.
    Supports single .npy, directory of .npy files, list/tuple of arrays/paths, or ndarray.
    Returns float32 ndarray shaped [N, T, F] or compatible with model.
    """
    if isinstance(x, str) and osp.isdir(x):
        npy_files = sorted([f for f in os.listdir(x) if f.endswith('.npy')])
        if not npy_files:
            raise FileNotFoundError(f"No .npy found in {x}")
        arr = np.stack([np.load(osp.join(x, f)) for f in npy_files], axis=0)
    elif isinstance(x, str):
        arr = np.load(x)
    elif isinstance(x, (list, tuple)):
        loaded = []
        for xi in x:
            loaded.append(np.load(xi) if isinstance(xi, str) else np.asarray(xi))
        arr = np.stack(loaded, axis=0)
    else:
        arr = np.asarray(x)
    return arr.astype(np.float32, copy=False)


def run_ffmpeg(args: List[str], quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=quiet, check=False)


# ============================ Data classes
