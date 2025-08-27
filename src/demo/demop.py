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

def prepare_audio_input(x):
    """
    Prepare audio features for inference_recognizer.
    Accepts: single .npy, dir of .npy files, list/tuple of arrays/paths, or ndarray.
    Returns float32 array. If a single 2D feature [T,F] is given, expands to [1,T,F].
    """
    if isinstance(x, str) and os.path.isdir(x):
        npy_files = sorted([f for f in os.listdir(x) if f.endswith(".npy")])
        if not npy_files:
            raise FileNotFoundError(f"No .npy found in {x}")
        arr = np.stack([np.load(os.path.join(x, f)) for f in npy_files], axis=0)
    elif isinstance(x, str):
        arr = np.load(x)
    elif isinstance(x, (list, tuple)):
        loaded = []
        for xi in x:
            loaded.append(np.load(xi) if isinstance(xi, str) else np.asarray(xi))
        arr = np.stack(loaded, axis=0)
    else:
        arr = np.asarray(x)

    if arr.ndim == 2:  # [T, F] -> [1, T, F]
        arr = arr[None, ...]
    return arr.astype(np.float32, copy=False)


def run_ffmpeg(args: List[str], quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=quiet, check=False)


# ============================ Data classes

@dataclass
class Args:
    video: str
    out: str
    label_maps: List[str]
    rgb_config: str
    rgb_checkpoint: str
    skeleton_config: str
    skeleton_checkpoint: str | None
    audio_config: str
    audio_checkpoint: str | None
    pose_config: str
    pose_checkpoint: str
    det_config: str
    det_checkpoint: str
    det_score_thr: float
    num_processes: int
    device: str
    subclip_len: int
    short_side: int
    coefficients: List[float]
    pose_score_thr: float
    min_correct_rate: float
    loudness_weights: str
    topk: int
    timestamps: bool
    verbose: bool


# ============================ Pipeline

class Pipeline:
    def __init__(self, args: Args):
        self.args = args
        self.log = verbose(args.verbose)
        self.temp_dir = Path(TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # labels
        self.rgb_labels, self.pose_labels, self.audio_labels = [
            [x.strip() for x in open(p).readlines()] for p in args.label_maps
        ]

        # models (lazy as needed)
        self.rgb_model = init_recognizer(args.rgb_config, args.rgb_checkpoint, device=args.device)
        self.audio_model = None
        self.pose_model = None
        self.det_model = None
        self.sk_model = None

        # loudness gating
        lw = yaml.safe_load(open(args.loudness_weights))
        self.loudness_weight = float(sum(lw.values()) / max(len(lw), 1)) if args.audio_checkpoint else None

        # results per clip
        self.preds: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.used_mods: Dict[str, List[int]] = {}

    # ------------- Video prep & clip extraction
    def _resize_video(self, src: str, height: int = 480) -> str:
        clip = mpy.VideoFileClip(src)
        # fix rotation metadata
        if clip.rotation in (90, 270):
            clip = clip.resize(clip.size[::-1])
            clip.rotation = 0
        resized = clip.resize(height=height)
        out_path = self.temp_dir / osp.basename(src)
        resized.write_videofile(str(out_path), logger=None, audio=True)
        return str(out_path)

    @staticmethod
    def _extract_clip_one(items: Tuple[Dict[str, Tuple[int, int]], str]) -> str | None:
        ts, video = items
        key = next(iter(ts))
        start, finish = next(iter(ts.values()))
        if finish - start < 2:
            return None
        try:
            v = mpy.VideoFileClip(video).subclip(start, finish)
            out = Path(TEMP_DIR) / f"{key}_{start}_{finish}.mp4"
            v.write_videofile(str(out), logger=None, audio=True)
            return str(out)
        except Exception:
            return None

    def extract_clips(self, video: str, s_len: int, num_proc: int) -> List[str]:
        v = mpy.VideoFileClip(video)
        duration = float(v.duration)
        splits, remainder = divmod(int(duration), s_len)
        self.log(f"Extracting {splits} sublicps for {video}...", style='green')
        segments = [{f"ts{i:02}": (s_len * i, s_len * (i + 1))} for i in range(splits)]
        if remainder:
            last_key = f"ts{(int(list(segments[-1])[0][2:]) + 1) if segments else 0}"
            segments.append({last_key: (int(duration - remainder), int(duration))})
        with Pool(processes=num_proc) as pool:
            outs = list(pool.map(Pipeline._extract_clip_one, zip(segments, repeat(video))))
        return [o for o in outs if o]

    # ------------- Modalities
    def rgb_inference(self, clips: List[str]):
        for clip in tqdm(clips, desc='RGB', unit='clip'):
            self.preds[clip] = {'rgb': {}}
            self.used_mods[clip] = [0]
            with torch.no_grad():
                results = inference_recognizer(self.rgb_model, clip)
            # map to label names
            for idx, score in results:
                self.preds[clip]['rgb'][self.rgb_labels[idx]] = float(score)

    def _det_inference(self, frame_paths: List[str]) -> List[np.ndarray]:
        assert self.det_model is not None
        # sanity check: detector trained on COCO has 'person' class at index 0
        assert getattr(self.det_model, 'CLASSES', ['person'])[0] == 'person', (
            'Use a detector trained on COCO where class 0 == person')
        dets = []
        for fp in frame_paths:
            res = inference_detector(self.det_model, fp)
            person = res[0]
            keep = person[person[:, 4] >= self.args.det_score_thr] if len(person) else np.empty((0, 5))
            dets.append(keep)
        return dets

    def _pose_inference(self, frame_paths: List[str], det_results: List[np.ndarray]):
        assert self.pose_model is not None
        out = []
        for f, d in zip(frame_paths, det_results):
            d_aligned = [dict(bbox=x) for x in list(d)]
            poses = inference_top_down_pose_model(self.pose_model, f, d_aligned, format='xyxy')[0]
            out.append(poses)
        return out

    def skeleton_inference_one(self, clip: str):
        # gate: only run skeleton if rgb top-k overlaps pose-label set
        top_rgb = list(sorted(self.preds[clip]['rgb'].items(), key=lambda kv: kv[1], reverse=True))
        sk_gate_k = 3
        if set([k for k, _ in top_rgb[:sk_gate_k]]).isdisjoint(set(self.pose_labels)):
            self.log(f"Skipped {clip} (skeleton). No overlap in top {sk_gate_k} RGB labels.", style='yellow')
            self.preds[clip]['pose'] = dict(PLACEHOLDER)
            return

        frame_paths, original_frames = frame_extraction(clip, self.args.short_side)
        if not frame_paths or not original_frames:
            self.preds[clip]['pose'] = dict(PLACEHOLDER)
            return
        h, w, _ = original_frames[0].shape
        det_results = self._det_inference(frame_paths)
        torch.cuda.empty_cache()
        pose_results = self._pose_inference(frame_paths, det_results)
        torch.cuda.empty_cache()

        num_frames = len(frame_paths)
        num_person = max([len(x) for x in pose_results]) if pose_results else 0
        keypoint = np.zeros((num_person, num_frames, NUM_KEYPOINTS, 2), dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frames, NUM_KEYPOINTS), dtype=np.float16)

        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                kp = pose['keypoints']
                keypoint[j, i] = kp[:, :2]
                keypoint_score[j, i] = kp[:, 2]

        # correct-rate check
        total = max(num_person * num_frames * NUM_KEYPOINTS, 1)
        below = float((keypoint_score < self.args.pose_score_thr).sum())
        correct_rate = 1.0 - round(below / total, 3)
        if correct_rate < self.args.min_correct_rate:
            self.log(f"Clip correct-rate {correct_rate:.2f} < {self.args.min_correct_rate}. Skipped.", style='yellow')
            # cleanup extracted frames directory
            try:
                tmp_frame_dir = osp.dirname(frame_paths[0])
                shutil.rmtree(tmp_frame_dir, ignore_errors=True)
            finally:
                self.preds[clip]['pose'] = dict(PLACEHOLDER)
            return

        data = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=num_frames,
            keypoint=keypoint,
            keypoint_score=keypoint_score,
        )

        with torch.no_grad():
            results = inference_recognizer(self.sk_model, data)
        self.preds[clip]['pose'] = {}
        for idx, score in results:
            self.preds[clip]['pose'][self.pose_labels[idx]] = float(score)
        # cleanup frames
        try:
            tmp_frame_dir = osp.dirname(frame_paths[0])
            shutil.rmtree(tmp_frame_dir, ignore_errors=True)
        finally:
            pass
        self.used_mods[clip].append(1)

    def audio_inference_one(self, clip: str):
        # gate with RGB top-2 overlap on audio labels
        top_rgb = list(sorted(self.preds[clip]['rgb'].items(), key=lambda kv: kv[1], reverse=True))
        a_gate_k = 2
        if set([k for k, _ in top_rgb[:a_gate_k]]).isdisjoint(set(self.audio_labels)):
            self.log(f"Skipped {clip} (audio). No overlap in top {a_gate_k} RGB labels.", style='yellow')
            self.preds[clip]['audio'] = dict(PLACEHOLDER)
            return

        out_audio = f"{osp.splitext(clip)[0]}.wav"
        # extract audio; if the clip has no audio stream, ffmpeg will fail silently with capture_output=True
        run_ffmpeg(['ffmpeg', '-i', clip, '-map', '0:a', '-vn', '-y', out_audio])
        time.sleep(0.2)
        if not osp.exists(out_audio):
            self.log(f"No audio stream in {clip}. Skipped.", style='yellow')
            self.preds[clip]['audio'] = dict(PLACEHOLDER)
            return

        data, rate = sf.read(out_audio)
        # loudness check
        if self.loudness_weight is not None:
            try:
                meter = pyln.Meter(rate)
                if meter.integrated_loudness(data) < self.loudness_weight:
                    self.log(f"Audio not loud enough for {clip}. Skipped.", style='yellow')
                    self.preds[clip]['audio'] = dict(PLACEHOLDER)
                    os.remove(out_audio)
                    return
            except Exception:
                # if loudness estimation fails for any reason, just continue
                pass

        out_feature = f"{osp.splitext(out_audio)[0]}.npy"
        run_ffmpeg(['python', AUDIO_FEATURE_SCRIPT, out_audio, out_feature], quiet=False)
        if not osp.exists(out_feature):
            self.log(f"Audio features not generated for {clip}. Skipped.", style='yellow')
            self.preds[clip]['audio'] = dict(PLACEHOLDER)
            os.remove(out_audio)
            return

        feats = prepare_audio_input(out_feature)
        with torch.no_grad():
            results = inference_recognizer(self.audio_model, feats)
        self.preds[clip]['audio'] = {}
        for idx, score in results:
            self.preds[clip]['audio'][self.audio_labels[idx]] = float(score)

        try:
            os.remove(out_audio)
            os.remove(out_feature)
        finally:
            pass

        # bookkeeping (mod indices: rgb=0, pose=1, audio=2 if 3 coeffs else 1)
        if len(self.args.coefficients) == 3:
            self.used_mods[clip].append(2)
        else:
            self.used_mods[clip].append(1)

    # ------------- Fusion & outputs
    def weighted_scores(self, clip: str) -> Dict[str, float]:
        # build per-modality score vectors aligned to rgb label space
        scores = []
        for module, mod_scores in self.preds[clip].items():
            vec = [0.0] * len(self.rgb_labels)
            for lbl, val in mod_scores.items():
                try:
                    vec[self.rgb_labels.index(lbl)] = float(val)
                except ValueError:
                    # skip labels not in RGB label space
                    pass
            scores.append(vec)
        fused = get_weighted_score(scores, self.args.coefficients)
        # map back to label dict sorted desc
        m: Dict[str, float] = {}
        for i, sc in enumerate(fused):
            if sc > 0:
                m[self.rgb_labels[i]] = float(sc)
        # sort desc
        return dict(sorted(m.items(), key=lambda kv: kv[1], reverse=True))

    def write_video(self, original_video: str, out_path: str):
        tmp_out = f"{osp.splitext(out_path)[0]}_tmp.mp4"
        cap_in = cv2.VideoCapture(original_video)
        fps = cap_in.get(cv2.CAP_PROP_FPS) or 25
        w = int(round(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH) or 640))
        h = int(round(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360))
        cap_in.release()

        vw = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
        console.print(f"Writing results to video {out_path}...", style='bold green')

        # ensure deterministic iteration
        for c in sorted(self.preds.keys()):
            # read clip frames
            clip = cv2.VideoCapture(c)
            frames: List[np.ndarray] = []
            while True:
                ok, fr = clip.read()
                if not ok:
                    clip.release()
                    break
                frames.append(fr)

            result = self.weighted_scores(c)
            # draw top-k
            for fr in frames:
                i = 1
                for lbl, sc in list(result.items())[: self.args.topk]:
                    # scale score depending on used mods
                    n_mods = max(len(self.used_mods.get(c, [])), 1)
                    denom = sum([self.args.coefficients[m] for m in self.used_mods.get(c, [0])]) or 1.0
                    score = round((sc / n_mods) * (n_mods / denom), 4)

                    cv2.putText(fr, lbl, (10, 30 * i), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
                    cv2.putText(fr, f"{score * 100:.2f}", (210, 30 * i), FONTFACE, FONTSCALE, FONTCOLOR_SCORE, THICKNESS, LINETYPE)
                    i += 1
                vw.write(fr.astype(np.uint8))
        vw.release()

        # reattach original audio (if any)
        self._mux_audio(original_video, tmp_out, out_path)
        try:
            os.remove(tmp_out)
        except Exception:
            pass

    @staticmethod
    def _mux_audio(original_video: str, tmp_video: str, out_video: str):
        # extract audio
        audio_path = 'audio.mp3'
        run_ffmpeg(['ffmpeg', '-i', original_video, '-f', 'mp3', '-ab', '192000', '-vn', audio_path, '-y'])
        if osp.exists(audio_path):
            run_ffmpeg(['ffmpeg', '-i', tmp_video, '-i', audio_path, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', out_video, '-y'])
            try:
                os.remove(audio_path)
            finally:
                pass
        else:
            # no audio in original; just rename tmp -> out
            shutil.move(tmp_video, out_video)

    def write_json(self, out_json: str):
        results = [self.weighted_scores(c) for c in sorted(self.preds.keys())]
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        console.print(f"Wrote results to {out_json}", style='green')

    def write_timestamps(self, out_stem: str):
        results: Dict[str, str] = {}
        for idx, c in enumerate(sorted(self.preds.keys())):
            top = next(iter(self.weighted_scores(c).items()), None)
            if not top:
                continue
            start = idx * self.args.subclip_len
            end = start + self.args.subclip_len
            results[f"{start}:{end}"] = top[0]
        out = f"{osp.splitext(out_stem)[0]}_ts.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        console.print(f"Wrote timestamps to {out}", style='green')

    # ------------- Orchestration
    def run(self):
        start = time.time()
        console.print('Resizing video for faster inference...', style='green')
        original_video = self.args.video
        resized_video = self._resize_video(original_video, height=480)

        clips = self.extract_clips(resized_video, self.args.subclip_len, self.args.num_processes)
        if not clips:
            console.print('No clips extracted. Exiting.', style='bold red')
            return

        console.print('Performing RGB inference...', style='bold green')
        self.rgb_inference(clips)
        torch.cuda.empty_cache()

        # audio (optional)
        if self.args.audio_checkpoint:
            self.audio_model = init_recognizer(self.args.audio_config, self.args.audio_checkpoint, device=self.args.device)
            console.print('Performing audio inference...', style='bold green')
            for c in tqdm(sorted(self.preds.keys()), desc='AUDIO', unit='clip'):
                self.audio_inference_one(c)
            torch.cuda.empty_cache()

        # skeleton (optional)
        if self.args.skeleton_checkpoint:
            self.det_model = init_detector(self.args.det_config, self.args.det_checkpoint, self.args.device)
            self.pose_model = init_pose_model(self.args.pose_config, self.args.pose_checkpoint, self.args.device)
            self.sk_model = init_recognizer(self.args.skeleton_config, self.args.skeleton_checkpoint, self.args.device)
            console.print('Performing skeleton inference...', style='bold green')
            for c in tqdm(sorted(self.preds.keys()), desc='POSE', unit='clip'):
                self.skeleton_inference_one(c)
            torch.cuda.empty_cache()

        # outputs
        if self.args.out.endswith('.json'):
            self.write_json(self.args.out)
        else:
            self.write_video(original_video, self.args.out)
        if self.args.timestamps:
            self.write_timestamps(self.args.out)

        # cleanup temp dir
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            shutil.rmtree('./tmp', ignore_errors=True)
        finally:
            pass

        self.log(f"Finished in {round((time.time() - start) / 60, 2)} min", style='green')


# ============================ CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Image/Pose/Audio based inference')
    p.add_argument('video', help='input video file')
    p.add_argument('out', help='output file (.mp4 or .json)')
    p.add_argument('--label-maps', type=str, nargs='+', default=[
        'resources/annotations/annotations.txt',
        'resources/annotations/annotations_pose.txt',
        'resources/annotations/annotations_audio.txt',
    ], help='label maps for rgb/pose/audio')

    p.add_argument('--rgb-config', default='checkpoints/har/timesformer_divST_16x12x1_kinetics.py')
    p.add_argument('--rgb-checkpoint', default='checkpoints/har/timeSformer.pth')

    p.add_argument('--skeleton-config', default='checkpoints/har/slowonly_u54_kinetics.py')
    p.add_argument('--skeleton-checkpoint', default='checkpoints/har/posec3d.pth')

    p.add_argument('--audio-config', default='checkpoints/har/audioonly_64x1x1.py')
    p.add_argument('--audio-checkpoint', default='checkpoints/har/audio.pth')

    p.add_argument('--pose-config', default='mmaction2/demo/hrnet_w32_coco_256x192.py')
    p.add_argument('--pose-checkpoint', default='checkpoints/pose/hrnet_w32_coco_256x192.pth')

    p.add_argument('--det-config', default='mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py')
    p.add_argument('--det-checkpoint', default='checkpoints/detector/faster_rcnn_r50_fpn_1x_coco-person.pth')

    p.add_argument('--det-score-thr', type=float, default=0.8, help='human detection score threshold')
    p.add_argument('--num-processes', type=int, default=4, help='processes for subclip extraction')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--subclip-len', type=int, default=7, help='duration (s) of sliding-window clips')
    p.add_argument('--short-side', type=int, default=480, help='short side for pose frame extraction')
    p.add_argument('--coefficients', nargs='+', type=float, default=[0.5, 0.6, 1.0], help='weights for rgb, skeleton, audio')
    p.add_argument('--pose-score-thr', type=float, default=0.4, help='min keypoint score to count as valid')
    p.add_argument('--min-correct-rate', type=float, default=0.05, help='min fraction of valid keypoints to keep a clip')
    p.add_argument('--loudness-weights', type=str, default='resources/audio/db_20_config.yml', help='audio loudness gating file')
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--timestamps', action='store_true')
    p.add_argument('--verbose', action='store_true')
    return p


def parse_args() -> Args:
    p = build_parser()
    a = p.parse_args()
    return Args(
        video=a.video,
        out=a.out,
        label_maps=a.label_maps,
        rgb_config=a.rgb_config,
        rgb_checkpoint=a.rgb_checkpoint,
        skeleton_config=a.skeleton_config,
        skeleton_checkpoint=a.skeleton_checkpoint,
        audio_config=a.audio_config,
        audio_checkpoint=a.audio_checkpoint,
        pose_config=a.pose_config,
        pose_checkpoint=a.pose_checkpoint,
        det_config=a.det_config,
        det_checkpoint=a.det_checkpoint,
        det_score_thr=a.det_score_thr,
        num_processes=a.num_processes,
        device=a.device,
        subclip_len=a.subclip_len,
        short_side=a.short_side,
        coefficients=a.coefficients,
        pose_score_thr=a.pose_score_thr,
        min_correct_rate=a.min_correct_rate,
        loudness_weights=a.loudness_weights,
        topk=a.topk,
        timestamps=a.timestamps,
        verbose=a.verbose,
    )


def main():
    args = parse_args()
    pl = Pipeline(args)
    pl.run()


if __name__ == '__main__':
    main()

