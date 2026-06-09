#!/usr/bin/env python3
"""Dump end-to-end reference data for the Swift Core ML pipeline parity harness.

Saves the exact frame sequence (incl. warmup duplication), the seed mask, and the
PyTorch reference alphas as raw float32 blobs + a manifest. The Swift Mac harness
(`coreml/e2e_validate.swift`) replays the same step sequence through the Core ML
models + Swift orchestration and compares alphas.

    PYTHONPATH=/tmp/matanyone2:reference .venv/bin/python coreml/dump_e2e_ref.py
"""
import json
import os
import numpy as np
import torch
from omegaconf import OmegaConf

import matanyone2.model.matanyone2 as m2
from matanyone2.model.matanyone2 import MatAnyone2
from matanyone2.inference.inference_core import InferenceCore

EVAL_CFG = "/tmp/matanyone2/matanyone2/config/eval_matanyone_config.yaml"
OUT = "/tmp/e2eref"
H, W = 512, 288
N_FRAMES = 10
N_WARMUP = 2


def load_real_clip():
    import cv2
    base = os.path.join(os.path.dirname(__file__), "..", "inputs")
    cap = cv2.VideoCapture(os.path.join(base, "test-portrait.mp4"))
    frames = []
    while len(frames) < N_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    cap.release()
    seed_img = cv2.imread(os.path.join(base, "test-portrait_seed.png"), cv2.IMREAD_GRAYSCALE)
    seed_img = cv2.resize(seed_img, (W, H)).astype(np.float32)
    return frames, torch.from_numpy(seed_img)


def main():
    os.makedirs(OUT, exist_ok=True)
    m2.device = torch.device("cpu")
    model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2").eval().to("cpu")
    cfg = OmegaConf.load(EVAL_CFG); cfg.model = model.cfg.model

    frames, seed = load_real_clip()
    frames = [frames[0]] * N_WARMUP + frames

    core = InferenceCore(model, cfg=cfg, device="cpu")
    alphas = []
    with torch.inference_mode():
        for ti, img in enumerate(frames):
            if ti == 0:
                core.step(img, seed, objects=[1])
                out = core.step(img, first_frame_pred=True)
            elif ti <= N_WARMUP:
                out = core.step(img, first_frame_pred=True)
            else:
                out = core.step(img)
            if ti >= N_WARMUP:
                alphas.append(core.output_prob_to_mask(out).numpy().astype("<f4"))

    # frames as [T,3,H,W]; seed [H,W]; alphas [Nout,H,W]
    np.stack([f.numpy() for f in frames]).astype("<f4").tofile(os.path.join(OUT, "frames.bin"))
    seed.numpy().astype("<f4").tofile(os.path.join(OUT, "seed.bin"))
    np.stack(alphas).astype("<f4").tofile(os.path.join(OUT, "alphas.bin"))

    manifest = {"H": H, "W": W, "T": len(frames), "n_warmup": N_WARMUP, "n_out": len(alphas)}
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(frames)} frames, seed, {len(alphas)} ref alphas to {OUT}")
    a = np.stack(alphas)
    print(f"  alpha range [{a.min():.3f}, {a.max():.3f}]  fg cov {(a>0.5).mean():.2%}")


if __name__ == "__main__":
    main()
