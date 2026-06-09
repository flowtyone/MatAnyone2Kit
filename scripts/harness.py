#!/usr/bin/env python3
"""Phase 2: end-to-end CoreML parity harness for MatAnyone2 single-object matting.

Runs the full InferenceCore loop twice on the same frames:
  (1) pure PyTorch (reference)
  (2) the 5 model boundaries swapped for the exported CoreML packages, while the stateful
      memory manager (get_affinity / readout / bank) stays PyTorch.

Then reports per-frame final-alpha parity. This validates the orchestration I'll port to
Swift AND tells us whether the fp16 intermediates (shrinkage, mem_readout) actually matter
for the matte.

    Run from scripts/:
    PYTHONPATH=/tmp/matanyone2:. uv run python harness.py
"""
import json
import os
import numpy as np
import torch
import coremltools as ct
from omegaconf import OmegaConf

import matanyone2.model.matanyone2 as m2
from matanyone2.model.matanyone2 import MatAnyone2
from matanyone2.inference.inference_core import InferenceCore

EVAL_CFG = "/tmp/matanyone2/matanyone2/config/eval_matanyone_config.yaml"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
H, W = 512, 288
N_FRAMES = 10
N_WARMUP = 2


def load_models():
    manifest = json.load(open(os.path.join(MODELS_DIR, "manifest.json")))
    models = {}
    for m in manifest["models"]:
        models[m["name"]] = ct.models.MLModel(
            os.path.join(MODELS_DIR, os.path.basename(m["path"])),
            compute_units=ct.ComputeUnit.CPU_AND_NE)
    return models


def load_real_clip():
    """First N_FRAMES of the portrait test clip at H×W, RGB [0,1], + the seed mask in [0,255]."""
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
    seed = torch.from_numpy(seed_img)
    print(f"loaded {len(frames)} real frames + seed (fg coverage {(seed_img>127).mean():.2%})")
    return frames, seed


def run_cml(mlmodel, **kw):
    feeds = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in kw.items()}
    out = mlmodel.predict(feeds)
    return {k: torch.from_numpy(np.asarray(v).astype(np.float32)) for k, v in out.items()}


def run_reference(model, cfg, frames, seed):
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
                alphas.append(core.output_prob_to_mask(out).numpy())
    return alphas


def patch_to_coreml(model, M):
    """Shadow the 5 model boundaries with CoreML-backed callables. Memory ops stay torch."""
    stash = {}

    def encode_feature(self, index, image, last_feats=None):
        o = run_cml(M["encoder"], image=image)
        ms = [o["f16"], o["f8"], o["f4"], o["f2"], o["f1"]]
        self._store[index] = (ms, o["pix_feat"], o["key"], o["shrinkage"], o["selection"])

    def mask_decoder(*a, **k):
        ms, mr, sens = a[0], a[1], a[2]
        o = run_cml(M["decoder"], f16=ms[0], f8=ms[1], f4=ms[2], f2=ms[3], f1=ms[4],
                    memory_readout=mr, sensory=sens)
        return o["new_sensory"], o["logits"]

    def encode_mask(image, pix_feat, sensory, masks, deep_update=True, chunk_size=-1,
                    need_weights=False):
        o = run_cml(M["maskencoder"], image=image, pix_feat=pix_feat, sensory=sensory, masks=masks)
        return o["mask_value"], o["new_sensory"], o["obj_summaries"], None

    def pred_uncertainty(last_pix_feat, cur_pix_feat, last_mask, mem_val_diff):
        o = run_cml(M["uncert"], last_pix_feat=last_pix_feat, cur_pix_feat=cur_pix_feat,
                    last_mask=last_mask, mem_val_diff=mem_val_diff)
        return {"prob": o["prob"]}

    def pixel_fusion(pix_feat, pixel, sensory, last_mask, chunk_size=-1):
        stash["fusion"] = (pix_feat, pixel, sensory, last_mask)
        return None

    def readout_query(pixel_readout, obj_memory, selector=None, need_weights=False, seg_pass=False):
        pf = stash["fusion"]
        o = run_cml(M["readout"], pix_feat=pf[0], pixel=pf[1], sensory=pf[2], last_mask=pf[3],
                    obj_memory=obj_memory)
        return o["mem_readout"], None

    object.__setattr__(model, "mask_decoder", mask_decoder)
    object.__setattr__(model, "encode_mask", encode_mask)
    object.__setattr__(model, "pred_uncertainty", pred_uncertainty)
    object.__setattr__(model, "pixel_fusion", pixel_fusion)
    object.__setattr__(model, "readout_query", readout_query)
    return encode_feature


def main():
    m2.device = torch.device("cpu")
    print("Loading model + CoreML packages ...")
    model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2").eval().to("cpu")
    cfg = OmegaConf.load(EVAL_CFG); cfg.model = model.cfg.model
    M = load_models()

    frames, seed = load_real_clip()
    frames = [frames[0]] * N_WARMUP + frames

    print(f"Running PyTorch reference ({len(frames)} frames @ {H}x{W}) ...")
    ref_alphas = run_reference(model, cfg, frames, seed)

    print("Patching boundaries to CoreML; running CoreML pipeline ...")
    encode_feature = patch_to_coreml(model, M)
    core = InferenceCore(model, cfg=cfg, device="cpu")
    # patch the store instance method
    import types
    core.image_feature_store._encode_feature = types.MethodType(encode_feature, core.image_feature_store)

    cml_alphas = []
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
                cml_alphas.append(core.output_prob_to_mask(out).numpy())

    print(f"\nfinal-alpha parity over {len(ref_alphas)} frames (alpha in [0,1]):")
    all_d = []
    for i, (r, c) in enumerate(zip(ref_alphas, cml_alphas)):
        d = np.abs(r - c)
        all_d.append(d)
        n = d.size
        print(f"  frame {i}: maxΔ {d.max():.3f}  meanΔ {d.mean():.5f}  "
              f"p99.9 {np.percentile(d,99.9):.3f}  "
              f">.1: {(d>0.1).sum()/n:.3%}  >.25: {(d>0.25).sum()/n:.4%}")
    d = np.concatenate([x.ravel() for x in all_d])
    print(f"\noverall: meanΔ {d.mean():.5f}  p99 {np.percentile(d,99):.3f}  "
          f"p99.9 {np.percentile(d,99.9):.3f}  maxΔ {d.max():.3f}")
    print(f"pixels >0.1: {(d>0.1).mean():.3%}   >0.25: {(d>0.25).mean():.4%}   "
          f">0.5: {(d>0.5).mean():.5%}")


if __name__ == "__main__":
    main()
