#!/usr/bin/env python3
"""Export MatAnyone2 single-object matting to CoreML (.mlpackage) for the ANE.

Splits the model into 5 stateless CoreML programs at clean stage boundaries; the stateful glue
(memory bank, get_affinity, readout matmul) is done by the caller (Swift / the Python harness).
Real steady-state input tensors are captured by recording the model's own method calls during a
live inference step, so shapes/dtypes are exact.

Run from scripts/ (coreml_prod_op is now a local module):

    PYTHONPATH=/tmp/matanyone2:. uv run python export.py

Outputs: scripts/models/<name>.mlpackage  +  models/manifest.json
(compile to .mlmodelc and copy into ../Sources/MatAnyoneKitCoreML/Resources/MatAnyone/ to ship)
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from omegaconf import OmegaConf

import matanyone2.model.matanyone2 as m2
from matanyone2.model.matanyone2 import MatAnyone2
from matanyone2.inference.inference_core import InferenceCore

import coreml_prod_op  # registers aten::prod -> MIL reduce_prod (noqa)
import math
import contextlib


@contextlib.contextmanager
def decomposed_sdpa():
    """Trace `nn.MultiheadAttention` (need_weights=False) as primitive matmul/softmax instead of the
    fused `scaled_dot_product_attention` aten op.

    At `minimum_deployment_target >= iOS18`, coremltools maps that aten op to the MIL
    `ios18.scaled_dot_product_attention`, which the device ANE backend has no EIR kernel for — so
    `readout` fails MIL->EIR translation and won't even load. Decomposing to primitives keeps iOS18
    while letting the attention run (and stay ANE-eligible). Numerically identical to the documented
    SDPA semantics (bool mask: keep where True; float mask: additive).
    """
    orig = torch.nn.functional.scaled_dot_product_attention

    def manual(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
               scale=None, enable_gqa=False):
        s = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * s
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    torch.nn.functional.scaled_dot_product_attention = manual
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


def _rebind(cls, method_name, replacements):
    """Source-rewrite a method (string replacements) and rebind it on the class, reproducibly."""
    import inspect
    src = inspect.getsource(getattr(cls, method_name))
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new); changed = True
    if not changed:
        return  # already patched upstream
    ns = dict(vars(inspect.getmodule(cls)))
    exec(textwrap.dedent(src), ns)
    setattr(cls, method_name, ns[method_name])


def patch_ane_hostile_ops():
    """Make the `readout` graph ANE/GPU-safe on the A18.

    Two ops in `QueryTransformer` lower to forms the A18 ANE+GPU reject (the macOS Espresso path
    tolerates them, which is why parity passed but the phone crashed):

    1. `forward`: `obj_summaries[..., :-1]` / `[..., -1:]` — negative-index slices of a runtime
       input. The MIL CPU interpreter even *traps* (EXC_BREAKPOINT). → static positive indices.
    2. `_get_aux_mask`: `aux_mask[torch.where(row_all_masked)] = False` — a data-dependent
       `nonzero` + dynamic `slice_by_index` (`generic_general_slice: Invalid begin_ids` on ANE,
       `slice start -1` on GPU). → equivalent static elementwise `aux_mask & ~all_masked`.

    Both rewrites are numerically identical to the originals.
    """
    from matanyone2.model.transformer.object_transformer import QueryTransformer
    _rebind(QueryTransformer, "forward", [
        ("obj_summaries[:, :, :, :-1]", "obj_summaries[:, :, :, :self.embed_dim]"),
        ("obj_summaries[:, :, :, -1:]", "obj_summaries[:, :, :, self.embed_dim:self.embed_dim + 1]"),
    ])
    _rebind(QueryTransformer, "_get_aux_mask", [
        ("aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False",
         "all_masked = (aux_mask.sum(-1, keepdim=True) == aux_mask.shape[-1])\n"
         "        aux_mask = aux_mask & ~all_masked"),
    ])


def patch_eca_rank4():
    """Keep `CAResBlock`'s ECA channel-attention rank-4 so the A18 ANE will compile it.

    The stock block pools to [B,C,1,1], then `view(b,1,c)` -> rank-3, runs `Conv1d(1,1,k)`
    across channels, then `transpose(-1,-2).unsqueeze(-1)`. That rank-3 Conv1d-over-channels
    wrapped in reshape/transpose is the structure the device ANE backend rejects
    (`ANECCompile FAILED (11)`), which is why both models that contain a `CAResBlock`
    (maskencoder.fuser, readout's pixel_fuser + object_transformer pixel_ffn) fail to load on
    the ANE while the CAResBlock-free decoder compiles fine.

    A 1-D conv of kernel k over the C-length channel vector is identical to a 2-D conv with
    kernel (k, 1) over a [B, 1, C, 1] tensor (channels carried on the H axis). Staying NCHW the
    whole way keeps the block ANE-eligible. Numerically identical to the original.
    """
    import torch.nn.functional as F
    from matanyone2.model.group_modules import CAResBlock

    def forward(self, x):
        r = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        b, c = x.shape[:2]
        w = self.pool(x)                       # [b, c, 1, 1]
        w = w.reshape(b, 1, c, 1)              # [b, 1, c, 1] — channels on H, stays rank-4
        k2 = self.conv.weight.unsqueeze(-1)    # [1, 1, k] -> [1, 1, k, 1]
        w = F.conv2d(w, k2, None, stride=1, padding=(self.conv.padding[0], 0))
        w = w.reshape(b, c, 1, 1).sigmoid()    # B*C*1*1
        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w
        return x

    CAResBlock.forward = forward


import textwrap  # noqa: E402
patch_ane_hostile_ops()
patch_eca_rank4()

REPO = "PeiqingYang/MatAnyone2"
EVAL_CFG = "/tmp/matanyone2/matanyone2/config/eval_matanyone_config.yaml"
H, W = 512, 288
OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE
DEPLOY_TARGET = ct.target.iOS18


def record_method(obj, name, store):
    """Shadow a bound method (via __dict__) so the first call's (args, kwargs) is stashed."""
    fn = getattr(obj, name)

    def wrapper(*a, **k):
        if name not in store:
            store[name] = (a, k)
        return fn(*a, **k)

    object.__setattr__(obj, name, wrapper)  # bypass nn.Module.__setattr__
    return name


def unrecord(obj, name):
    obj.__dict__.pop(name, None)


def to_inputs(names, tensors):
    return [ct.TensorType(name=n, shape=tuple(t.shape)) for n, t in zip(names, tensors)]


def convert_save(name, wrapper, in_names, tensors, out_names):
    wrapper = wrapper.eval()
    with torch.inference_mode(), decomposed_sdpa():
        traced = torch.jit.trace(wrapper, tensors)
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=to_inputs(in_names, tensors),
        outputs=[ct.TensorType(name=n) for n in out_names],
        minimum_deployment_target=DEPLOY_TARGET,
        compute_units=COMPUTE_UNITS,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    mlmodel.save(path)
    feeds = {n: t.numpy() for n, t in zip(in_names, tensors)}

    # numerical parity vs PyTorch (the fp16/ANE pipeline introduces small error). The host may be
    # an older macOS than the target opset (e.g. iOS26 / CoreML9) and unable to predict() locally —
    # in that case skip parity/timing but still save the model (verify parity on-device instead).
    med = 0.0
    try:
        with torch.inference_mode():
            ref = wrapper(*tensors)
        ref = (ref,) if torch.is_tensor(ref) else ref
        pred = mlmodel.predict(feeds)
        diffs = []
        for n, r in zip(out_names, ref):
            r = r.detach().numpy()
            p = pred[n]
            diffs.append(f"{n} maxΔ {np.abs(r - p).max():.4f} (|r|≤{np.abs(r).max():.2f})")
        for _ in range(3):
            mlmodel.predict(feeds)
        ts = [(_t := time.perf_counter(), mlmodel.predict(feeds), (time.perf_counter() - _t) * 1000)[2]
              for _ in range(20)]
        med = float(np.median(ts))
        print(f"  saved {name}.mlpackage  ({time.perf_counter()-t0:.1f}s convert, {med:.1f} ms ANE)")
        print(f"    parity: " + " | ".join(diffs))
    except Exception as e:
        print(f"  saved {name}.mlpackage  ({time.perf_counter()-t0:.1f}s convert) "
              f"— host predict() unavailable for this opset, parity deferred to device ({str(e)[:60]})")
    return {
        "name": name,
        "path": f"models/{name}.mlpackage",
        "inputs": [{"name": n, "shape": list(t.shape)} for n, t in zip(in_names, tensors)],
        "outputs": out_names,
        "median_ms_mac_ane": round(med, 2),
    }


# ----------------------------------------------------------------- wrappers
class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, image):
        ms, pix_feat = self.m.encode_image(image)
        key, shrinkage, selection = self.m.transform_key(ms[0])
        return ms[0], ms[1], ms[2], ms[3], ms[4], pix_feat, key, shrinkage, selection


class UncertWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, last_pix_feat, cur_pix_feat, last_mask, mem_val_diff):
        return self.m.pred_uncertainty(last_pix_feat, cur_pix_feat, last_mask, mem_val_diff)["prob"]


class ReadoutWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, pix_feat, pixel, sensory, last_mask, obj_memory):
        fused = self.m.pixel_fusion(pix_feat, pixel, sensory, last_mask)
        mem_readout, _ = self.m.readout_query(fused, obj_memory, selector=None, seg_pass=False)
        return mem_readout


class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, f16, f8, f4, f2, f1, memory_readout, sensory):
        new_sensory, logits = self.m.mask_decoder(
            [f16, f8, f4, f2, f1], memory_readout, sensory,
            chunk_size=-1, update_sensory=True)
        return logits, new_sensory


class MaskEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, image, pix_feat, sensory, masks):
        mask_value, new_sensory = self.m.mask_encoder(
            image, pix_feat, sensory, masks, None, deep_update=True, chunk_size=-1)
        return mask_value, new_sensory


class ObjectSummarizerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, masks, mask_value):
        obj_summaries, _ = self.m.object_summarizer(masks, mask_value, False)
        return obj_summaries


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m2.device = torch.device("cpu")
    print(f"Loading {REPO} ...")
    model = MatAnyone2.from_pretrained(REPO).eval().to("cpu")
    cfg = OmegaConf.load(EVAL_CFG); cfg.model = model.cfg.model
    core = InferenceCore(model, cfg=cfg, device="cpu")
    print("Loaded. coremltools", ct.__version__)

    # Warm to steady state.
    f0 = torch.rand(3, H, W); seed = torch.rand(H, W) * 255
    with torch.inference_mode():
        core.step(f0, seed, objects=[1])
        core.step(f0, first_frame_pred=True)
        for _ in range(2):
            core.step(f0, first_frame_pred=True)
        for _ in range(6):
            core.step(torch.rand(3, H, W))

    # Record one steady-state frame's real method inputs. Pure methods are shadowed via
    # __dict__; mask_decoder is an nn.Module so we hook it instead.
    store = {}
    method_names = ["pred_uncertainty", "pixel_fusion", "readout_query", "encode_mask"]
    for n in method_names:
        record_method(model, n, store)

    def dec_hook(module, args, kwargs):
        store.setdefault("mask_decoder", (args, kwargs))

    def objsum_hook(module, args, kwargs):
        store.setdefault("object_summarizer", (args, kwargs))

    hdec = model.mask_decoder.register_forward_pre_hook(dec_hook, with_kwargs=True)
    hobj = model.object_summarizer.register_forward_pre_hook(objsum_hook, with_kwargs=True)
    with torch.inference_mode():
        core.step(torch.rand(3, H, W))
    hdec.remove()
    hobj.remove()
    for n in method_names:
        unrecord(model, n)

    print("captured methods:", {k: [tuple(t.shape) for t in v[0] if torch.is_tensor(t)]
                                 for k, v in store.items()})

    manifest = {"working_h": H, "working_w": W, "compute_units": "CPU_AND_NE", "models": []}

    img = torch.rand(1, 3, H, W)
    manifest["models"].append(convert_save(
        "encoder", EncoderWrapper(model),
        ["image"], [img],
        ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"]))

    ua, _ = store["pred_uncertainty"]
    manifest["models"].append(convert_save(
        "uncert", UncertWrapper(model),
        ["last_pix_feat", "cur_pix_feat", "last_mask", "mem_val_diff"], list(ua[:4]),
        ["prob"]))

    pa, _ = store["pixel_fusion"]; ra, _ = store["readout_query"]
    manifest["models"].append(convert_save(
        "readout", ReadoutWrapper(model),
        ["pix_feat", "pixel", "sensory", "last_mask", "obj_memory"], [pa[0], pa[1], pa[2], pa[3], ra[1]],
        ["mem_readout"]))

    da, _ = store["mask_decoder"]
    ms_feat, mem_readout, sensory = da[0], da[1], da[2]
    manifest["models"].append(convert_save(
        "decoder", DecoderWrapper(model),
        ["f16", "f8", "f4", "f2", "f1", "memory_readout", "sensory"],
        [ms_feat[0], ms_feat[1], ms_feat[2], ms_feat[3], ms_feat[4], mem_readout, sensory],
        ["logits", "new_sensory"]))

    ea, _ = store["encode_mask"]
    manifest["models"].append(convert_save(
        "maskencoder", MaskEncoderWrapper(model),
        ["image", "pix_feat", "sensory", "masks"], [ea[0], ea[1], ea[2], ea[3]],
        ["mask_value", "new_sensory"]))

    # Split out of maskencoder so the conv-heavy encoder runs every frame on the ANE, while the
    # rank-5 / matmul / reduce object_summarizer (ANE-hostile) runs on GPU only on memory frames.
    oa, _ = store["object_summarizer"]
    manifest["models"].append(convert_save(
        "objsummary", ObjectSummarizerWrapper(model),
        ["masks", "mask_value"], [oa[0], oa[1]],
        ["obj_summaries"]))

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nWrote manifest.json with", len(manifest["models"]), "models.")


if __name__ == "__main__":
    main()
