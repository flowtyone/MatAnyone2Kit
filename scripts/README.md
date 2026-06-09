# MatAnyone2 → Core ML / ANE

This folder is the conversion toolchain that turns the upstream PyTorch
[MatAnyone2](https://github.com/pq-yang/MatAnyone) single-object matting model into the six Core ML
programs that ship inside [`../Sources/MatAnyoneKitCoreML`](../Sources/MatAnyoneKitCoreML) and run
**real-time on the Apple Neural Engine** (stable 30 fps on an iPhone 16 / A18).

To our knowledge this is the first Core ML/ANE port of MatAnyone2. The notes below document *what*
had to change to make a research video-matting model compile for, and stay resident on, the ANE — and
the Swift-side work that got it from "runs" to "runs at 30 fps".

## How the model is split

MatAnyone2 is a recurrent, memory-based matting network. We export it as **six stateless Core ML
programs** cut at clean stage boundaries, and keep the *stateful* glue (the memory bank, key/affinity
matching, top-k softmax readout, sensory state) in Swift. Per-frame data flows:

```
                       ┌─────────── per frame ───────────┐
camera frame ──▶ encoder ──▶ (key/shrinkage/selection, pix_feat, f16..f1)
                   │
                   ├─▶ readout  (memory_readout from obj_memory + affinity readout)
                   ├─▶ decoder  (logits → alpha, new_sensory)
                   ├─▶ uncert   (per-frame uncertainty prob)
                   └─▶ maskencoder ──▶ objsummary   (only on memory frames)
                       └────────── every memEvery frames ──────────┘
```

| model         | role                                              | runs            | inputs (key shapes)                | ~ms (Mac M2 ANE) |
|---------------|---------------------------------------------------|-----------------|------------------------------------|------------------|
| `encoder`     | image → multi-scale feats + key/shrinkage         | every frame     | `image [1,3,512,288]`              | 9.0  |
| `readout`     | obj-memory + affinity → memory readout            | every frame     | `pix_feat, pixel, sensory, …`      | 2.92 |
| `decoder`     | feats + readout → alpha logits + new sensory      | every frame     | `f16..f1, memory_readout, sensory` | 7.81 |
| `uncert`      | temporal uncertainty                              | every frame     | `last/cur_pix_feat, last_mask, …`  | 0.55 |
| `maskencoder` | image + mask → mask value (memory write)          | every memEvery  | `image, pix_feat, sensory, masks`  | 2.0  |
| `objsummary`  | object summarizer (rank-5 matmul/reduce)          | every memEvery  | `masks, mask_value`                | 0.54 |

Working resolution is portrait **288×512** (`manifest.json: working_w/working_h`). The matte comes
back as `[H*W]` and is framed identically to the camera frame (no letterbox), so a compositor samples
it with the same aspect-fill UVs.

## ANE optimizations (what made it compile + hit 30 fps)

The A18 ANE compiler (and the MIL→EIR translator) reject several patterns that a research model emits
freely. Each of these was found by loading `MLComputePlan` on device and logging exactly which ops
fell off the ANE, then patching the export graph (`export.py`) until the per-frame models were ANE-
resident. In rough order of impact:

1. **`CAResBlock` ECA attention: rank-3 `Conv1d` → rank-4 `Conv2d`** — the key
   `ANECompile() FAILED (11)` fix. The efficient-channel-attention block reshaped to rank-3 and ran a
   `Conv1d`; the ANE compiler rejected that whole subgraph (and dragged ~8 `relu`s off the ANE with
   it). `export.py` rewrites the block to keep everything rank-4 with an equivalent `Conv2d`, which
   made `maskencoder` fully ANE-eligible and `readout` almost fully.
2. **SDPA decomposition** — at `minimum_deployment_target >= iOS18`, coremltools maps
   `nn.MultiheadAttention` to the fused `ios18.scaled_dot_product_attention` MIL op, which the device
   ANE backend has *no EIR kernel* for, so `readout` failed MIL→EIR translation and wouldn't even
   load. The `decomposed_sdpa()` context manager monkeypatches attention into primitive
   matmul/softmax (numerically identical), keeping iOS18 while staying ANE-eligible.
3. **Negative-index slices → static positive indices** — `obj_summaries[..., :-1]` / `[..., -1:]`
   style slices crash the ANE/GPU compilers (`generic_general_slice: Invalid values 0`). Rewritten to
   static positive ranges.
4. **Data-dependent `torch.where` → static elementwise** — `aux_mask[torch.where(...)] = False`
   compiles to a dynamic nonzero/scatter the ANE can't take; rewritten to
   `aux_mask = aux_mask & ~all_masked`.
5. **Splitting `objsummary` off the per-frame path** — the object summarizer is rank-5 with
   matmul/reduce that broke `maskencoder`'s ANE compile. It became its own model, pinned to the GPU
   (`unitOverrides["objsummary"] = .cpuAndGPU`) and only run on memory frames, so the conv-only
   `maskencoder` hits the ANE every frame.
6. **`aten::prod` registration** — `coreml_prod_op.py` registers an `aten::prod` → MIL `reduce_prod`
   translation that coremltools was missing for this graph.

A leftover `identity×1` op on `decoder` is benign — it's a terminal op that doesn't split the ANE
graph and doesn't cost anything.

## Swift-side wins (in the package)

The model time was only ~half the budget; the stateful glue had to keep up too:

- **Parallelized `topKSoftmax`** across independent columns with `DispatchQueue.concurrentPerform`
  (~11 ms → ~2.5 ms).
- **SIMD fp16 readback** — `MLMultiArray` fp16→fp32 via `vImageConvert_Planar16FtoPlanarF` plus bulk
  `memcpy` for contiguous trailing runs, instead of per-element `NSNumber` subscripting.
- **Vectorized `blend`** with `vDSP_vsub`/`vDSP_vma`.
- **Back-to-back streamer** — keep only the freshest pending camera frame and run matte passes
  back-to-back instead of on the camera tick, which broke the 33 ms `alwaysDiscardsLateVideoFrames`
  cliff that pinned throughput at 15 fps. (App-side, in your own capture loop.)

## Environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From this directory (`scripts/`):

```bash
uv sync          # creates ./.venv (Python 3.11) with torch, coremltools, omegaconf, numpy
```

`uv run python …` then runs inside that venv. Versions are pinned in `pyproject.toml` / `uv.lock`
(notably `torch==2.7.0`, the version coremltools 9 was tested against). The `.venv` is gitignored.

The upstream MatAnyone2 PyTorch repo + weights are *not* pip-installable — clone them to
`/tmp/matanyone2` and put them on `PYTHONPATH`. `coreml_prod_op` is a local module, so `.` must also
be on `PYTHONPATH`.

## Reproducing the conversion

Run everything **from this directory** (`scripts/`):

```bash
# 1. Export the 6 .mlpackage programs + manifest.json into ./models/
PYTHONPATH=/tmp/matanyone2:. uv run python export.py

# 2. Compile each to .mlmodelc and stage them in the package (precompiled = no launch-time compile)
DST=../Sources/MatAnyoneKitCoreML/Resources/MatAnyone
cp models/manifest.json "$DST/"
for m in encoder uncert readout decoder maskencoder objsummary; do
  xcrun coremlcompiler compile "models/$m.mlpackage" "$DST/"
done
```

The package loader prefers a sibling `.mlmodelc` over the `.mlpackage`, so the bundled precompiled
models load with no compile step at launch.

## Validation harness

End-to-end numerical parity against the PyTorch reference — this replays the reference step pattern
through the exported models and the Swift memory math, so it covers the memory bank / affinity /
top-k softmax readout too:

```bash
PYTHONPATH=/tmp/matanyone2:. uv run python dump_e2e_ref.py
swiftc -O -parse-as-library e2e_validate.swift \
  ../Sources/MatAnyoneKitCoreML/MemoryBank.swift \
  ../Sources/MatAnyoneKitCoreML/MatAnyoneCoreML.swift \
  ../Sources/MatAnyoneKitCoreML/MatAnyoneCoreMLEngine.swift \
  -o /tmp/e2evalidate && /tmp/e2evalidate
```

## Files

| file                  | purpose                                                              |
|-----------------------|----------------------------------------------------------------------|
| `pyproject.toml` / `uv.lock` | pinned Python deps (torch/coremltools/omegaconf/numpy)        |
| `export.py`           | the converter — model split, ANE graph rewrites, `.mlpackage` export |
| `coreml_prod_op.py`   | registers `aten::prod` → MIL `reduce_prod` (export-time dep)          |
| `harness.py`          | Python reference harness for the exported models                     |
| `dump_e2e_ref.py`     | dumps the end-to-end PyTorch reference (alphas + step pattern)       |
| `e2e_validate.swift`  | Swift end-to-end parity harness                                      |
