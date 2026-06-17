# Licensing notice

MatAnyone2Kit is distributed under **two** licenses:

| Component | Location | License |
|-----------|----------|---------|
| Swift package source code | everything except the folder below | GNU GPL-3.0 (see [`LICENSE`](LICENSE)) |
| Bundled MatAnyone2 model weights | [`Sources/MatAnyoneKitCoreML/Resources/MatAnyone/`](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/) | NTU S-Lab License 1.0 — **non-commercial** |

The bundled Core ML models are a conversion of the **MatAnyone2** weights by
S-Lab, Nanyang Technological University
(https://github.com/pq-yang/MatAnyone2). They remain governed by the
[NTU S-Lab License 1.0](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/LICENSE),
which permits **non-commercial use only**. The GPL-3.0 license of this package's
own source code does **not** grant any commercial rights to those weights.

In GPL terms, the weights are an *aggregate* shipped alongside the GPL'd code,
not a derivative of it. Bundling them does not relicense them.

**Practical effect:** using or redistributing this package *with the bundled
weights* is non-commercial only. For commercial use of the weights, contact the
authors (see the weights
[`NOTICE.md`](Sources/MatAnyoneKitCoreML/Resources/MatAnyone/NOTICE.md)).
