# Third-party notices

This release bundles or depends on the following third-party code. Each item is licensed only
under its own terms; nothing here implies those components fall under the repository's (pending)
license.

| Component | Where | License / note |
|---|---|---|
| INP-Former (Wei Luo) | `src/models`, `src/data`, `src/utils.py`, `src/optimizers`, `src/continual` | MIT, (c) 2025 Wei Luo — see LICENSE §1 |
| DINOv2 (`facebookresearch/dinov2`) | `src/backbones/dinov2/` (pruned `models/` + `layers/`) | Apache-2.0, (c) Meta Platforms — source headers confirm LICENsE in upstream root; version marker `0.0.1` in tree |
| DINOv1 (`facebookresearch/dino`) | `src/backbones/dinov1/` | See upstream: https://github.com/facebookresearch/dino |
| BEiT-v2 (`microsoft/beit`) | `src/backbones/beit/` | See upstream: https://github.com/microsoft/unilm (veit/beit) |
| ADEval (`winggan/adeval`) | runtime dependency, imported by `src/utils.py` | https://github.com/winggan/adeval — v1.1.0 |
| torch / torchvision | runtime | BSD-style (PyTorch); CUDA builds carry NVIDIA terms |
| timm, kornia | runtime | Apache-2.0 (timm), Apache-2.0 (kornia) |
| scikit-learn, scikit-image, opencv | runtime | BSD-3-Clause (sklearn/skimage), Apache-2.0 (opencv) |
| MVTec AD dataset | data (not redistributed) | (c) MVTec Software GmbH, CC BY-NC-SA 4.0 — users download it |
| UCAD | **not included** | No explicit upstream license; only URL + commit cited in README |
| DINOv2/BEiT random-weights checkpoints | downloaded by `src/models/vit_encoder.py` at runtime | Released by the upstream authors (fb, microsoft) under their own terms |

We attempt to preserve upstream license headers in vendored files. If a header is missing for a
component that requires it, that is an oversight and the upstream terms still apply.
