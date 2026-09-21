# ruff: noqa: E501
"""F5.5 — Cuadrícula cualitativa: 12 imágenes de validación (3 VP, 3 FN, 3 FP, 3 VN en τ95)
con imagen original, máscara automática, CAM superpuesto, probabilidad calibrada y
solapamiento. Las imágenes usadas se listan por image_id en reports/f5_figures.json (ISIC 2020
es CC-BY-NC: reproducibles con atribución). El panel de artefactos lo genera f5_artifacts.py.

Selección: al azar con semilla dentro de cada grupo, entre las imágenes con máscara válida
(para poder mostrar el solapamiento); los FN son los que haya (4 en τ95 sobre validación).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from melanoma.data import resolve_paths
from melanoma.explain import upsample
from melanoma.explain.features import crop_transform
from melanoma.explain.masks import lesion_mask
from melanoma.models import build_model
from melanoma.train.run import git_sha

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUP_ORDER = ["TP", "FN", "FP", "TN"]
GROUP_ES = {"TP": "VP", "FN": "FN", "FP": "FP", "TN": "VN"}


def pick(df: pd.DataFrame, group: str, n: int, seed: int) -> list[str]:
    sub = df[(df["group"] == group) & df["mask_valid"]]
    if len(sub) <= n:
        return sub["image_id"].tolist()
    return sub.sample(n, random_state=seed)["image_id"].tolist()


@hydra.main(config_path="../configs", config_name="f5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    paths = resolve_paths(cfg.data, root=ROOT)
    df = pd.read_csv(ROOT / cfg.f5.overlap_csv)
    cams = np.load(ROOT / cfg.f5.cams_npz)
    cam_by_id = dict(zip(cams["image_id"].tolist(), cams["cam"], strict=True))
    _, data_config = build_model(
        cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    size = int(cfg.data.image_size)
    crop_tf = crop_transform(data_config, size, cfg.data.val_resize)
    seed = int(cfg.f5.seed)
    chosen = {g: pick(df, g, 3, seed) for g in GROUP_ORDER}
    ids = [i for g in GROUP_ORDER for i in chosen[g]]
    info = df.set_index("image_id")

    fig, axes = plt.subplots(len(ids), 3, figsize=(8.4, 2.6 * len(ids)))
    for row, iid in enumerate(ids):
        with Image.open(paths.images_dir / f"{iid}.jpg") as img:
            crop = crop_tf(image=np.asarray(img.convert("RGB")))["image"]
        mask, _ = lesion_mask(crop)
        r = info.loc[iid]
        canvas = crop.copy()
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        axes[row, 0].imshow(crop)
        axes[row, 0].set_title(
            f"{iid} · {GROUP_ES[r['group']]} · target {int(r['target'])}", fontsize=8
        )
        axes[row, 1].imshow(canvas)
        axes[row, 1].set_title(f"máscara automática · área {r['mask_area_frac']:.2f}", fontsize=8)
        axes[row, 2].imshow(crop)
        axes[row, 2].imshow(upsample(cam_by_id[iid], size), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title(
            f"CAM · P = {r['prob_platt']:.4f} · solapamiento {r['overlap']:.2f} (azar {r['chance']:.2f})",
            fontsize=8,
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Validación · grupos en τ95 · CAM 7×7 interpolado a 224 px", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_png = ROOT / cfg.f5.figures_dir / "f5_qualitative_grid.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "seed": seed,
        "grid": {g: chosen[g] for g in GROUP_ORDER},
        "rows": [
            {
                "image_id": iid,
                "group": info.loc[iid, "group"],
                "target": int(info.loc[iid, "target"]),
                "prob_platt": float(info.loc[iid, "prob_platt"]),
                "overlap": float(info.loc[iid, "overlap"]),
                "chance": float(info.loc[iid, "chance"]),
            }
            for iid in ids
        ],
        "figure": str(out_png.relative_to(ROOT)),
        "license": "ISIC 2020 Challenge Dataset, CC-BY-NC 4.0, https://doi.org/10.34970/2020-ds01",
        "minutes": (time.perf_counter() - t0) / 60,
    }
    out = ROOT / "reports" / "f5_figures.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{len(ids)} imágenes: "
        + ", ".join(f"{g}={len(chosen[g])}" for g in GROUP_ORDER)
        + f" → {out_png.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
