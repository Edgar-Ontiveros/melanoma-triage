# ruff: noqa: E501
"""F5.3 — Máscara automática de la lesión, filtro de calidad y solapamiento del CAM con la
lesión, desglosado por grupo en τ95 (VP / FN / FP / VN) y comparado con el azar (la fracción
de área que ocupa la máscara).

Entradas: reports/f5_cams_val.npz (scripts/f5_cam.py), reports/predictions/f4_val.csv
(probabilidad Platt) y reports/f4_calibration.json (τ95). Salidas:
reports/predictions/f5_overlap_val.csv (una fila por imagen), reports/f5_overlap.json y la
figura reports/figures/f5_mask_examples.png con máscaras buenas y malas.
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

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.eval import Platt
from melanoma.eval.predict import split_records
from melanoma.explain import energy_fraction, upsample
from melanoma.explain.features import crop_transform
from melanoma.explain.masks import lesion_mask
from melanoma.explain.stats import bootstrap_cluster, describe
from melanoma.models import build_model
from melanoma.train.run import git_sha

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "TP": "verdaderos positivos",
    "FN": "falsos negativos",
    "FP": "falsos positivos",
    "TN": "verdaderos negativos",
}


def load_crop(images_dir: Path, image_id: str, crop_tf) -> np.ndarray:
    with Image.open(images_dir / f"{image_id}.jpg") as img:
        arr = np.asarray(img.convert("RGB"))
    return crop_tf(image=arr)["image"]


def confusion_group(target: int, prob: float, tau: float) -> str:
    referred = prob >= tau
    if target == 1:
        return "TP" if referred else "FN"
    return "FP" if referred else "TN"


def mask_examples_figure(
    rows: pd.DataFrame, images_dir: Path, crop_tf, path: Path, seed: int
) -> list[str]:
    """Seis máscaras válidas y seis descartadas (mezclando razones), con el contorno dibujado."""
    rng = np.random.default_rng(seed)
    good = rows[rows["mask_valid"]]
    bad = rows[~rows["mask_valid"]]
    picks = list(
        good.sample(min(6, len(good)), random_state=int(rng.integers(1 << 31)))["image_id"]
    )
    bad_picks: list[str] = []
    for reason in ("too_small", "too_large", "no_component"):
        sub = bad[bad["mask_reason"] == reason]
        bad_picks += list(
            sub.sample(min(2, len(sub)), random_state=int(rng.integers(1 << 31)))["image_id"]
        )
    rest = bad[~bad["image_id"].isin(bad_picks)]
    if len(bad_picks) < 6 and len(rest):
        bad_picks += list(
            rest.sample(
                min(6 - len(bad_picks), len(rest)), random_state=int(rng.integers(1 << 31))
            )["image_id"]
        )
    ids = picks + bad_picks
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    info = rows.set_index("image_id")
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= len(ids):
            continue
        iid = ids[k]
        crop = load_crop(images_dir, iid, crop_tf)
        mask, _ = lesion_mask(crop)
        canvas = crop.copy()
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        ax.imshow(canvas)
        r = info.loc[iid]
        tag = "válida" if r["mask_valid"] else r["mask_reason"]
        ax.set_title(
            f"{iid}\n{tag} · área {r['mask_area_frac']:.2f} · borde {r['mask_border_frac']:.2f}",
            fontsize=7,
        )
    axes[0, 0].text(
        -0.1, 1.32, "Máscaras válidas", transform=axes[0, 0].transAxes, fontsize=10, weight="bold"
    )
    axes[1, 0].text(
        -0.1,
        1.32,
        "Máscaras descartadas",
        transform=axes[1, 0].transAxes,
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return ids


@hydra.main(config_path="../configs", config_name="f5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    recs = split_records(
        read_manifest(paths.manifest_path), read_split_ids(paths.splits_dir, VAL_SPLIT)
    )
    cams = np.load(ROOT / cfg.f5.cams_npz)
    if not np.array_equal(cams["image_id"], recs["image_id"].to_numpy().astype(str)):
        raise SystemExit(
            "el orden de reports/f5_cams_val.npz no coincide con el split de validación"
        )
    preds = pd.read_csv(ROOT / cfg.f5.val_predictions).set_index("image_id")
    cal = json.loads((ROOT / cfg.f5.calibration_json).read_text())
    tau = float(cal["thresholds"]["tau_95"]["tau"])
    # La probabilidad se recalcula desde el logit con la Platt congelada: τ95 coincide con la
    # probabilidad exacta de un melanoma y el redondeo del CSV lo dejaría por debajo del corte.
    platt = Platt(**cal["platt"])
    preds["prob_platt"] = platt.apply(preds["logit"].to_numpy())
    _, data_config = build_model(
        cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    crop_tf = crop_transform(data_config, cfg.data.image_size, cfg.data.val_resize)
    size = int(cfg.data.image_size)

    rows = []
    for i, r in recs.iterrows():
        iid = str(r["image_id"])
        crop = load_crop(paths.images_dir, iid, crop_tf)
        mask, info = lesion_mask(crop)
        prob = float(preds.loc[iid, "prob_platt"])
        row = {
            "image_id": iid,
            "patient_id": r["patient_id"],
            "target": int(r["target"]),
            "prob_platt": prob,
            "group": confusion_group(int(r["target"]), prob, tau),
            "mask_valid": info.valid,
            "mask_reason": info.reason,
            "mask_area_frac": info.area_frac,
            "mask_border_frac": info.border_frac,
            "overlap": np.nan,
            "chance": np.nan,
        }
        if info.valid:
            cam_up = upsample(cams["cam"][i], size)
            row["overlap"] = energy_fraction(cam_up, mask)
            row["chance"] = float(mask.mean())
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(recs)} máscaras en {(time.perf_counter() - t0) / 60:.1f} min")
    df = pd.DataFrame(rows)
    df["overlap_minus_chance"] = df["overlap"] - df["chance"]
    out_csv = ROOT / cfg.f5.overlap_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    valid = df[df["mask_valid"]]
    n_res, seed = int(cfg.f5.bootstrap_resamples), int(cfg.f5.seed)
    by_group = {}
    for g in GROUPS:
        sub = valid[valid["group"] == g]
        allsub = df[df["group"] == g]
        by_group[g] = {
            "label": GROUPS[g],
            "n_total": int(len(allsub)),
            "n_valid_mask": int(len(sub)),
            "overlap": describe(sub["overlap"].to_numpy()),
            "chance": describe(sub["chance"].to_numpy()),
            "frac_above_chance": float((sub["overlap"] > sub["chance"]).mean())
            if len(sub)
            else None,
            "overlap_minus_chance_ci": bootstrap_cluster(
                sub["overlap_minus_chance"].to_numpy(), sub["patient_id"].to_numpy(), n_res, seed
            )
            if len(sub)
            else None,
            "overlap_ci": bootstrap_cluster(
                sub["overlap"].to_numpy(), sub["patient_id"].to_numpy(), n_res, seed
            )
            if len(sub)
            else None,
        }
    reasons = df["mask_reason"].value_counts().to_dict()
    fig_ids = mask_examples_figure(
        df, paths.images_dir, crop_tf, ROOT / cfg.f5.figures_dir / "f5_mask_examples.png", seed
    )
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "split_sha256": hashes,
        "tau_95": tau,
        "n_images": int(len(df)),
        "n_valid_mask": int(df["mask_valid"].sum()),
        "mask_reasons": {k: int(v) for k, v in reasons.items()},
        "mask_rules": {"min_area_frac": 0.03, "max_area_frac": 0.70, "max_border_frac": 0.30},
        "confusion_at_tau95": df["group"].value_counts().to_dict(),
        "overall": {
            "overlap": describe(valid["overlap"].to_numpy()),
            "chance": describe(valid["chance"].to_numpy()),
            "frac_above_chance": float((valid["overlap"] > valid["chance"]).mean()),
            "overlap_minus_chance_ci": bootstrap_cluster(
                valid["overlap_minus_chance"].to_numpy(),
                valid["patient_id"].to_numpy(),
                n_res,
                seed,
            ),
        },
        "by_group": by_group,
        "mask_examples": fig_ids,
        "minutes": (time.perf_counter() - t0) / 60,
    }
    out = ROOT / cfg.f5.overlap_json
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"máscaras válidas: {result['n_valid_mask']}/{result['n_images']} · razones {reasons}")
    for g, d in by_group.items():
        o, c = d["overlap"], d["chance"]
        if o.get("n"):
            print(
                f"  {g}: n={o['n']:4d}  solapamiento media {o['mean']:.3f} mediana {o['median']:.3f} · azar {c['mean']:.3f} · > azar {d['frac_above_chance']:.0%}"
            )
    print(f"→ {out.relative_to(ROOT)} · {result['minutes']:.1f} min")


if __name__ == "__main__":
    main()
