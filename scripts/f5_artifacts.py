# ruff: noqa: E501
"""F5.4 — Experimento pareado de artefactos: a cada imagen se le inyectan cinco perturbaciones
(regla, tinta, vello, viñeteado y ruido gaussiano como control) y se mide, con probabilidad
calibrada (Platt de F4), Δ = P(con artefacto) − P(original), la fracción de benignos que
cruzan τ95 por el artefacto y si el CAM se mueve hacia la zona del artefacto.

Muestra: los 88 melanomas de validación y 1,000 benignos al azar (semilla de configs/f5.yaml).
Salidas: reports/predictions/f5_artifacts_val.csv (una fila por imagen y condición),
reports/f5_artifacts.json, reports/figures/f5_artifacts_delta.png y
reports/figures/f5_artifact_panel.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.eval import Platt
from melanoma.eval.predict import load_checkpoint, sha256_file, split_records
from melanoma.explain import cam_from_features, energy_fraction, upsample
from melanoma.explain.features import (
    crop_transform,
    extract_features,
    linear_weight,
    tensor_transform,
    to_batch,
)
from melanoma.explain.masks import lesion_mask
from melanoma.explain.perturb import LABELS_ES, PERTURBATIONS, pixel_change
from melanoma.explain.stats import bootstrap_cluster, describe
from melanoma.train.run import git_sha

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["original", *PERTURBATIONS]


def load_rgb(images_dir: Path, image_id: str) -> np.ndarray:
    with Image.open(images_dir / f"{image_id}.jpg") as img:
        return np.asarray(img.convert("RGB"))


def sample_records(recs: pd.DataFrame, n_benign: int, seed: int) -> pd.DataFrame:
    mel = recs[recs["target"] == 1]
    ben = recs[recs["target"] == 0]
    ben = ben.sample(min(n_benign, len(ben)), random_state=seed) if n_benign < len(ben) else ben
    return pd.concat([mel, ben]).sort_values("image_id").reset_index(drop=True)


def delta_figure(df: pd.DataFrame, tau: float, path: Path) -> None:
    """Distribución de Δ por perturbación, benignos y melanomas por separado."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (target, title) in zip(axes, ((0, "benignos"), (1, "melanomas")), strict=True):
        sub = df[(df["target"] == target) & (df["condition"] != "original")]
        data = [sub.loc[sub["condition"] == c, "delta"].to_numpy() for c in PERTURBATIONS]
        ax.axhline(0, color="gray", lw=0.8)
        ax.boxplot(
            data,
            tick_labels=[LABELS_ES[c] for c in PERTURBATIONS],
            showfliers=True,
            flierprops={"markersize": 2, "alpha": 0.4},
        )
        ax.set_title(f"Δ probabilidad calibrada · {title} (n={sub['image_id'].nunique()})")
        ax.set_ylabel("P(con artefacto) − P(original)")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def panel_figure(
    image_id: str,
    crops: dict[str, np.ndarray],
    cams: dict[str, np.ndarray],
    probs: dict[str, float],
    size: int,
    path: Path,
) -> None:
    """Una imagen benigna con las cinco perturbaciones (arriba) y su CAM (abajo)."""
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(2.6 * len(CONDITIONS), 5.4))
    for j, c in enumerate(CONDITIONS):
        label = "original" if c == "original" else LABELS_ES[c]
        axes[0, j].imshow(crops[c])
        axes[0, j].set_title(f"{label}\nP = {probs[c]:.4f}", fontsize=8)
        axes[1, j].imshow(crops[c])
        axes[1, j].imshow(upsample(cams[c], size), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"{image_id} (benigna) · CAM 7×7 interpolado a {size} px", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="f5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    ckpt = ROOT / cfg.f5.checkpoint
    sha = sha256_file(ckpt)
    expected = json.loads((ROOT / cfg.f5.checkpoint_manifest).read_text())["sha256"]
    if sha != expected:
        raise SystemExit(f"SHA256 del checkpoint {sha} ≠ {expected}")
    lit, data_config = load_checkpoint(
        ckpt, cfg.model.backbone, cfg.model.num_classes, cfg.model.dropout
    )
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    recs = split_records(
        read_manifest(paths.manifest_path), read_split_ids(paths.splits_dir, VAL_SPLIT)
    )
    cal = json.loads((ROOT / cfg.f5.calibration_json).read_text())
    platt = Platt(**cal["platt"])
    tau = float(cal["thresholds"]["tau_95"]["tau"])
    f4 = pd.read_csv(ROOT / cfg.f5.val_predictions).set_index("image_id")
    seed = int(cfg.f5.seed)
    sample = sample_records(recs, int(cfg.f5.n_benign_sample), seed)
    size = int(cfg.data.image_size)
    crop_tf = crop_transform(data_config, size, cfg.data.val_resize)
    tensor_tf = tensor_transform(data_config)
    w = linear_weight(lit)
    print(
        f"checkpoint OK · muestra: {int((sample['target'] == 1).sum())} melanomas + {int((sample['target'] == 0).sum())} benignos · τ95 = {tau:.4f}"
    )

    rows = []
    panel = None
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for k, r in sample.iterrows():
        iid = str(r["image_id"])
        rng = np.random.default_rng([seed, k])
        crop = crop_tf(image=load_rgb(paths.images_dir, iid))["image"]
        mask, info = lesion_mask(crop)
        crops = {"original": crop}
        regions = {}
        for name, fn in PERTURBATIONS.items():
            crops[name], regions[name] = fn(crop, rng, mask if info.valid else None)
        feats, logits = extract_features(lit, to_batch([crops[c] for c in CONDITIONS], tensor_tf))
        cams = cam_from_features(feats, w)
        probs = platt.apply(logits)
        cam_up = {c: upsample(cams[j], size) for j, c in enumerate(CONDITIONS)}
        for j, c in enumerate(CONDITIONS):
            region = regions.get(c)
            rows.append(
                {
                    "image_id": iid,
                    "patient_id": r["patient_id"],
                    "target": int(r["target"]),
                    "condition": c,
                    "logit": float(logits[j]),
                    "prob_platt": float(probs[j]),
                    "delta": float(probs[j] - probs[0]),
                    "referred": bool(probs[j] >= tau),
                    "referred_original": bool(probs[0] >= tau),
                    "pixel_change": pixel_change(crop, crops[c]) if c != "original" else 0.0,
                    "region_area_frac": float(region.mean()) if region is not None else np.nan,
                    "cam_in_region": energy_fraction(cam_up[c], region)
                    if region is not None
                    else np.nan,
                    "cam_in_region_original": energy_fraction(cam_up["original"], region)
                    if region is not None
                    else np.nan,
                    "mask_valid": info.valid,
                }
            )
        if panel is None and r["target"] == 0 and info.valid:
            panel = (
                iid,
                crops,
                {c: cams[j] for j, c in enumerate(CONDITIONS)},
                {c: float(probs[j]) for j, c in enumerate(CONDITIONS)},
            )
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(sample)} imágenes en {(time.perf_counter() - t0) / 60:.1f} min")
    df = pd.DataFrame(rows)
    df["cam_shift_to_region"] = df["cam_in_region"] - df["cam_in_region_original"]

    orig = df[df["condition"] == "original"].set_index("image_id")
    diff = np.abs(orig["logit"] - f4.loc[orig.index, "logit"]).max()
    if diff > 1e-3:
        raise SystemExit(
            f"los logits originales difieren de F4 (máx {diff:.2e}): la geometría no es la misma"
        )
    out_csv = ROOT / cfg.f5.artifacts_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_res = int(cfg.f5.bootstrap_resamples)
    results: dict[str, dict] = {}
    for c in PERTURBATIONS:
        results[c] = {"label": LABELS_ES[c]}
        for target, tag in ((0, "benign"), (1, "melanoma")):
            sub = df[(df["condition"] == c) & (df["target"] == target)]
            pid = sub["patient_id"].to_numpy()
            d = sub["delta"].to_numpy()
            entry = {
                "n": int(len(sub)),
                "delta": describe(d),
                "delta_mean_ci": bootstrap_cluster(d, pid, n_res, seed),
                "delta_median_ci": bootstrap_cluster(d, pid, n_res, seed, statistic=np.median),
                "frac_delta_positive": float((d > 0).mean()),
                "frac_delta_positive_ci": bootstrap_cluster(
                    (d > 0).astype(float), pid, n_res, seed
                ),
                "pixel_change_mean": float(sub["pixel_change"].mean()),
                "region_area_frac_mean": float(sub["region_area_frac"].mean()),
                "cam_in_region_original_mean": float(sub["cam_in_region_original"].mean()),
                "cam_in_region_perturbed_mean": float(sub["cam_in_region"].mean()),
                "cam_shift_ci": bootstrap_cluster(
                    sub["cam_shift_to_region"].to_numpy(), pid, n_res, seed
                ),
            }
            if target == 0:
                below = sub[~sub["referred_original"]]
                cross = (below["referred"]).astype(float).to_numpy()
                entry["n_not_referred_original"] = int(len(below))
                entry["n_cross_to_referred"] = int(cross.sum())
                entry["frac_cross_to_referred_ci"] = (
                    bootstrap_cluster(cross, below["patient_id"].to_numpy(), n_res, seed)
                    if len(below)
                    else None
                )
                entry["frac_referred_original"] = float(sub["referred_original"].mean())
                entry["frac_referred_perturbed"] = float(sub["referred"].mean())
            else:
                above = sub[sub["referred_original"]]
                lost = (~above["referred"]).astype(float).to_numpy()
                entry["n_referred_original"] = int(len(above))
                entry["n_lose_referral"] = int(lost.sum())
                entry["frac_lose_referral_ci"] = (
                    bootstrap_cluster(lost, above["patient_id"].to_numpy(), n_res, seed)
                    if len(above)
                    else None
                )
            results[c][tag] = entry

    fig_dir = ROOT / cfg.f5.figures_dir
    delta_figure(df, tau, fig_dir / "f5_artifacts_delta.png")
    if panel is not None:
        panel_figure(
            panel[0], panel[1], panel[2], panel[3], size, fig_dir / "f5_artifact_panel.png"
        )
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "checkpoint": {"file": ckpt.name, "sha256": sha},
        "split_sha256": hashes,
        "tau_95": tau,
        "platt": cal["platt"],
        "seed": seed,
        "n_melanoma": int((sample["target"] == 1).sum()),
        "n_benign": int((sample["target"] == 0).sum()),
        "n_benign_available": int((recs["target"] == 0).sum()),
        "n_patients": int(sample["patient_id"].nunique()),
        "logit_max_abs_diff_vs_f4": float(diff),
        "conditions": CONDITIONS,
        "bootstrap": {"n_resamples": n_res, "unit": "patient", "seed": seed},
        "results": results,
        "panel_image_id": panel[0] if panel else None,
        "minutes": (time.perf_counter() - t0) / 60,
    }
    out = ROOT / cfg.f5.artifacts_json
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    for c, r in results.items():
        b, m = r["benign"], r["melanoma"]
        print(
            f"  {LABELS_ES[c]:16s} benignos: Δ media {b['delta_mean_ci']['point']:+.4f} [{b['delta_mean_ci']['lo']:+.4f}, {b['delta_mean_ci']['hi']:+.4f}] · Δ>0 {b['frac_delta_positive']:.0%} · cruzan τ95 {b['n_cross_to_referred']}/{b['n_not_referred_original']} · CAM→zona {b['cam_shift_ci']['point']:+.3f}"
            f" | melanomas: Δ media {m['delta_mean_ci']['point']:+.4f} · pierden referencia {m['n_lose_referral']}/{m['n_referred_original']}"
        )
    print(f"→ {out.relative_to(ROOT)} · {result['minutes']:.1f} min")


if __name__ == "__main__":
    main()
