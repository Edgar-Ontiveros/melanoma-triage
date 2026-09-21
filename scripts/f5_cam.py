# ruff: noqa: E501
"""F5.1 y F5.2 — CAM de toda la validación con el checkpoint final, verificación de la
equivalencia CAM ≡ Grad-CAM contra pytorch-grad-cam sobre 50 imágenes, y prueba de cordura de
Adebayo et al. (2018) sobre 200 imágenes con pesos reinicializados.

Solo validación. Salidas: reports/f5_cams_val.npz (CAM 7×7 por imagen, fuera de git) y
reports/f5_cam.json (resumen para el reporte).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.data.dataset import ImageDataset
from melanoma.data.transforms import build_transforms
from melanoma.eval.predict import load_checkpoint, sha256_file, split_records
from melanoma.explain import cam_from_features, upsample
from melanoma.explain.features import (
    TARGET_LAYER,
    extract_features,
    gradcam_reference,
    linear_weight,
)
from melanoma.explain.sanity import SCOPES, randomized_copy, spearman_maps, summarize_correlations
from melanoma.train.run import git_sha

ROOT = Path(__file__).resolve().parents[1]


def load_final_model(cfg: DictConfig):
    ckpt = ROOT / cfg.f5.checkpoint
    sha = sha256_file(ckpt)
    expected = json.loads((ROOT / cfg.f5.checkpoint_manifest).read_text())["sha256"]
    if sha != expected:
        raise SystemExit(f"SHA256 del checkpoint {sha} ≠ {expected} ({cfg.f5.checkpoint_manifest})")
    lit, data_config = load_checkpoint(
        ckpt, cfg.model.backbone, cfg.model.num_classes, cfg.model.dropout
    )
    return lit, data_config, {"file": ckpt.name, "sha256": sha}


def val_records(cfg: DictConfig) -> tuple[pd.DataFrame, Path, dict]:
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    manifest = read_manifest(paths.manifest_path)
    recs = split_records(manifest, read_split_ids(paths.splits_dir, VAL_SPLIT))
    return recs, paths.images_dir, hashes


def all_cams(lit, data_config, recs, images_dir, cfg) -> tuple[np.ndarray, np.ndarray]:
    """CAM 7×7 y logit de cada imagen de validación, en el orden de ``recs``."""
    transform = build_transforms(
        data_config, cfg.data.image_size, None, train=False, val_resize=cfg.data.val_resize
    )
    ds = ImageDataset(recs, images_dir, transform)
    loader = DataLoader(
        ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    w = linear_weight(lit)
    cams, logits = [], []
    for x, _ in loader:
        feats, z = extract_features(lit, x)
        cams.append(cam_from_features(feats, w))
        logits.append(z)
    return np.concatenate(cams).astype(np.float32), np.concatenate(logits).astype(np.float64)


def load_batch(ds: ImageDataset, idx: np.ndarray) -> torch.Tensor:
    return torch.stack([ds[int(i)][0] for i in idx])


@hydra.main(config_path="../configs", config_name="f5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    lit, data_config, ckpt_info = load_final_model(cfg)
    recs, images_dir, hashes = val_records(cfg)
    print(f"checkpoint OK: {ckpt_info['file']} · {len(recs)} imágenes de validación")

    # ---- F5.1: CAM de toda la validación y comprobación contra los logits de F4
    npz = ROOT / cfg.f5.cams_npz
    if bool(cfg.f5.reuse_cams) and npz.exists():
        saved = np.load(npz)
        if not np.array_equal(saved["image_id"], recs["image_id"].to_numpy().astype(str)):
            raise SystemExit(f"{npz} no corresponde al split de validación actual")
        cams, logits = saved["cam"], saved["logit"]
        print(f"CAM reutilizados de {npz.relative_to(ROOT)}")
    else:
        cams, logits = all_cams(lit, data_config, recs, images_dir, cfg)
    f4 = pd.read_csv(ROOT / cfg.f5.val_predictions).set_index("image_id")
    diff = np.abs(logits - f4.loc[recs["image_id"], "logit"].to_numpy())
    print(
        f"CAM de {len(cams)} imágenes en {(time.perf_counter() - t0) / 60:.1f} min · |logit − F4| máx {diff.max():.2e}"
    )
    if diff.max() > 1e-3:
        raise SystemExit("los logits no coinciden con reports/predictions/f4_val.csv")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz, image_id=recs["image_id"].to_numpy().astype(str), cam=cams, logit=logits
    )

    # ---- F5.1: equivalencia con pytorch-grad-cam sobre n imágenes al azar
    rng = np.random.default_rng(cfg.f5.seed)
    transform = build_transforms(
        data_config, cfg.data.image_size, None, train=False, val_resize=cfg.data.val_resize
    )
    ds = ImageDataset(recs, images_dir, transform)
    eq_idx = np.sort(rng.choice(len(ds), size=int(cfg.f5.n_equivalence), replace=False))
    corrs = []
    for start in range(0, len(eq_idx), 10):
        idx = eq_idx[start : start + 10]
        x = load_batch(ds, idx)
        ref = gradcam_reference(lit, x)
        for j, i in enumerate(idx):
            ours = upsample(cams[i], cfg.data.image_size)
            if ours.max() == 0 and ref[j].max() == 0:
                # Sin evidencia positiva en ninguna celda: ambos mapas son cero y coinciden,
                # pero la correlación no está definida. Se cuentan aparte.
                corrs.append(np.nan)
                continue
            corrs.append(float(np.corrcoef(ours.ravel(), ref[j].ravel())[0, 1]))
    corrs = np.array(corrs)
    both_zero = int(np.isnan(corrs).sum())
    corrs = corrs[~np.isnan(corrs)]
    equivalence = {
        "n": int(len(corrs)) + both_zero,
        "n_correlated": int(len(corrs)),
        "n_both_zero": both_zero,
        "layer": TARGET_LAYER,
        "map_shape": [int(s) for s in cams.shape[1:]],
        "min_corr": float(corrs.min()),
        "mean_corr": float(corrs.mean()),
        "median_corr": float(np.median(corrs)),
        "threshold": float(cfg.f5.equivalence_min_corr),
        "all_above_threshold": bool((corrs > float(cfg.f5.equivalence_min_corr)).all()),
        "image_ids": recs["image_id"].iloc[eq_idx].tolist(),
    }
    print(
        f"equivalencia ({equivalence['n']} imágenes): correlación mín {corrs.min():.4f} media {corrs.mean():.4f}"
    )
    if not equivalence["all_above_threshold"]:
        raise SystemExit("la CAM en numpy no coincide con pytorch-grad-cam: revisar capa o signo")

    # ---- F5.2: prueba de cordura (reinicialización de pesos)
    sa_idx = np.sort(rng.choice(len(ds), size=int(cfg.f5.n_sanity), replace=False))
    x_all = [
        load_batch(ds, sa_idx[s : s + cfg.data.batch_size])
        for s in range(0, len(sa_idx), cfg.data.batch_size)
    ]
    trained = cams[sa_idx]
    sanity: dict[str, dict] = {}
    per_image: dict[str, np.ndarray] = {}
    for scope in SCOPES:
        rows = []
        for seed in cfg.f5.sanity_seeds:
            rand = randomized_copy(lit, scope, seed=int(seed))
            w_r = linear_weight(rand)
            rcams = np.concatenate(
                [cam_from_features(extract_features(rand, x)[0], w_r) for x in x_all]
            )
            rows.append(np.array([spearman_maps(trained[i], rcams[i]) for i in range(len(sa_idx))]))
        stacked = np.stack(rows)  # [seeds, n]
        per_image[scope] = stacked
        sanity[scope] = {
            "per_seed": [summarize_correlations(r, cfg.f5.sanity_threshold) for r in rows],
            "pooled": summarize_correlations(stacked.ravel(), cfg.f5.sanity_threshold),
            "seeds": [int(s) for s in cfg.f5.sanity_seeds],
        }
        p = sanity[scope]["pooled"]
        print(
            f"cordura {scope:16s}: Spearman media {p['mean']:.3f} mediana {p['median']:.3f} · < {cfg.f5.sanity_threshold}: {p['frac_below_threshold']:.0%} · indefinidas {p['n_undefined']}/{p['n']}"
        )
    np.savez_compressed(
        npz.with_name("f5_sanity_val.npz"),
        idx=sa_idx,
        image_id=recs["image_id"].iloc[sa_idx].to_numpy().astype(str),
        **per_image,
    )

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "checkpoint": ckpt_info,
        "split_sha256": hashes,
        "n_images": int(len(recs)),
        "n_positives": int(recs["target"].sum()),
        "layer": TARGET_LAYER,
        "map_shape": [int(s) for s in cams.shape[1:]],
        "input_size": int(cfg.data.image_size),
        "logit_max_abs_diff_vs_f4": float(diff.max()),
        "cam_all_zero_images": int((cams.reshape(len(cams), -1).max(axis=1) == 0).sum()),
        "equivalence": equivalence,
        "sanity": {
            "n": int(len(sa_idx)),
            "threshold": float(cfg.f5.sanity_threshold),
            "scopes": sanity,
            "image_ids": recs["image_id"].iloc[sa_idx].tolist(),
        },
        "minutes": (time.perf_counter() - t0) / 60,
    }
    out = ROOT / cfg.f5.cam_json
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"→ {out.relative_to(ROOT)}, {npz.relative_to(ROOT)} · {result['minutes']:.1f} min")


if __name__ == "__main__":
    main()
