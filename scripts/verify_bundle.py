# ruff: noqa: E501
"""F6.2 — Paridad del paquete exportado con PyTorch sobre 200 imágenes de validación al azar:

  1. logit ONNX vs torch (mismo tensor de entrada, cadena de entrenamiento): máx |Δ| < 1e-4
  2. Platt en numpy (API) vs melanoma.eval.Platt: máx |Δ| < 1e-6
  3. CAM desde features ONNX × cam_weights.npy vs melanoma.explain.cam_from_features sobre
     features de torch: correlación > 0.999 (los mapas nulos en ambos cuentan como iguales)
  4. decisión en τ95 idéntica en las 200
  5. cadena completa de la API (bytes JPEG de 512 px → PIL → cv2 → ONNX → Platt) vs cadena de
     entrenamiento (albumentations → torch): máx |Δ logit| < 1e-3 y decisiones idénticas
  6. lo mismo desde 100 ORIGINALES de data/raw (hasta 6,000 px) de validación: la API (etapa 1
     de F1 en memoria + etapa 2) contra resize_one de F1 a disco + cadena de entrenamiento.
     Se reporta aparte de la prueba a 512 px.

Uso: uv run python scripts/verify_bundle.py  → reports/f6_parity.json
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from services.api.bundle import load_bundle
from services.api.cam import cam_from_features as api_cam
from services.api.inference import Predictor

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.data.dataset import ImageDataset
from melanoma.data.resize import resize_one
from melanoma.data.transforms import build_transforms
from melanoma.eval import Platt
from melanoma.eval.predict import load_checkpoint, sha256_file, split_records
from melanoma.explain import cam_from_features
from melanoma.explain.features import extract_features
from melanoma.train.run import git_sha

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="f6", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    bundle = load_bundle(ROOT / cfg.f6.bundle_dir)
    ckpt = ROOT / cfg.f6.checkpoint
    sha = sha256_file(ckpt)
    if bundle.manifest["checkpoint"]["sha256"] != sha:
        raise SystemExit("el paquete no corresponde al checkpoint local")
    lit, data_config = load_checkpoint(
        ckpt, cfg.model.backbone, cfg.model.num_classes, cfg.model.dropout
    )
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    recs = split_records(
        read_manifest(paths.manifest_path), read_split_ids(paths.splits_dir, VAL_SPLIT)
    )
    rng = np.random.default_rng(int(cfg.f6.seed))
    idx = np.sort(rng.choice(len(recs), size=int(cfg.f6.n_parity), replace=False))
    sample = recs.iloc[idx].reset_index(drop=True)
    transform = build_transforms(
        data_config, cfg.data.image_size, None, train=False, val_resize=cfg.data.val_resize
    )
    ds = ImageDataset(sample, paths.images_dir, transform)
    predictor = Predictor(bundle, int(cfg.f6.intra_op_threads))
    platt = Platt(**json.loads((ROOT / cfg.f6.calibration_json).read_text())["platt"])
    tau = predictor.threshold
    tol = cfg.f6.tolerances

    d_logit, d_platt, cam_corr, cam_both_zero = [], [], [], 0
    dec_torch, dec_onnx, d_chain, dec_chain = [], [], [], []
    for i in range(len(ds)):
        x, _ = ds[i]
        xb = x[None]
        feats_t, logit_t = extract_features(lit, xb)
        logit_t = float(logit_t[0])
        # 1–4: mismo tensor
        logit_o, feats_o = predictor.session.run(
            predictor.output_names, {predictor.input_name: xb.numpy()}
        )
        logit_o = float(logit_o[0, 0])
        d_logit.append(abs(logit_o - logit_t))
        p_ref = float(platt.apply(np.array([logit_o]))[0])
        p_api = predictor.calibrate(logit_o)
        d_platt.append(abs(p_api - p_ref))
        cam_t = cam_from_features(feats_t[0], bundle.cam_weights)
        cam_a = api_cam(feats_o[0], bundle.cam_weights)
        if cam_t.max() == 0 and cam_a.max() == 0:
            cam_both_zero += 1
        else:
            cam_corr.append(float(np.corrcoef(cam_t.ravel(), cam_a.ravel())[0, 1]))
        dec_torch.append(bool(platt.apply(np.array([logit_t]))[0] >= tau))
        dec_onnx.append(bool(p_api >= tau))
        # 5: cadena completa desde los bytes del JPEG
        data = (paths.images_dir / f"{sample['image_id'][i]}.jpg").read_bytes()
        pred = predictor.predict_bytes(data)
        d_chain.append(abs(pred.logit - logit_t))
        dec_chain.append(pred.refer)

    # 6: originales de data/raw → API (dos etapas) vs resize_one de F1 + cadena de entrenamiento
    raw_dir = ROOT / cfg.f6.raw_images_dir
    n_orig = int(cfg.f6.n_parity_originals)
    orig_idx = np.sort(rng.choice(len(recs), size=n_orig, replace=False))
    originals = recs.iloc[orig_idx].reset_index(drop=True)
    s1 = bundle.preprocess["stage1"]
    d_orig, dec_orig_ref, dec_orig_api, orig_sides, orig_bytes = [], [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(len(originals)):
            iid = str(originals["image_id"][i])
            src = raw_dir / f"{iid}.jpg"
            dst = Path(tmp) / f"{iid}.jpg"
            info = resize_one(src, dst, int(s1["long_side"]), int(s1["jpeg_quality"]))
            ref_ds = ImageDataset(originals.iloc[[i]].reset_index(drop=True), tmp, transform)
            _, logit_ref = extract_features(lit, ref_ds[0][0][None])
            logit_ref = float(logit_ref[0])
            data = src.read_bytes()
            pred = predictor.predict_bytes(data)
            d_orig.append(abs(pred.logit - logit_ref))
            dec_orig_ref.append(bool(platt.apply(np.array([logit_ref]))[0] >= tau))
            dec_orig_api.append(pred.refer)
            orig_sides.append(max(pred.input_info["width"], pred.input_info["height"]))
            orig_bytes.append(len(data))
            del info
    d_orig = np.array(d_orig)

    d_logit, d_platt, d_chain = np.array(d_logit), np.array(d_platt), np.array(d_chain)
    corr = np.array(cam_corr)
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "model_version": bundle.model_version,
        "checkpoint_sha256": sha,
        "split_sha256": hashes,
        "n": int(len(ds)),
        "seed": int(cfg.f6.seed),
        "image_ids": sample["image_id"].tolist(),
        "logit": {
            "max_abs_diff": float(d_logit.max()),
            "mean_abs_diff": float(d_logit.mean()),
            "tolerance": float(tol.logit),
            "pass": bool(d_logit.max() < tol.logit),
        },
        "platt": {
            "max_abs_diff": float(d_platt.max()),
            "tolerance": float(tol.platt),
            "pass": bool(d_platt.max() < tol.platt),
        },
        "cam": {
            "n_correlated": int(corr.size),
            "n_both_zero": int(cam_both_zero),
            "min_corr": float(corr.min()) if corr.size else None,
            "mean_corr": float(corr.mean()) if corr.size else None,
            "tolerance": float(tol.cam_corr),
            "pass": bool((corr > tol.cam_corr).all()) if corr.size else True,
        },
        "decision_tau95": {
            "n_differ": int(sum(a != b for a, b in zip(dec_torch, dec_onnx, strict=True))),
            "n_refer": int(sum(dec_onnx)),
            "pass": dec_torch == dec_onnx,
        },
        "full_chain": {
            "max_abs_diff_logit": float(d_chain.max()),
            "mean_abs_diff_logit": float(d_chain.mean()),
            "p95_abs_diff_logit": float(np.quantile(d_chain, 0.95)),
            "tolerance": float(tol.full_chain_logit),
            "pass": bool(d_chain.max() < tol.full_chain_logit),
            "n_decisions_differ": int(
                sum(a != b for a, b in zip(dec_torch, dec_chain, strict=True))
            ),
            "decisions_identical": dec_torch == dec_chain,
        },
        "full_chain_originals": {
            "n": int(len(d_orig)),
            "image_ids": originals["image_id"].tolist(),
            "source": str(cfg.f6.raw_images_dir),
            "long_side_min": int(min(orig_sides)),
            "long_side_max": int(max(orig_sides)),
            "bytes_max": int(max(orig_bytes)),
            "stage1": s1,
            "max_abs_diff_logit": float(d_orig.max()),
            "mean_abs_diff_logit": float(d_orig.mean()),
            "p95_abs_diff_logit": float(np.quantile(d_orig, 0.95)),
            "tolerance": float(tol.full_chain_logit),
            "pass": bool(d_orig.max() < tol.full_chain_logit),
            "n_decisions_differ": int(
                sum(a != b for a, b in zip(dec_orig_ref, dec_orig_api, strict=True))
            ),
            "decisions_identical": dec_orig_ref == dec_orig_api,
        },
        "minutes": (time.perf_counter() - t0) / 60,
    }
    result["all_pass"] = all(
        result[k]["pass"]
        for k in ("logit", "platt", "cam", "decision_tau95", "full_chain", "full_chain_originals")
    )
    out = ROOT / cfg.f6.parity_json
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"logit máx |Δ| {d_logit.max():.2e} (< {tol.logit}) · Platt máx |Δ| {d_platt.max():.2e} · CAM corr mín {result['cam']['min_corr']} (ambas cero: {cam_both_zero}) · decisiones distintas: {result['decision_tau95']['n_differ']}"
    )
    print(
        f"cadena completa: logit máx |Δ| {d_chain.max():.2e} media {d_chain.mean():.2e} p95 {result['full_chain']['p95_abs_diff_logit']:.2e} (< {tol.full_chain_logit}) · decisiones distintas: {result['full_chain']['n_decisions_differ']}"
    )
    fo = result["full_chain_originals"]
    print(
        f"originales ({fo['n']}, lado largo {fo['long_side_min']}–{fo['long_side_max']}): logit máx |Δ| {fo['max_abs_diff_logit']:.2e} media {fo['mean_abs_diff_logit']:.2e} p95 {fo['p95_abs_diff_logit']:.2e} (< {tol.full_chain_logit}) · decisiones distintas: {fo['n_decisions_differ']}"
    )
    print(f"all_pass={result['all_pass']} → {out.relative_to(ROOT)} · {result['minutes']:.1f} min")


if __name__ == "__main__":
    main()
