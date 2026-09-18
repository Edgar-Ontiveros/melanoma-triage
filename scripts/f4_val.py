# ruff: noqa: E501
"""F4.0–F4.2 — Predicciones sobre validación con el checkpoint final, calibración (Platt y
temperatura), ECE/Brier antes y después, y umbrales τ95/τ90 por bootstrap por paciente.

Solo validación. Salidas: reports/predictions/f4_val.csv, reports/f4_calibration.json y
figuras. Los valores que van al protocolo (docs/f4_protocol.yaml) se imprimen al final; el
protocolo se escribe y commitea aparte, antes de que exista scripts/evaluate_test.py.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import hydra
import matplotlib
from omegaconf import DictConfig

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.eval import (
    bootstrap_patient,
    bootstrap_threshold,
    calibration_summary,
    compute_metrics,
    fit_platt,
    fit_temperature,
)
from melanoma.eval.predict import load_checkpoint, predict_frame, sha256_file, split_records
from melanoma.train.run import git_sha

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def reliability_figure(curves: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(4.5 * len(curves), 4.5), squeeze=False)
    for ax, (label, summ) in zip(axes[0], curves.items(), strict=True):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for strat, marker in (("ece_uniform", "o"), ("ece_quantile", "s")):
            rows = [b for b in summ[strat]["bins"] if b["n"]]
            ax.plot(
                [b["confidence"] for b in rows],
                [b["observed"] for b in rows],
                marker=marker,
                label=f"{strat.split('_')[1]} (ECE {summ[strat]['ece']:.3f})",
            )
            for b in rows:
                ax.annotate(
                    str(b["n"]),
                    (b["confidence"], b["observed"]),
                    fontsize=6,
                    textcoords="offset points",
                    xytext=(3, 3),
                )
        ax.set_title(f"{label} · Brier {summ['brier']:.4f}")
        ax.set_xlabel("probabilidad predicha")
        ax.set_ylabel("fracción observada")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="f4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    ckpt = ROOT / cfg.f4.checkpoint
    sha = sha256_file(ckpt)
    expected = json.loads((ROOT / cfg.f4.checkpoint_manifest).read_text())["sha256"]
    if sha != expected:
        raise SystemExit(f"SHA256 del checkpoint {sha} ≠ {expected} (reports/f3_final_model.json)")
    print(f"checkpoint OK: {ckpt.name} sha256={sha[:16]}…")

    lit, data_config = load_checkpoint(
        ckpt, cfg.model.backbone, cfg.model.num_classes, cfg.model.dropout
    )
    manifest = read_manifest(paths.manifest_path)
    recs = split_records(manifest, read_split_ids(paths.splits_dir, VAL_SPLIT))
    preds = predict_frame(
        lit,
        data_config,
        recs,
        paths.images_dir,
        cfg.data.image_size,
        cfg.data.val_resize,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"{len(preds)} predicciones de validación en {(time.perf_counter() - t0) / 60:.1f} min")

    z, y, pid = (
        preds["logit"].to_numpy(),
        preds["target"].to_numpy(),
        preds["patient_id"].to_numpy(),
    )
    platt, temp = fit_platt(z, y), fit_temperature(z, y)
    preds["prob_platt"] = platt.apply(z)
    preds["prob_temp"] = temp.apply(z)
    out_pred = ROOT / cfg.f4.val_predictions
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_pred, index=False)

    ev = cfg.train.eval
    calib = {
        "raw": calibration_summary(preds["prob_raw"].to_numpy(), y),
        "platt": calibration_summary(preds["prob_platt"].to_numpy(), y),
        "temperature": calibration_summary(preds["prob_temp"].to_numpy(), y),
    }
    reliability_figure(
        {"cruda": calib["raw"], "Platt": calib["platt"], "temperatura": calib["temperature"]},
        ROOT / cfg.f4.figures_dir / "f4_val_reliability.png",
    )
    pos_weight = float(cfg.f4.pos_weight_used)
    thresholds = {
        "tau_95": bootstrap_threshold(
            preds["prob_platt"].to_numpy(), y, pid, 0.95, ev.bootstrap_resamples, ev.bootstrap_seed
        ),
        "tau_90": bootstrap_threshold(
            preds["prob_platt"].to_numpy(), y, pid, 0.90, ev.bootstrap_resamples, ev.bootstrap_seed
        ),
    }
    metrics_raw = compute_metrics(
        preds["prob_raw"].to_numpy(), y, pid, fixed_levels=list(ev.fixed_levels)
    )
    metrics_platt = compute_metrics(
        preds["prob_platt"].to_numpy(), y, pid, fixed_levels=list(ev.fixed_levels)
    )
    ci = bootstrap_patient(
        preds["prob_platt"].to_numpy(), y, pid, ev.bootstrap_resamples, ev.bootstrap_seed
    )
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": git_sha(ROOT),
        "checkpoint": {"file": ckpt.name, "sha256": sha},
        "split_sha256": hashes,
        "n_images": int(len(preds)),
        "n_positives": int(y.sum()),
        "n_patients": int(len(set(pid))),
        "platt": platt.as_dict(),
        "temperature": temp.as_dict(),
        "pos_weight_used": pos_weight,
        "expected_b": -math.log(pos_weight),
        "calibration": calib,
        "thresholds": thresholds,
        "metrics_raw": {k: v for k, v in metrics_raw.items() if k != "curves"},
        "metrics_platt": {k: v for k, v in metrics_platt.items() if k != "curves"},
        "ci_platt": ci,
        "ranking_preserved": bool(abs(metrics_raw["auroc"] - metrics_platt["auroc"]) < 1e-9),
    }
    out = ROOT / cfg.f4.calibration_json
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("platt", "temperature", "expected_b", "ranking_preserved")},
            indent=2,
        )
    )
    for name, c in calib.items():
        print(
            f"  {name:12s} Brier {c['brier']:.4f}  ECE uniforme {c['ece_uniform']['ece']:.4f}  ECE cuantiles {c['ece_quantile']['ece']:.4f}"
        )
    for k, t in thresholds.items():
        o = t["on_full_set"]
        print(
            f"  {k}: τ={t['tau']:.4f} [p5 {t['p05']:.4f}, p95 {t['p95']:.4f}] corte único {t['single_cut']:.4f} → sens {o['sensitivity']:.3f} spec {o['specificity']:.3f} VPP {o['ppv']:.3f} VPN {o['npv']:.4f}"
        )
    print(f"→ {out.relative_to(ROOT)}, {out_pred.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
