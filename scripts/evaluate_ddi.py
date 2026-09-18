# ruff: noqa: E501
"""F4.6 — Evaluación externa en DDI (Diverse Dermatology Images), local y en CPU.

DDI no sale de la laptop: este script no tiene nada que ver con el entrenamiento ni con la
nube. Aplica el checkpoint final de F3 con la transformación de validación y la calibración
de ISIC tal cual (protocolo), y reporta dos tareas (maligno vs benigno, melanoma vs resto) por
grupo de tono de piel de Fitzpatrick, con bootstrap a nivel imagen (DDI no tiene id de
paciente). No se adapta nada a DDI: la degradación es el hallazgo.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from melanoma.eval import Platt, bootstrap_patient, calibration_summary
from melanoma.eval.metrics import assert_no_accuracy, auroc_auprc, confusion_at_threshold
from melanoma.eval.predict import load_checkpoint, predict_frame, sha256_file
from melanoma.train.run import git_sha

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {12: "fst_12", 34: "fst_34", 56: "fst_56"}


def task_metrics(p: np.ndarray, y: np.ndarray, tau95: float, n_boot: int, seed: int) -> dict:
    """AUROC/AUPRC con IC a nivel imagen (cada imagen es su propio 'paciente')."""
    ids = np.arange(y.size).astype(str)
    out = {"n": int(y.size), "positives": int(y.sum()), "prevalence": float(y.mean())}
    if 0 < y.sum() < y.size:
        auroc, auprc = auroc_auprc(p, y)
        ci = bootstrap_patient(p, y, ids, n_boot, seed)
        out.update(
            {
                "auroc": auroc,
                "auroc_lo": ci["auroc"]["lo"],
                "auroc_hi": ci["auroc"]["hi"],
                "auprc": auprc,
                "auprc_lo": ci["auprc"]["lo"],
                "auprc_hi": ci["auprc"]["hi"],
            }
        )
    else:
        out.update(
            {
                k: float("nan")
                for k in ("auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi")
            }
        )
    out["at_tau_95"] = confusion_at_threshold(p, y, tau95)
    return out


@hydra.main(config_path="../configs", config_name="f4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    protocol = OmegaConf.load(ROOT / cfg.f4.protocol)
    ckpt = ROOT / "data" / "models" / str(protocol.model.checkpoint_file)
    sha = sha256_file(ckpt)
    assert sha == str(protocol.model.checkpoint_sha256), "SHA256 del checkpoint ≠ protocolo"
    meta = pd.read_csv(ROOT / cfg.f4.ddi.metadata)
    meta["image_id"] = meta["DDI_file"].str.replace(".png", "", regex=False)
    meta["patient_id"] = meta["image_id"]  # sin id de paciente: cada imagen es independiente
    meta["target"] = meta["malignant"].astype(str).str.lower().eq("true").astype(int)
    meta["melanoma"] = meta["disease"].str.contains("melanoma", case=False).astype(int)
    meta["group"] = meta["skin_tone"].map(GROUPS)
    assert meta["group"].notna().all(), meta["skin_tone"].unique()
    lit, data_config = load_checkpoint(
        ckpt, str(protocol.model.backbone), cfg.model.num_classes, cfg.model.dropout
    )
    preds = predict_frame(
        lit,
        data_config,
        meta[["image_id", "target", "patient_id", "DDI_file", "melanoma", "group", "disease"]],
        ROOT / cfg.f4.ddi.images_dir,
        int(protocol.model.image_size),
        str(protocol.model.val_resize),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        file_column="DDI_file",
    )
    platt = Platt(a=float(protocol.calibration.a), b=float(protocol.calibration.b))
    preds["prob"] = platt.apply(preds["logit"].to_numpy())
    n_boot, seed = int(protocol.bootstrap.n), int(protocol.bootstrap.seed)
    tau95 = float(protocol.thresholds.tau_95)
    p = preds["prob"].to_numpy()
    tasks = {}
    for task, col in (("malignant_vs_benign", "target"), ("melanoma_vs_rest", "melanoma")):
        y_all = preds[col].to_numpy()
        tasks[task] = {"all": task_metrics(p, y_all, tau95, n_boot, seed), "by_group": {}}
        for g in GROUPS.values():
            m = preds["group"].to_numpy() == g
            tasks[task]["by_group"][g] = task_metrics(p[m], y_all[m], tau95, n_boot, seed)
    result = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(ROOT),
        "checkpoint_sha256": sha,
        "protocol": {k: str(v) for k, v in OmegaConf.to_container(protocol.ddi).items()},
        "n_images": int(len(preds)),
        "n_malignant": int(preds["target"].sum()),
        "n_melanoma": int(preds["melanoma"].sum()),
        "melanoma_diagnoses": sorted(
            preds.loc[preds["melanoma"] == 1, "disease"].unique().tolist()
        ),
        "group_sizes": {
            g: {
                "n": int((preds["group"] == g).sum()),
                "malignant": int(preds.loc[preds["group"] == g, "target"].sum()),
                "melanoma": int(preds.loc[preds["group"] == g, "melanoma"].sum()),
            }
            for g in GROUPS.values()
        },
        "tasks": tasks,
        "calibration_isic_platt_on_ddi": calibration_summary(p, preds["target"].to_numpy()),
        "elapsed_min": (time.perf_counter() - t0) / 60,
    }
    for task in tasks:
        assert_no_accuracy(tasks[task])
    out = ROOT / cfg.f4.ddi.results_json
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    # predicciones fuera de git (data/raw/ddi/), no en reports/
    preds.to_csv(ROOT / cfg.f4.ddi.root / "f4_ddi_predictions.csv", index=False)
    for task, r in tasks.items():
        a = r["all"]
        print(
            f"{task}: n={a['n']} pos={a['positives']}  AUROC {a['auroc']:.3f} [{a['auroc_lo']:.3f}, {a['auroc_hi']:.3f}]  AUPRC {a['auprc']:.3f}  sens@τ95 {a['at_tau_95']['sensitivity']:.3f} spec {a['at_tau_95']['specificity']:.3f}"
        )
        for g, v in r["by_group"].items():
            print(
                f"   {g}: n={v['n']} pos={v['positives']}  AUROC {v['auroc']:.3f} [{v['auroc_lo']:.3f}, {v['auroc_hi']:.3f}]"
            )
    c = result["calibration_isic_platt_on_ddi"]
    print(
        f"calibración ISIC en DDI: ECE uniforme {c['ece_uniform']['ece']:.3f}, Brier {c['brier']:.3f}  ({result['elapsed_min']:.1f} min) → {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
