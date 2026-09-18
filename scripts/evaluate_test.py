# ruff: noqa: E501
"""F4.4 — El acceso único al conjunto de prueba.

Único archivo del proyecto autorizado a leer ``test.txt`` (``tests/test_test_split_isolation``).
Orden obligatorio, verificado por ``tests/test_f4_protocol.py``:

1. el protocolo ``docs/f4_protocol.yaml`` está commiteado sin cambios (si no, se niega);
2. el SHA256 del checkpoint coincide con el protocolo (si no, se niega);
3. se escribe la bitácora ``logs/test_set_access.log`` ANTES de leer una sola imagen; el
   registro es incondicional: si algo falla después, el acceso ya quedó registrado;
4. se lee ``test.txt``, se infiere en CPU;
5. se aplican calibración y umbrales tal como están en el protocolo, nada se reajusta;
6. métricas con bootstrap por paciente y subgrupos (F4.5);
7. ``reports/f4_test_results.json``, ``reports/predictions/f4_test.csv`` y figuras.

Uso (una sola vez): ``uv run python scripts/evaluate_test.py``. Un segundo acceso solo se
justifica por un error de código y queda registrado con su motivo (``--reason``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma.data import read_manifest, resolve_paths  # noqa: E402
from melanoma.data.datamodule import (  # noqa: E402
    read_split_ids,
    split_file_sha256,
    verify_split_hashes,
)
from melanoma.eval import (  # noqa: E402
    Platt,
    bootstrap_patient,
    calibration_summary,
    compute_metrics,
    plot_all,
    ppv_npv_at_prevalence,
)
from melanoma.eval.metrics import (  # noqa: E402
    assert_no_accuracy,
    auroc_auprc,
    confusion_at_threshold,
)
from melanoma.eval.predict import (  # noqa: E402
    load_checkpoint,
    predict_frame,
    sha256_file,
    split_records,
)
from melanoma.train.run import git_sha  # noqa: E402

TEST_SPLIT = "test"
PROTOCOL = "docs/f4_protocol.yaml"


# ---- pasos 1-3: guardarraíles -------------------------------------------------------------
def ensure_protocol_committed(root: Path, protocol: str = PROTOCOL) -> str:
    """El protocolo debe estar versionado y sin cambios locales. Devuelve su SHA256."""
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--", protocol], text=True
    ).strip()
    if not tracked:
        raise SystemExit(
            f"{protocol} no está en git: el protocolo debe commitearse antes de evaluar"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", protocol], text=True
    ).strip()
    if dirty:
        raise SystemExit(f"{protocol} tiene cambios sin commitear ({dirty!r}): se niega a evaluar")
    return hashlib.sha256((root / protocol).read_bytes()).hexdigest()


def verify_checkpoint(root: Path, protocol_cfg) -> str:
    ckpt = root / "data" / "models" / str(protocol_cfg.model.checkpoint_file)
    if not ckpt.exists():
        raise SystemExit(f"falta el checkpoint {ckpt}")
    sha = sha256_file(ckpt)
    if sha != str(protocol_cfg.model.checkpoint_sha256):
        raise SystemExit(
            f"SHA256 del checkpoint {sha} ≠ protocolo {protocol_cfg.model.checkpoint_sha256}"
        )
    return sha


def append_access_log(root: Path, log_path: str, protocol_sha: str, reason: str) -> str:
    """Registra el acceso ANTES de leer el split. Incondicional."""
    line = (
        f"{datetime.now(UTC).isoformat(timespec='seconds')}\tgit={git_sha(root)}\t"
        f"protocol_sha256={protocol_sha}\tsplit={TEST_SPLIT}\treason={reason}\n"
    )
    path = root / log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    return line


def read_test_ids(splits_dir: Path) -> list[str]:
    """Paso 4: la única lectura de test.txt del proyecto."""
    return read_split_ids(splits_dir, TEST_SPLIT)


# ---- pasos 5-7 -----------------------------------------------------------------------------
def age_group(age) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    return "<40" if age < 40 else "40-59" if age < 60 else "60+"


def subgroup_table(
    frame: pd.DataFrame, prob_col: str, tau: float, n_boot: int, seed: int
) -> list[dict]:
    rows = []
    groups = {
        "age": frame["age_approx"].map(age_group),
        "sex": frame["sex"].fillna("unknown").astype(str),
        "site": frame["anatom_site"].fillna("unknown").astype(str),
    }
    for dim, series in groups.items():
        for value in sorted(series.unique(), key=str):
            sub = frame[series == value]
            y = sub["target"].to_numpy()
            p = sub[prob_col].to_numpy()
            row = {
                "dimension": dim,
                "group": str(value),
                "n": int(len(sub)),
                "positives": int(y.sum()),
            }
            if 0 < y.sum() < len(sub):
                auroc, auprc = auroc_auprc(p, y)
                ci = bootstrap_patient(p, y, sub["patient_id"].to_numpy(), n_boot, seed)
                row.update(
                    {
                        "auroc": auroc,
                        "auroc_lo": ci["auroc"]["lo"],
                        "auroc_hi": ci["auroc"]["hi"],
                        "auprc": auprc,
                    }
                )
            else:
                row.update(
                    {
                        "auroc": float("nan"),
                        "auroc_lo": float("nan"),
                        "auroc_hi": float("nan"),
                        "auprc": float("nan"),
                    }
                )
            c = confusion_at_threshold(p, y, tau)
            row.update(
                {"sensitivity_at_tau95": c["sensitivity"], "specificity_at_tau95": c["specificity"]}
            )
            rows.append(row)
    return rows


def evaluate(cfg, protocol_cfg, preds: pd.DataFrame) -> dict:
    """Paso 5-6: calibración y umbrales del protocolo, métricas con IC por paciente."""
    platt = Platt(a=float(protocol_cfg.calibration.a), b=float(protocol_cfg.calibration.b))
    preds = preds.copy()
    preds["prob"] = platt.apply(preds["logit"].to_numpy())
    y, pid, p = preds["target"].to_numpy(), preds["patient_id"].to_numpy(), preds["prob"].to_numpy()
    n_boot, seed = int(protocol_cfg.bootstrap.n), int(protocol_cfg.bootstrap.seed)
    tau95, tau90 = float(protocol_cfg.thresholds.tau_95), float(protocol_cfg.thresholds.tau_90)
    metrics = compute_metrics(p, y, pid, fixed_levels=[0.90, 0.95], threshold=tau95)
    ci = bootstrap_patient(p, y, pid, n_boot, seed)
    conf95, conf90 = confusion_at_threshold(p, y, tau95), confusion_at_threshold(p, y, tau90)
    projections = [
        ppv_npv_at_prevalence(conf95["sensitivity"], conf95["specificity"], prev)
        for prev in protocol_cfg.ppv_npv_projection_prevalences
    ]
    out = {
        "metrics": {k: v for k, v in metrics.items() if k != "curves"},
        "curves": metrics["curves"],
        "ci": ci,
        "calibration": calibration_summary(p, y),
        "calibration_raw": calibration_summary(preds["prob_raw"].to_numpy(), y),
        "confusion_tau_95": conf95,
        "confusion_tau_90": conf90,
        "ppv_npv_projection_tau_95": projections,
        "subgroups": subgroup_table(preds, "prob", tau95, n_boot, seed),
    }
    assert_no_accuracy(out["metrics"])
    return out, preds


def run(cfg, root: Path, reason: str) -> dict:
    t0 = time.perf_counter()
    protocol_cfg = OmegaConf.load(root / cfg.f4.protocol)
    protocol_sha = ensure_protocol_committed(root)  # 1
    ckpt_sha = verify_checkpoint(root, protocol_cfg)  # 2
    log_line = append_access_log(root, cfg.f4.access_log, protocol_sha, reason)  # 3
    print("acceso registrado:", log_line.strip())
    paths = resolve_paths(cfg.data, root=root)
    verify_split_hashes(paths.splits_dir)
    ids = read_test_ids(paths.splits_dir)  # 4
    test_sha = split_file_sha256(paths.splits_dir, TEST_SPLIT)
    manifest = read_manifest(paths.manifest_path)
    recs = split_records(manifest, ids)
    lit, data_config = load_checkpoint(
        root / "data" / "models" / str(protocol_cfg.model.checkpoint_file),
        str(protocol_cfg.model.backbone),
        cfg.model.num_classes,
        cfg.model.dropout,
    )
    preds = predict_frame(
        lit,
        data_config,
        recs,
        paths.images_dir,
        int(protocol_cfg.model.image_size),
        str(protocol_cfg.model.val_resize),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    result, preds = evaluate(cfg, protocol_cfg, preds)  # 5-6
    result.update(
        {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(root),
            "protocol_sha256": protocol_sha,
            "checkpoint_sha256": ckpt_sha,
            "test_split_sha256": test_sha,
            "access_log_line": log_line.strip(),
            "n_images": int(len(preds)),
            "n_positives": int(preds["target"].sum()),
            "n_patients": int(preds["patient_id"].nunique()),
            "elapsed_min": (time.perf_counter() - t0) / 60,
        }
    )
    out = root / cfg.f4.test_results_json  # 7
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pred_path = root / cfg.f4.test_predictions
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    plot_all(
        {**result["metrics"], "curves": result["curves"]}, root / cfg.f4.figures_dir, "f4_test"
    )
    m, c = result["metrics"], result["ci"]
    print(
        f"TEST: AUROC {m['auroc']:.3f} [{c['auroc']['lo']:.3f}, {c['auroc']['hi']:.3f}]  AUPRC {m['auprc']:.3f} [{c['auprc']['lo']:.3f}, {c['auprc']['hi']:.3f}]"
    )
    print(
        f"τ95: sens {result['confusion_tau_95']['sensitivity']:.3f} spec {result['confusion_tau_95']['specificity']:.3f}  → {out.relative_to(root)}"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="F4.4: acceso único al conjunto de prueba")
    ap.add_argument("--reason", default="F4.4 evaluación clínica (primer acceso)")
    args = ap.parse_args()
    with __import__("hydra").initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base="1.3"
    ):
        cfg = __import__("hydra").compose(config_name="f4")
    run(cfg, ROOT, args.reason)


if __name__ == "__main__":
    main()
