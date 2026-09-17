"""F2.4 — Línea base B0: regresión logística sobre metadatos, sin imágenes.

Responde "¿las imágenes aportan algo?": cualquier modelo de imágenes que no supere
claramente este número tiene un aporte cuestionable, y cualquiera que quede por debajo
tiene un bug. Corre en segundos en la laptop. Solo usa train y val.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from melanoma.data import read_manifest, resolve_paths
from melanoma.data.datamodule import TRAIN_SPLIT, VAL_SPLIT, read_split_ids, verify_split_hashes
from melanoma.eval import (
    assert_no_accuracy,
    bootstrap_patient,
    compute_metrics,
    confusion_table,
    metrics_table,
    plot_all,
)
from melanoma.train.run import git_sha

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(cfg: DictConfig) -> Pipeline:
    numeric = list(cfg.b0.features.numeric)
    categorical = list(cfg.b0.features.categorical)
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy=cfg.b0.impute_numeric, add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value=cfg.b0.missing_category),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    clf = LogisticRegression(C=cfg.b0.C, class_weight=cfg.b0.class_weight, max_iter=cfg.b0.max_iter)
    return Pipeline([("pre", pre), ("clf", clf)])


def feature_names(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]


def missing_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| columna | faltantes | pct |", "|:--|--:|--:|"]
    for c in cols:
        n = int(df[c].isna().sum())
        lines.append(f"| {c} | {n} | {100 * n / len(df):.2f} |")
    return "\n".join(lines)


@hydra.main(config_path="../configs", config_name="baseline_b0", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg.b0))
    paths = resolve_paths(cfg.data, root=ROOT)
    hashes = verify_split_hashes(paths.splits_dir)
    manifest = read_manifest(paths.manifest_path).set_index("image_id")
    cols = list(cfg.b0.features.numeric) + list(cfg.b0.features.categorical)
    train = manifest.loc[read_split_ids(paths.splits_dir, TRAIN_SPLIT)].reset_index()
    val = manifest.loc[read_split_ids(paths.splits_dir, VAL_SPLIT)].reset_index()
    for df in (train, val):
        df["target"] = df["target"].astype(int)

    pipe = build_pipeline(cfg)
    pipe.fit(train[cols], train["target"])
    probs = pipe.predict_proba(val[cols])[:, 1].astype(np.float64)
    y = val["target"].to_numpy()
    pid = val["patient_id"].to_numpy()
    ev = cfg.train.eval
    metrics = compute_metrics(
        probs,
        y,
        pid,
        fixed_levels=list(ev.fixed_levels),
        threshold_sensitivity=ev.threshold_sensitivity,
        n_reliability_bins=ev.reliability_bins,
    )
    ci = bootstrap_patient(
        probs, y, pid, n_resamples=ev.bootstrap_resamples, seed=ev.bootstrap_seed
    )
    assert_no_accuracy(metrics)
    figures = plot_all(metrics, ROOT / cfg.b0.figures_dir, "b0")

    coefs = pipe.named_steps["clf"].coef_.ravel()
    coef_table = "\n".join(
        ["| variable | coeficiente | odds ratio |", "|:--|--:|--:|"]
        + [
            f"| {n} | {c:+.3f} | {np.exp(c):.2f} |"
            for n, c in sorted(
                zip(feature_names(pipe), coefs, strict=True), key=lambda t: -abs(t[1])
            )
        ]
    )
    sha = git_sha(ROOT)
    out_pred = ROOT / cfg.b0.out_predictions
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    val.assign(prob=probs)[["image_id", "patient_id", "target", "prob"]].to_csv(
        out_pred, index=False
    )
    out_metrics = ROOT / cfg.b0.out_metrics
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(
        json.dumps(
            {
                "model": "B0 regresión logística sobre metadatos",
                "git_sha": sha,
                "split_sha256": hashes,
                "config": OmegaConf.to_container(cfg.b0, resolve=True),
                "metrics": {k: v for k, v in metrics.items() if k != "curves"},
                "ci": ci,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = f"""# B0 — Línea base de metadatos (sin imágenes)

Generado por `scripts/baseline_metadata.py` (commit `{sha}`, `configs/baseline_b0.yaml`).
Regresión logística de scikit-learn sobre `{"`, `".join(cols)}`; entrenada en `train.txt`
({len(train):,} imágenes, {int(train["target"].sum())} melanomas) y evaluada en `val.txt`
({len(val):,} imágenes, {int(y.sum())} melanomas, {metrics["n_patients"]} pacientes).
Solo train y val. SHA256 de los splits verificados contra `SHA256SUMS`:
train `{hashes["train"][:12]}…`, val `{hashes["val"][:12]}…`.

## Por qué existe

Responde la pregunta que hace un sinodal: **¿las imágenes aportan algo?** Edad, sexo y sitio
anatómico solos dan un AUC no trivial en ISIC 2020. B1 (F2.5) y los backbones de F3 tienen que
superar claramente estos números; un modelo de imágenes por debajo de B0 tiene un bug.

## Manejo de faltantes

Faltantes en el split de entrenamiento (F1 los cuantificó sobre el dataset completo):

{missing_table(train, cols)}

- `{cfg.b0.features.numeric[0]}`: imputación con la **{cfg.b0.impute_numeric}** del split de
  entrenamiento más una columna indicadora de faltante; después estandarización.
- Categóricas: categoría explícita `{cfg.b0.missing_category}` antes del one-hot, de modo que
  "no se registró el sitio" es una categoría con su propio coeficiente.
- `class_weight={cfg.b0.class_weight}`, `C={cfg.b0.C}`. El peso balanceado solo reescala la
  pérdida; no cambia el ordenamiento que miden AUROC y AUPRC de forma apreciable.

## Métricas sobre validación

{metrics_table(metrics, ci)}

Intervalos: bootstrap de {ev.bootstrap_resamples:,} remuestreos **a nivel paciente** (semilla
{ev.bootstrap_seed}); remuestreos descartados por no tener ambas clases: {ci["auroc"]["n_skipped"]}.

{confusion_table(metrics)}

## Coeficientes

{coef_table}

## Figuras

![ROC](figures/b0_roc.png) ![PR](figures/b0_pr.png) ![fiabilidad](figures/b0_reliability.png)

Predicciones por imagen en `{cfg.b0.out_predictions}`; métricas y IC en `{cfg.b0.out_metrics}`.
"""
    (ROOT / cfg.b0.out_report).write_text(report, encoding="utf-8")
    print(metrics_table(metrics, ci))
    print(f"→ {cfg.b0.out_report}, {cfg.b0.out_metrics}, {cfg.b0.out_predictions}")
    print(f"→ figuras: {[str(f) for f in figures.values()]}")


if __name__ == "__main__":
    main()
