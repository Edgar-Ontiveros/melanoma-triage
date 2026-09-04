"""F1.6 — Análisis exploratorio honesto: ``reports/eda.md`` y figuras.

Solo cuenta y describe. No afirma diferencias visuales entre clases ni conclusiones sobre
viabilidad. Contenido: clases, imágenes por paciente, faltantes, sitio/sexo/edad,
dimensiones originales, verificación de splits y contraste con el dataset anterior.

Uso: ``uv run python scripts/eda_report.py``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from melanoma.data.manifest import missing_report, read_manifest
from melanoma.data.splits import patients_in_two_splits, split_summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREVIOUS_VERSION = {"imagenes": 9625, "melanoma_pct": 48.0, "fuente": "versión anterior"}


def _by(df: pd.DataFrame, col: str, by_value: bool = False) -> pd.DataFrame:
    """Imágenes, melanomas y prevalencia por valor de ``col``; los faltantes son una fila más."""
    g = df.groupby(col, dropna=False).agg(
        imagenes=("image_id", "size"), melanomas=("target", "sum")
    )
    g["pct_imagenes"] = (100 * g["imagenes"] / len(df)).round(2)
    g["prevalencia_pct"] = (100 * g["melanomas"] / g["imagenes"]).round(2)
    if by_value:  # orden natural del valor (p. ej. edad), faltantes al final
        g = g.sort_index(na_position="last")
    else:
        g = g.sort_values("imagenes", ascending=False)
    g.index = pd.Index(["(faltante)" if pd.isna(v) else str(v) for v in g.index], name=col)
    return g.reset_index()


def _hist_images_per_patient(counts: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.arange(0, counts.max() + 5, 5)
    ax.hist(counts, bins=bins, color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xlabel("imágenes por paciente")
    ax.set_ylabel("pacientes (log)")
    ax.set_title("Imágenes por paciente, ISIC 2020")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _dims_png(m: pd.DataFrame, out: Path) -> None:
    dims = m.groupby(["width", "height"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dims["width"], dims["height"], s=np.sqrt(dims["n"]) * 4, alpha=0.6, color="#4c72b0")
    ax.set_xlabel("ancho (px)")
    ax.set_ylabel("alto (px)")
    ax.set_title("Dimensiones originales (área ∝ √n)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    figures = Path(d.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    m = read_manifest(d.manifest_path)
    n = len(m)
    pos = int(m["target"].sum())

    # 1. clases
    classes = pd.DataFrame(
        {
            "clase": ["benigno (target=0)", "melanoma (target=1)", "total"],
            "imagenes": [n - pos, pos, n],
            "pct": [round(100 * (n - pos) / n, 2), round(100 * pos / n, 2), 100.0],
        }
    )
    diag = _by(m, "diagnosis")

    # 2. imágenes por paciente
    per_patient = m.groupby("patient_id").size()
    _hist_images_per_patient(per_patient, figures / "images_per_patient.png")
    pp_stats = pd.DataFrame(
        {
            "estadistico": ["pacientes", "mínimo", "p25", "mediana", "p75", "máximo", "media"],
            "valor": [
                len(per_patient),
                int(per_patient.min()),
                float(per_patient.quantile(0.25)),
                float(per_patient.median()),
                float(per_patient.quantile(0.75)),
                int(per_patient.max()),
                round(float(per_patient.mean()), 2),
            ],
        }
    )
    pat_pos = m.groupby("patient_id")["target"].max()
    mel_per_patient = m[m["target"] == 1].groupby("patient_id").size()

    # 3. faltantes
    missing = missing_report(m[["sex", "age_approx", "anatom_site", "diagnosis"]])

    # 4. sitio, sexo, edad
    site = _by(m, "anatom_site")
    sex = _by(m, "sex")
    age = _by(m, "age_approx", by_value=True)

    # 5. dimensiones
    dims = (
        m.groupby(["width", "height"])
        .size()
        .reset_index(name="imagenes")
        .sort_values("imagenes", ascending=False)
    )
    dims["pct"] = (100 * dims["imagenes"] / n).round(2)
    _dims_png(m, figures / "original_dimensions.png")
    long_side = m[["width", "height"]].max(axis=1)
    mp = (m["width"] * m["height"] / 1e6).round(1)

    # 6. splits
    splits_dir = Path(d.splits_dir)
    names = list(d.split.names)
    split = pd.concat(
        [
            pd.Series(name, index=(splits_dir / f"{name}.txt").read_text().split(), name="split")
            for name in names
        ]
    )
    summary = split_summary(m, split, names)
    leaks = patients_in_two_splits(m, split)
    covered = set(split.index) == set(m["image_id"])

    prev = PREVIOUS_VERSION
    compare = pd.DataFrame(
        {
            "": [
                "imágenes",
                "melanoma (%)",
                "pacientes identificados",
                "split agrupado por paciente",
                "conjunto de prueba bloqueado",
            ],
            "versión anterior": [
                f"{prev['imagenes']:,}",
                f"{prev['melanoma_pct']:.0f} %",
                "no",
                "no",
                "no",
            ],
            "ISIC 2020 (esta versión)": [
                f"{n:,}",
                f"{100 * pos / n:.2f} %",
                f"sí ({m['patient_id'].nunique():,})",
                "sí (70/15/15)",
                "sí (2 accesos)",
            ],
        }
    )

    L = [
        "# EDA ISIC 2020 — análisis exploratorio",
        "",
        f"Generado por `scripts/eda_report.py` a partir de `{d.manifest_path}`. Solo conteos; "
        "no se hacen afirmaciones visuales sobre las clases ni juicios de viabilidad.",
        "",
        "## 1. Distribución de clases",
        "",
        classes.to_markdown(index=False),
        "",
        "Por diagnóstico (`diagnosis`):",
        "",
        diag.to_markdown(index=False),
        "",
        "## 2. Imágenes por paciente",
        "",
        pp_stats.to_markdown(index=False),
        "",
        f"- Pacientes con al menos un melanoma: {int(pat_pos.sum()):,} de {len(pat_pos):,} "
        f"({100 * pat_pos.mean():.2f} %)",
        f"- Melanomas por paciente positivo: mediana {float(mel_per_patient.median()):.0f}, "
        f"máximo {int(mel_per_patient.max())}",
        "",
        "![imágenes por paciente](figures/images_per_patient.png)",
        "",
        "## 3. Faltantes por columna de metadatos",
        "",
        missing.to_markdown(index=False),
        "",
        "`diagnosis` no tiene vacíos, pero el valor `unknown` cubre la mayoría de las filas "
        "(ver tabla por diagnóstico arriba).",
        "",
        "## 4. Sitio anatómico, sexo y edad",
        "",
        "### Sitio anatómico",
        "",
        site.to_markdown(index=False),
        "",
        "### Sexo",
        "",
        sex.to_markdown(index=False),
        "",
        "### Edad aproximada",
        "",
        age.to_markdown(index=False),
        "",
        "## 5. Dimensiones originales",
        "",
        f"- Lado largo: mín {int(long_side.min())}, mediana {int(long_side.median())}, "
        f"máx {int(long_side.max())} px",
        f"- Megapíxeles: mín {mp.min()}, mediana {mp.median()}, máx {mp.max()}",
        f"- Tamaño de archivo: mín {m['bytes'].min() / 1e6:.2f} MB, mediana "
        f"{m['bytes'].median() / 1e6:.2f} MB, máx {m['bytes'].max() / 1e6:.2f} MB, "
        f"total {m['bytes'].sum() / 1e9:.2f} GB",
        f"- Combinaciones (ancho, alto) distintas: {len(dims)}. Las 15 más frecuentes:",
        "",
        dims.head(15).to_markdown(index=False),
        "",
        "![dimensiones](figures/original_dimensions.png)",
        "",
        "## 6. Verificación de los splits",
        "",
        f"- Pacientes en más de un split: **{len(leaks)}**",
        "- La unión de los tres splits es exactamente el manifiesto: "
        f"**{'sí' if covered else 'NO'}**",
        "",
        summary.to_markdown(index=False),
        "",
        "## 7. Contraste con el dataset de la versión anterior",
        "",
        compare.to_markdown(index=False),
        "",
        "El dataset anterior tenía la mitad de sus imágenes etiquetadas como melanoma; la "
        "prevalencia real en ISIC 2020 es de menos del 2 %. Cualquier métrica de la versión "
        "anterior se calculó sobre una distribución de clases que no corresponde a la del "
        "problema, sin agrupación por paciente y sin conjunto de prueba independiente.",
        "",
    ]
    out = Path(d.reports_dir) / "eda.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"Reporte: {out}")


if __name__ == "__main__":
    main()
