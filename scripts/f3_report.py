# ruff: noqa: E501
"""F3 — `reports/f3_matrix.md`: tabla principal (media y rango sobre semillas), las tres
comparaciones aisladas, figura de la ablación de resolución, tiempos y utilización de GPU,
colapsos, anclaje en la literatura y selección del modelo final por la regla de F3.5.

Entradas: `reports/runs/<corrida>/runs/<run_name>/{metrics.json,config.yaml}` bajadas con
`scripts/kaggle_run.py`; se agrupan por la clave `experiment` (b1_pos_weight, m1, m2, a1).
La regla de decisión está escrita en la spec antes de ver los resultados y aquí solo se aplica.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
FIGURES = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "f3_matrix.md"
TIE = 0.02  # umbral de F2/F3: diferencias menores no distinguen nada
CONFIGS = {
    "b1_pos_weight": ("B1", "resnet50.a1_in1k", 224),
    "a1": ("A1", "tf_efficientnetv2_s.in21k_ft_in1k", 224),
    "m1": ("M1", "tf_efficientnetv2_s.in21k_ft_in1k", 384),
    "m2": ("M2", "convnext_tiny.fb_in22k_ft_in1k_384", 384),
}
METRICS = [
    "auroc",
    "auprc",
    "sens_at_spec090",
    "sens_at_spec095",
    "spec_at_sens090",
    "spec_at_sens095",
]
LABELS = {
    "auroc": "AUC-ROC",
    "auprc": "AUPRC",
    "sens_at_spec090": "sens@spec 0.90",
    "sens_at_spec095": "sens@spec 0.95",
    "spec_at_sens090": "spec@sens 0.90",
    "spec_at_sens095": "spec@sens 0.95",
}
# Latencia relativa estimada en CPU para desempatar (F3.5 → F6): GFLOPs aproximados a la
# resolución de la config, según las fichas de timm. Se mide de verdad en F6.
CPU_COST = {"B1": 4.1, "A1": 2.9, "M1": 8.4, "M2": 13.1}
PENDING = "_pendiente_"


def fmt(x: float | None, d: int = 3) -> str:
    return "n/d" if x is None or x != x else f"{x:.{d}f}"


def load_runs() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(RUNS.rglob("metrics.json")):
        run = json.loads(p.read_text(encoding="utf-8"))
        cfg_path = p.with_name("config.yaml")
        run["cfg"] = OmegaConf.load(cfg_path) if cfg_path.exists() else OmegaConf.create({})
        run["dir"] = p.parent
        name = str(run["cfg"].get("experiment") or "")
        if name in CONFIGS:
            groups.setdefault(name, []).append(run)
    for runs in groups.values():
        runs.sort(key=lambda r: int(r["cfg"].train.seed))
    return groups


def seed_values(runs: list[dict], key: str) -> list[float]:
    return [r["metrics"][key] for r in runs if r["metrics"][key] == r["metrics"][key]]


def mean_range(vals: list[float]) -> str:
    if not vals:
        return "n/d"
    if len(vals) == 1:
        return f"{vals[0]:.3f} (1 semilla)"
    return f"{statistics.mean(vals):.3f} ({min(vals):.3f}–{max(vals):.3f})"


def main_table(groups: dict[str, list[dict]]) -> list[str]:
    lines = [
        "| config | backbone | px | semillas | " + " | ".join(LABELS[m] for m in METRICS) + " |",
        "|:--|:--|--:|--:|" + "|".join(":--" for _ in METRICS) + "|",
    ]
    for key, (label, backbone, px) in CONFIGS.items():
        runs = groups.get(key, [])
        cells = [mean_range(seed_values(runs, m)) for m in METRICS]
        lines.append(
            f"| **{label}** | `{backbone}` | {px} | {len(runs)} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append(
        "Media y rango (mín–máx) sobre semillas; cada corrida evaluada sobre `val.txt` con IC por paciente en su `metrics.json`."
    )
    return lines


def comparison(groups: dict, a: str, b: str, what: str) -> list[str]:
    ra, rb = groups.get(a, []), groups.get(b, [])
    la, lb = CONFIGS[a][0], CONFIGS[b][0]
    if not ra or not rb:
        return [f"- **{what}** ({lb} − {la}): {PENDING}"]
    out = [f"- **{what}** ({lb} − {la}):"]
    for m in ("auroc", "auprc", "spec_at_sens090"):
        va, vb = seed_values(ra, m), seed_values(rb, m)
        d = statistics.mean(vb) - statistics.mean(va)
        verdict = "supera 0.02" if abs(d) >= TIE else "dentro del ruido (< 0.02)"
        out.append(
            f"  - {LABELS[m]}: {statistics.mean(vb):.3f} − {statistics.mean(va):.3f} = **{d:+.3f}** → {verdict}"
        )
    return out


def resolution_figure(groups: dict) -> Path | None:
    a1, m1 = groups.get("a1", []), groups.get("m1", [])
    if not a1 or not m1:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, ("auroc", "auprc"), strict=True):
        for label, runs, x in (("A1 · 224 px", a1, 0), ("M1 · 384 px", m1, 1)):
            ys = [r["metrics"][m] for r in runs]
            ax.scatter([x] * len(ys), ys, s=60, zorder=3, label=label)
            for r, y in zip(runs, ys, strict=True):
                ax.annotate(
                    f"s{r['cfg'].train.seed}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 0),
                    fontsize=8,
                )
        for r_a in a1:
            for r_m in m1:
                if int(r_a["cfg"].train.seed) == int(r_m["cfg"].train.seed):
                    ax.plot([0, 1], [r_a["metrics"][m], r_m["metrics"][m]], color="gray", alpha=0.4)
        ax.set_xticks([0, 1], ["A1 · 224", "M1 · 384"])
        ax.set_title(f"{LABELS[m]} por semilla (mismo backbone)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "f3_resolution_ablation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def times_table(groups: dict) -> list[str]:
    lines = [
        "| corrida | commit | épocas (mejor) | min entrenamiento | min/época | espera de datos | GPU media | colapso |",
        "|:--|:--|:--|--:|--:|--:|--:|:--|",
    ]
    for key in CONFIGS:
        for r in groups.get(key, []):
            ep = r.get("epochs", [])
            epoch_min = [
                e.get("train/epoch_min") for e in ep if e.get("train/epoch_min") is not None
            ]
            wait = [
                e.get("train/data_wait_frac")
                for e in ep
                if e.get("train/data_wait_frac") is not None
            ]
            util = [
                e.get("train/gpu_util")
                for e in ep
                if e.get("train/gpu_util") is not None
                and e["train/gpu_util"] == e["train/gpu_util"]
            ]
            lines.append(
                f"| `{r['run_name']}` | `{r['provenance']['git_sha']}` | {r['epochs_run']} ({r.get('best_epoch')}) | "
                f"{r['elapsed_s'] / 60:.0f} | {fmt(statistics.mean(epoch_min), 1) if epoch_min else 'n/d'} | "
                f"{fmt(100 * statistics.mean(wait), 0) + ' %' if wait else 'n/d'} | "
                f"{fmt(statistics.mean(util), 0) + ' %' if util else 'n/d'} | "
                f"{'sí: ' + str(r['collapse_epochs']) if r['collapse_epochs'] else 'no'} |"
            )
    lines.append("")
    lines.append(
        "Espera de datos: fracción del tiempo del bucle de entrenamiento que el DataLoader tiene a la GPU ociosa (`ThroughputMonitor`). Por debajo de ~70 % de GPU se documenta y se considera más `num_workers` o un redimensionado previo (F3.4)."
    )
    return lines


def decide(groups: dict) -> tuple[str | None, list[str]]:
    """Regla de F3.5: AUPRC medio, luego spec@sens 0.90; empate si ΔAUC-ROC < 0.02 y ΔAUPRC < 0.02 → más barata en CPU."""
    complete = {k: v for k, v in groups.items() if len(v) >= 3}
    if len(complete) < len(CONFIGS):
        missing = [CONFIGS[k][0] for k in CONFIGS if k not in complete]
        return None, [f"Sin decisión: faltan semillas en {missing}."]
    stats = {
        k: {m: statistics.mean(seed_values(v, m)) for m in ("auroc", "auprc", "spec_at_sens090")}
        for k, v in complete.items()
    }
    order = sorted(
        stats, key=lambda k: (stats[k]["auprc"], stats[k]["spec_at_sens090"]), reverse=True
    )
    best = order[0]
    tied = [
        k
        for k in order
        if abs(stats[k]["auroc"] - stats[best]["auroc"]) < TIE
        and abs(stats[k]["auprc"] - stats[best]["auprc"]) < TIE
    ]
    lines = ["Orden por AUPRC medio y luego especificidad a sensibilidad 0.90:", ""]
    for k in order:
        lines.append(
            f"- {CONFIGS[k][0]}: AUPRC {stats[k]['auprc']:.3f}, spec@sens0.90 {stats[k]['spec_at_sens090']:.3f}, AUC-ROC {stats[k]['auroc']:.3f}, costo CPU relativo {CPU_COST[CONFIGS[k][0]]}"
        )
    if len(tied) > 1:
        winner = min(tied, key=lambda k: CPU_COST[CONFIGS[k][0]])
        lines.append("")
        lines.append(
            f"Empate (ΔAUC-ROC y ΔAUPRC < {TIE}) entre {', '.join(CONFIGS[k][0] for k in tied)}: gana la más barata de servir, **{CONFIGS[winner][0]}**."
        )
    else:
        winner = best
        lines.append("")
        lines.append(f"Gana **{CONFIGS[winner][0]}** sin empate.")
    return winner, lines


def final_model(groups: dict, winner: str | None) -> list[str]:
    if winner is None:
        return [PENDING]
    runs = groups[winner]
    s0 = next((r for r in runs if int(r["cfg"].train.seed) == 0), None)
    label, backbone, px = CONFIGS[winner]
    lines = [
        f"- **Configuración:** {label} (`{backbone}` a {px} px), experimento `{winner}`.",
        "- **Checkpoint:** el de semilla 0, fijado antes de mirar los resultados (elegir la mejor semilla sería seleccionar sobre validación).",
    ]
    if s0:
        ckpt = next(iter(sorted((s0["dir"]).rglob("*.ckpt"))), None) or next(
            iter(sorted((RUNS).rglob(f"*{s0['run_name']}*.ckpt"))), None
        )
        if ckpt:
            sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            lines.append(
                f"- **Archivo:** `{ckpt.name}` ({ckpt.stat().st_size / 2**20:.0f} MB), SHA256 `{sha}`."
            )
        else:
            lines.append(
                f"- **Archivo:** checkpoint de `{s0['run_name']}` aún no descargado (`scripts/kaggle_run.py checkpoint {label.lower()}-s0`)."
            )
        lines.append(
            f"- **Corrida:** `{s0['run_name']}`, commit `{s0['provenance']['git_sha']}`, AUC-ROC {s0['metrics']['auroc']:.3f}, AUPRC {s0['metrics']['auprc']:.3f}."
        )
        pub = ROOT / "reports" / "f3_final_model.json"
        if pub.exists():
            info = json.loads(pub.read_text())
            lines.append(
                f"- **Publicado en Kaggle:** `{info['dataset']}` ({info['date']}), SHA256 verificado `{info['sha256'][:16]}…`."
            )
        else:
            lines.append("- **Publicado en Kaggle:** pendiente (`melanoma-f3-final`).")
    return lines


LITERATURE = """Verificado en la fuente el 2026-09-17 (texto completo en ar5iv y resumen en Semantic Scholar).

| referencia | AUC-ROC declarado | sobre qué conjunto | modelos | resoluciones | metadatos | datos externos | TTA |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Ha, Liu y Liu (2020), *Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge*, arXiv:2010.05351, <https://doi.org/10.48550/arXiv.2010.05351> | «0.9600 AUC on cross validation and 0.9490 AUC on private leaderboard» | validación cruzada de 5 pliegues sobre 2018+2019+2020 combinados, y leaderboard privado del reto | ensamble de **18 modelos** (EfficientNet B3–B7, SE-ResNeXt-101, ResNeSt-101) | 384, 448, 512, 576, 640, 768 y 896 | sí, en 4 de los 18 (sexo, edad, sitio anatómico, tamaño de imagen, n_images) | sí: ISIC 2018 y 2019 junto con 2020 | no se menciona en el texto consultado |
| Cassidy, Kendrick, Brodzicki, Jaworek-Korjakowska y Yap (2022), *Analysis of the ISIC image datasets: Usage, benchmarks and recommendations*, Medical Image Analysis 75, 102305, <https://doi.org/10.1016/j.media.2021.102305> | «an AUC of 0.80 for the best performing model» | conjunto de prueba de ISIC 2020, tras eliminar 14,310 duplicados del entrenamiento y balancear | modelos únicos (varias arquitecturas; el resumen no las nombra) | no indicado en el resumen | no | no («our aim was not to maximise network performance») | no |

Lectura: el ganador del reto es un ensamble de 18 redes grandes a resoluciones de hasta 896 px, con datos de tres años y metadatos en parte de los modelos. Un modelo único sin datos externos, sin TTA y sin metadatos, entrenado con la cuota gratuita de Kaggle, queda por diseño varios puntos por debajo; la referencia de modelo único de Cassidy et al. (0.80 en el test de ISIC 2020) es el punto de comparación más cercano a este pipeline, con la salvedad de que aquí se evalúa sobre un split de validación por paciente y no sobre el test del reto."""


def literature_paragraph(groups: dict, winner: str | None) -> str:
    if winner is None:
        return "_El párrafo de situación se redacta con la configuración ganadora._"
    label = CONFIGS[winner][0]
    auc = statistics.mean(seed_values(groups[winner], "auroc"))
    return (
        f"**Situación de {label}:** AUC-ROC medio de {auc:.3f} sobre validación (por paciente, 88 melanomas). "
        "Está muy por debajo del 0.949 del ganador en el leaderboard privado, y eso es lo esperable: aquel número sale de 18 modelos "
        "de hasta 896 px, con ISIC 2018 y 2019 añadidos al entrenamiento y metadatos en parte del ensamble; aquí hay un modelo único "
        f"a {CONFIGS[winner][2]} px, sin datos externos, sin metadatos, sin TTA y con unas 12 h de GPU gratuita para toda la matriz. "
        f"Frente al modelo único de Cassidy et al. (0.80 en el test de ISIC 2020 tras eliminar duplicados), {label} queda "
        f"{'por encima' if auc > 0.80 else 'por debajo' if auc < 0.79 else 'a la par'}, con la salvedad de que los conjuntos de evaluación no son el mismo."
    )


def main() -> None:
    groups = load_runs()
    winner, decision = decide(groups)
    fig = resolution_figure(groups)
    lines = [
        "# F3 — Matriz de modelos: comparación controlada sobre validación",
        "",
        "Generado por `scripts/f3_report.py`. Lo fijo (idéntico a B1): cabeza de la fábrica, BCE con `pos_weight` automático, AdamW 1e-4, cosine con warmup, 15 épocas con early stopping por AUPRC (paciencia 5), aumentación de F2, lado corto + CenterCrop en validación, **lote efectivo 128** (acumulación de gradiente), semillas 0/1/2, 16-mixed. Lo variable: backbone y resolución. Lo vigila `tests/test_f3_matrix.py`. `test.txt` no se toca.",
        "",
        "## 1. Tabla principal",
        "",
        *main_table(groups),
        "",
        "## 2. Comparaciones aisladas",
        "",
        *comparison(groups, "a1", "m1", "Efecto de la resolución, mismo backbone (224 → 384)"),
        *comparison(
            groups,
            "b1_pos_weight",
            "a1",
            "Efecto de la arquitectura a 224 px (ResNet50 → EfficientNetV2-S)",
        ),
        *comparison(
            groups, "m1", "m2", "Efecto de la arquitectura a 384 px (EfficientNetV2-S → ConvNeXt-T)"
        ),
        "",
        f"Umbral: diferencias de medias menores que {TIE} no distinguen nada (dos ejecuciones de la misma semilla difirieron 0.016 de AUC-ROC en F2).",
        "",
        "## 3. Ablación de resolución por semilla",
        "",
        f"![A1 vs M1]({fig.relative_to(ROOT / 'reports').as_posix()})" if fig else PENDING,
        "",
        "## 4. Tiempos y utilización de GPU",
        "",
        *times_table(groups),
        "",
        "## 5. Colapsos",
        "",
        *(
            [
                f"- `{r['run_name']}`: épocas {r['collapse_epochs']}"
                for k in CONFIGS
                for r in groups.get(k, [])
                if r["collapse_epochs"]
            ]
            or ["Ninguno."]
        ),
        "",
        "## 6. Regla de decisión (F3.5) aplicada",
        "",
        *decision,
        "",
        "## 7. Anclaje en la literatura (F3.6)",
        "",
        LITERATURE,
        "",
        literature_paragraph(groups, winner),
        "",
        "## 8. Modelo final para F4 (F3.7)",
        "",
        *final_model(groups, winner),
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"corridas: { {CONFIGS[k][0]: len(v) for k, v in groups.items()} } → {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
