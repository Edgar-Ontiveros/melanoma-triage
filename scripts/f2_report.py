# ruff: noqa: E501
"""F2 — Reportes: `reports/f2_baselines.md` (B0 vs B1, pos_weight vs muestreo ponderado) y
`reports/preprocessing_experiment.md` (F2.6, condiciones A/B/C).

Entradas: `reports/metrics/b0_val.json` (scripts/baseline_metadata.py) y, por cada corrida de
Kaggle copiada a `reports/runs/<run_name>/`, su `metrics.json` y `config.yaml`. Las corridas se
agrupan por la clave `experiment` de la config (b1_pos_weight, b1_sampler, prep_a/b/c). Lo que
falte se marca como pendiente, nunca se inventa. Se reejecuta cada vez que llegan corridas nuevas.
"""

from __future__ import annotations

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
B0 = ROOT / "reports" / "metrics" / "b0_val.json"
FIGURES = ROOT / "reports" / "figures"
PENDING = "_pendiente: corrida de Kaggle no copiada a `reports/runs/`_"


def fmt(x: float | None, d: int = 3) -> str:
    return "n/d" if x is None or x != x else f"{x:.{d}f}"


def ci_str(ci: dict, key: str) -> str:
    return f"[{fmt(ci[key]['lo'])}, {fmt(ci[key]['hi'])}]" if key in ci else ""


def load_runs() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    # kaggle_run.py output deja la salida del kernel anidada: reports/runs/<corrida>/runs/<run_name>/
    for metrics_path in sorted(RUNS.rglob("metrics.json")):
        run = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg_path = metrics_path.with_name("config.yaml")
        cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else OmegaConf.create({})
        run["cfg"] = cfg
        run["dir"] = metrics_path.parent
        name = str(cfg.get("experiment") or "sin_experimento")
        groups.setdefault(name, []).append(run)
    for runs in groups.values():
        runs.sort(key=lambda r: int(r["cfg"].train.seed) if "train" in r["cfg"] else 0)
    return groups


def run_row(label: str, run: dict[str, Any]) -> str:
    m, ci = run["metrics"], run["ci"]
    return (
        f"| {label} | `{run['run_name']}` | {fmt(m['auroc'])} {ci_str(ci, 'auroc')} | "
        f"{fmt(m['auprc'])} {ci_str(ci, 'auprc')} | {fmt(m['sens_at_spec090'])} | "
        f"{fmt(m['spec_at_sens090'])} | {run.get('best_epoch', 'n/d')} / {run['epochs_run']} | "
        f"{'sí: ' + str(run['collapse_epochs']) if run['collapse_epochs'] else 'no'} |"
    )


HEADER = (
    "| modelo | corrida | AUC-ROC [IC 95 %] | AUPRC [IC 95 %] | sens@spec 0.90 | spec@sens 0.90 "
    "| mejor época / corridas | colapso |\n|:--|:--|:--|:--|--:|--:|:--|:--|"
)


def mean_range(runs: list[dict], key: str) -> str:
    vals = [r["metrics"][key] for r in runs]
    if not vals:
        return "n/d"
    if len(vals) == 1:
        return f"{vals[0]:.3f} (una semilla)"
    return f"{statistics.mean(vals):.3f} (rango {min(vals):.3f}–{max(vals):.3f}, n={len(vals)})"


def baselines_report(b0: dict | None, groups: dict[str, list[dict]]) -> str:
    lines = [
        "# F2 — Líneas base sobre validación: B0 (metadatos) vs B1 (ResNet50 a 224 px)",
        "",
        "Generado por `scripts/f2_report.py`. Todas las métricas son sobre `val.txt` (4,963 imágenes,",
        "88 melanomas, 568 pacientes, prevalencia 1.77 %); intervalos por bootstrap de 2,000",
        "remuestreos **a nivel paciente**. el split de prueba no se tocó. La exactitud no se reporta.",
        "",
        "Referencia: un clasificador aleatorio tiene AUC-ROC 0.5 y AUPRC ≈ prevalencia (0.0177).",
        "",
        "## Tabla comparativa",
        "",
        HEADER,
    ]
    if b0:
        m, ci = b0["metrics"], b0["ci"]
        lines.append(
            f"| **B0** metadatos (regresión logística) | `b0` commit `{b0['git_sha']}` | "
            f"{fmt(m['auroc'])} {ci_str(ci, 'auroc')} | {fmt(m['auprc'])} {ci_str(ci, 'auprc')} | "
            f"{fmt(m['sens_at_spec090'])} | {fmt(m['spec_at_sens090'])} | — | — |"
        )
    else:
        lines.append(
            f"| **B0** | {PENDING.replace('Kaggle', 'B0 (make baseline-b0)')} | | | | | | |"
        )
    b1 = groups.get("b1_pos_weight", [])
    for run in b1:
        lines.append(
            run_row(f"**B1** ResNet50-224, pos_weight, semilla {run['cfg'].train.seed}", run)
        )
    if not b1:
        lines.append(f"| **B1** ResNet50-224, pos_weight (3 semillas) | {PENDING} | | | | | | |")
    for run in groups.get("b1_sampler", []):
        lines.append(
            run_row(
                f"**B1** ResNet50-224, muestreo ponderado, semilla {run['cfg'].train.seed}", run
            )
        )
    lines += ["", "## B1: media y rango sobre semillas", ""]
    if b1:
        lines += [
            f"- AUC-ROC: {mean_range(b1, 'auroc')}",
            f"- AUPRC: {mean_range(b1, 'auprc')}",
            f"- Semillas: {[int(r['cfg'].train.seed) for r in b1]}"
            + (" — **faltan semillas para llegar a 3**" if len(b1) < 3 else ""),
        ]
    else:
        lines.append(PENDING)
    lines += ["", "## pos_weight vs. muestreo ponderado (misma semilla)", ""]
    sampler = groups.get("b1_sampler", [])
    pairs = [
        (a, b) for a in b1 for b in sampler if int(a["cfg"].train.seed) == int(b["cfg"].train.seed)
    ]
    if pairs:
        a, b = pairs[0]
        lines += [
            "| estrategia | AUC-ROC [IC] | AUPRC [IC] | colapso |",
            "|:--|:--|:--|:--|",
            f"| pos_weight = {a['provenance']['pos_weight_used']:.1f} | {fmt(a['metrics']['auroc'])} "
            f"{ci_str(a['ci'], 'auroc')} | {fmt(a['metrics']['auprc'])} {ci_str(a['ci'], 'auprc')} | "
            f"{a['collapse_epochs'] or 'no'} |",
            f"| muestreo ponderado | {fmt(b['metrics']['auroc'])} {ci_str(b['ci'], 'auroc')} | "
            f"{fmt(b['metrics']['auprc'])} {ci_str(b['ci'], 'auprc')} | {b['collapse_epochs'] or 'no'} |",
            "",
        ]
        better = a if a["metrics"]["auprc"] >= b["metrics"]["auprc"] else b
        other = b if better is a else a
        overlap = better["ci"]["auprc"]["lo"] <= other["ci"]["auprc"]["hi"]
        lines.append(
            f"**Recomendación para F3:** {'pos_weight' if better is a else 'muestreo ponderado'} "
            f"(mayor AUPRC puntual: {better['metrics']['auprc']:.3f} vs {other['metrics']['auprc']:.3f}). "
            + (
                "Los intervalos se traslapan: la diferencia no es concluyente con una semilla; se elige "
                "por AUPRC puntual y simplicidad, y F3 puede revisarlo."
                if overlap
                else "Los intervalos no se traslapan."
            )
        )
    else:
        lines.append(PENDING)
    lines += ["", "## Lectura", ""]
    if b0 and b1:
        b0_auprc, b1_auprc = (
            b0["metrics"]["auprc"],
            statistics.mean(r["metrics"]["auprc"] for r in b1),
        )
        lines.append(
            f"B1 supera a B0 en AUPRC ({b1_auprc:.3f} vs {b0_auprc:.3f}) "
            + (
                "— las imágenes aportan información que los metadatos no tienen."
                if b1_auprc > b0_auprc
                else "**no**: revisar B1, algo está mal."
            )
        )
    elif b0:
        lines.append(
            "B0 es el piso: cualquier modelo de imágenes por debajo de estos números tiene un bug. "
            "La comparación con B1 se completa cuando lleguen las corridas de Kaggle."
        )
    lines += [
        "",
        "Detalle de B0 en `reports/b0_metadata.md`; de cada corrida en `reports/runs/<run_name>/summary.md`.",
        "",
    ]
    return "\n".join(lines)


def curves_figure(conds: dict[str, dict | None]) -> Path | None:
    if not any(conds.values()):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, run in conds.items():
        if not run:
            continue
        ep = run["epochs"]
        x = [e["epoch"] + 1 for e in ep]
        axes[0].plot(x, [e.get("train/loss", float("nan")) for e in ep], marker="o", label=label)
        axes[1].plot(x, [e["val/auroc"] for e in ep], marker="o", label=label)
        axes[2].plot(x, [e["val/auprc"] for e in ep], marker="o", label=label)
    axes[0].set_title("pérdida de entrenamiento")
    axes[1].set_title("AUC-ROC validación")
    axes[1].axhline(0.5, ls="--", color="gray")
    axes[2].set_title("AUPRC validación")
    axes[2].axhline(0.0177, ls="--", color="gray", label="azar (prevalencia)")
    for ax in axes:
        ax.set_xlabel("época")
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "preprocessing_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def preprocessing_report(groups: dict[str, list[dict]]) -> str:
    conds = {
        "A — correcto": (groups.get("prep_a") or [None])[0],
        "B — ÷255 de más": (groups.get("prep_b") or [None])[0],
        "C — ×255 de más": (groups.get("prep_c") or [None])[0],
    }
    fig = curves_figure(conds)
    lines = [
        "# F2.6 — Experimento de preprocesamiento: qué pasa cuando la escala de entrada es 255× incorrecta",
        "",
        "Generado por `scripts/f2_report.py` a partir de las corridas `prep_a`, `prep_b` y `prep_c`",
        "(`configs/experiment/prep_*.yaml`). Mismo backbone (ResNet50 preentrenado, 224 px), misma",
        "semilla, mismos datos, mismos hiperparámetros; solo cambia `data.preprocess_scale`.",
        "",
        "## Qué afirma la v1 de la tesis y qué se puede probar",
        "",
        "La versión anterior reporta que EfficientNetB3 con capas congeladas obtuvo 50.75 % de",
        "exactitud y concluye que «las capas congeladas impidieron un aprendizaje adecuado». Congelar",
        "el backbone y entrenar solo la cabeza es una práctica estándar que produce resultados muy por",
        "encima del azar con pesos de ImageNet, así que esa conclusión no explica el número. La",
        "hipótesis alternativa es un desajuste de preprocesamiento.",
        "",
        "- **Se puede probar aquí:** que un desajuste de escala de 255× entre lo que el modelo espera y lo",
        "  que recibe produce desempeño de azar (condiciones B y C).",
        "- **No se puede probar sin volver a correr el código original de la v1:** que ese fue exactamente el",
        "  error. Se puede mostrar que es consistente con el resultado y que es la explicación más probable.",
        "- **Es verificable documentalmente:** que en Keras el reescalado de EfficientNet vive dentro del",
        "  grafo del modelo. La documentación oficial (keras.io, *EfficientNet B0 to B7*, nota bajo cada",
        "  función `EfficientNetB0`…`EfficientNetB7`) dice textualmente:",
        "",
        "  > Note: each Keras Application expects a specific kind of input preprocessing. For EfficientNet,",
        "  > input preprocessing is included as part of the model (as a `Rescaling` layer), and thus",
        "  > `keras.applications.efficientnet.preprocess_input` is actually a pass-through function.",
        "  > EfficientNet models expect their inputs to be float tensors of pixels with values in the",
        "  > `[0-255]` range.",
        "",
        "  y sobre `preprocess_input` (*EfficientNet preprocessing utilities*): «A placeholder method for",
        "  backward compatibility. The preprocessing logic has been included in the efficientnet model",
        "  implementation. Users are no longer required to call this method to normalize the input data.»",
        "  Fuentes: <https://keras.io/api/applications/efficientnet/> (consultado el 2026-09-17).",
        "",
        "  Consecuencia: si un pipeline entrega a EfficientNet imágenes ya divididas por 255 (en [0, 1],",
        "  como es habitual con `ImageDataGenerator(rescale=1./255)` o con `preprocess_input` de otras",
        "  familias), el modelo vuelve a reescalar internamente y recibe valores en [0, 1/255]: toda la",
        "  imagen queda comprimida en un rango donde el contraste entre píxeles es ~255 veces menor del",
        "  que vieron los pesos preentrenados.",
        "",
        "## Mecanismo",
        "",
        "Con normalización `(x/255 − media)/desv`, la entrada a la primera convolución tiene media ≈ 0 y",
        "desviación ≈ 1 por canal. Si la imagen llega 255× más pequeña (condición B), `x/255` es ≈ 0 para",
        "todo píxel y la entrada se vuelve casi la constante `−media/desv`: las activaciones son",
        "prácticamente idénticas para toda imagen, la cabeza solo puede aprender el sesgo y el modelo",
        "colapsa a predecir la prevalencia (AUC ≈ 0.5). Con capas congeladas el efecto es total, porque",
        "el backbone no puede reaprender la escala; con fine-tuning completo puede recuperarse en parte,",
        "y por eso el experimento reporta también la curva por época. Si la imagen llega 255× más grande",
        "(condición C), las activaciones se saturan o explotan (valores del orden de cientos en la",
        "entrada), el gradiente es inestable y la pérdida diverge o el modelo colapsa.",
        "",
        "En este repositorio el error se inyecta con `data.preprocess_scale` (`melanoma.data.transforms`):",
        "`divide255` multiplica por 255 el rango de píxel que asume la normalización (la señal útil queda",
        "÷255) y `multiply255` lo divide (señal ×255). La condición A usa el `data_config` de timm sin tocar.",
        "",
        "## Resultados",
        "",
        "| condición | preprocesamiento | AUC-ROC final [IC] | AUPRC final [IC] | colapso (épocas) | std de prob. última época |",
        "|:--|:--|:--|:--|:--|--:|",
    ]
    for label, run in conds.items():
        if run:
            last = run["epochs"][-1] if run["epochs"] else {}
            lines.append(
                f"| {label} | `{run['cfg'].data.preprocess_scale}` | {fmt(run['metrics']['auroc'])} "
                f"{ci_str(run['ci'], 'auroc')} | {fmt(run['metrics']['auprc'])} {ci_str(run['ci'], 'auprc')} | "
                f"{run['collapse_epochs'] or 'no'} | {fmt(last.get('val/prob_std'), 4)} |"
            )
        else:
            lines.append(f"| {label} | | {PENDING} | | | |")
    lines += [""]
    if fig:
        lines += [f"![curvas por época]({fig.relative_to(ROOT / 'reports').as_posix()})", ""]
        lines += [
            "Curvas por época (pérdida de entrenamiento, AUC-ROC y AUPRC de validación) de las tres condiciones.",
            "",
        ]
    else:
        lines += ["_Las curvas se generan cuando las tres corridas estén en `reports/runs/`._", ""]
    a, b, c = conds.values()
    lines += ["## Párrafo para la tesis (corrige la afirmación de la v1)", ""]
    if a and b and c:
        lines.append(
            "> En la versión anterior de este trabajo, EfficientNetB3 con el backbone congelado obtuvo una "
            "exactitud del 50.75 %, y se atribuyó el resultado a que «las capas congeladas impidieron un "
            "aprendizaje adecuado». Esa explicación no se sostiene: congelar un backbone preentrenado es una "
            "práctica estándar que produce resultados muy por encima del azar. La explicación más probable es "
            "un desajuste de preprocesamiento: las implementaciones de EfficientNet en Keras incluyen el "
            "reescalado dentro del modelo y esperan píxeles en [0, 255], de modo que entregarles imágenes ya "
            "normalizadas a [0, 1] reduce 255 veces la señal de entrada. Para comprobar que ese mecanismo basta "
            "para producir desempeño de azar, se entrenó el mismo ResNet50 con la misma semilla y los mismos "
            f"datos bajo tres condiciones: preprocesamiento correcto (AUC-ROC {a['metrics']['auroc']:.3f}, "
            f"AUPRC {a['metrics']['auprc']:.3f}), entrada dividida por 255 de más (AUC-ROC "
            f"{b['metrics']['auroc']:.3f}, AUPRC {b['metrics']['auprc']:.3f}) y entrada multiplicada por 255 de "
            f"más (AUC-ROC {c['metrics']['auroc']:.3f}, AUPRC {c['metrics']['auprc']:.3f}). No es posible "
            "afirmar que ese fue exactamente el error de la versión anterior sin reejecutar aquel código; sí es "
            "posible afirmar que es consistente con el resultado observado y que la conclusión original era "
            "incorrecta."
        )
    else:
        lines.append(
            "_Se redacta con los números de las tres condiciones cuando estén disponibles. El borrador sin "
            "números: la v1 atribuyó el 50.75 % a las capas congeladas; la documentación de Keras muestra que "
            "EfficientNet reescala internamente y espera [0, 255]; el experimento A/B/C muestra que un desajuste "
            "de 255× basta para producir desempeño de azar; no se afirma que ese fue exactamente el error de la "
            "v1, solo que es la explicación más probable y consistente._"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    b0 = json.loads(B0.read_text(encoding="utf-8")) if B0.exists() else None
    groups = load_runs()
    (ROOT / "reports" / "f2_baselines.md").write_text(
        baselines_report(b0, groups), encoding="utf-8"
    )
    (ROOT / "reports" / "preprocessing_experiment.md").write_text(
        preprocessing_report(groups), encoding="utf-8"
    )
    print(
        f"B0: {'ok' if b0 else 'falta'}; corridas en reports/runs: "
        f"{ {k: len(v) for k, v in groups.items()} }"
    )
    print("→ reports/f2_baselines.md, reports/preprocessing_experiment.md")


if __name__ == "__main__":
    main()
