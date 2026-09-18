# ruff: noqa: E501
"""F4.7 — `reports/f4_clinical_evaluation.md`, diez secciones en el orden de la spec. Lo que
depende del test se marca como pendiente hasta que exista reports/f4_test_results.json. La
sección 10 (limitaciones) la escribe Edgar en docs/f4_limitations.md y aquí solo se incluye.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "f4_protocol.yaml"
CAL = ROOT / "reports" / "f4_calibration.json"
TEST = ROOT / "reports" / "f4_test_results.json"
DDI = ROOT / "reports" / "f4_ddi_results.json"
LIMITS = ROOT / "docs" / "f4_limitations.md"
LOG = ROOT / "logs" / "test_set_access.log"
OUT = ROOT / "reports" / "f4_clinical_evaluation.md"
PENDING = "_pendiente: el conjunto de prueba no se ha abierto (sesión 3 de F4)_"


def fmt(x, d=3):
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def ci(d: dict, key: str) -> str:
    return f"{fmt(d[key]['point'])} [{fmt(d[key]['lo'])}, {fmt(d[key]['hi'])}]"


def protocol_commit() -> str:
    out = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "log",
                "--diff-filter=A",
                "--format=%h %ci",
                "--",
                "docs/f4_protocol.yaml",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    return out[-1] if out else "n/d"


def section_calibration(cal: dict) -> list[str]:
    rows = [
        "| probabilidades | Brier | ECE (10 bins uniformes) | ECE (10 bins por cuantiles) |",
        "|:--|--:|--:|--:|",
    ]
    for k, label in (
        ("raw", "crudas (sigmoide del logit)"),
        ("temperature", f"temperatura T = {cal['temperature']['T']:.3f}"),
        ("platt", f"Platt a = {cal['platt']['a']:.3f}, b = {cal['platt']['b']:.3f} (primaria)"),
    ):
        c = cal["calibration"][k]
        rows.append(
            f"| {label} | {c['brier']:.4f} | {c['ece_uniform']['ece']:.4f} | {c['ece_quantile']['ece']:.4f} |"
        )
    bins = cal["calibration"]["platt"]["ece_uniform"]["bins"]
    occupied = [b for b in bins if b["n"]]
    return [
        f"Ajustadas sobre validación ({cal['n_images']:,} imágenes, {cal['n_positives']} melanomas, {cal['n_patients']} pacientes), commit `{cal['git_sha']}`.",
        "",
        *rows,
        "",
        f"**Verificación teórica de `b`:** con `pos_weight` = {cal['pos_weight_used']}, se esperaba b ≈ −ln({cal['pos_weight_used']}) = {cal['expected_b']:.3f}; Platt dio b = {cal['platt']['b']:.3f} (diferencia {cal['platt']['b'] - cal['expected_b']:+.3f}). La pendiente a = {cal['platt']['a']:.3f} < 1 es la sobreconfianza habitual de la red, que la temperatura sola (T = {cal['temperature']['T']:.2f}) corrige en escala pero no en desplazamiento: por eso Platt reduce el Brier de {cal['calibration']['raw']['brier']:.4f} a {cal['calibration']['platt']['brier']:.4f} y la temperatura apenas a {cal['calibration']['temperature']['brier']:.4f}.",
        f"El orden se preserva (AUC-ROC idéntico antes y después: {cal['ranking_preserved']}). Bins uniformes ocupados tras Platt: {len(occupied)} de 10 (el primero concentra {occupied[0]['n']:,} imágenes); por eso se reportan también los bins por cuantiles.",
        "",
        "![fiabilidad en validación](figures/f4_val_reliability.png)",
    ]


def section_thresholds(cal: dict) -> list[str]:
    out = [
        "| umbral | mediana (bootstrap) | p5 | p95 | corte único | sens (val) | spec (val) | VPP (val) | VPN (val) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for k in ("tau_95", "tau_90"):
        t = cal["thresholds"][k]
        o = t["on_full_set"]
        out.append(
            f"| {k} | {t['tau']:.4f} | {t['p05']:.4f} | {t['p95']:.4f} | {t['single_cut']:.4f} | {o['sensitivity']:.3f} | {o['specificity']:.3f} | {o['ppv']:.3f} | {o['npv']:.4f} |"
        )
    out.append("")
    out.append(
        "Sobre probabilidades calibradas con Platt; 2,000 remuestreos por paciente. Con 88 melanomas, sensibilidad ≥ 0.95 son 84 aciertos: el umbral de un corte depende de cuatro casos, la mediana no."
    )
    return out


def section_test(test: dict | None) -> list[str]:
    if not test:
        return [PENDING]
    m, c = test["metrics"], test["ci"]
    rows = ["| métrica | test (IC 95 % por paciente) |", "|:--|:--|"]
    for key, label in (
        ("auroc", "AUC-ROC"),
        ("auprc", "AUPRC"),
        ("sens_at_spec090", "sensibilidad a especificidad 0.90"),
        ("sens_at_spec095", "sensibilidad a especificidad 0.95"),
        ("spec_at_sens090", "especificidad a sensibilidad 0.90"),
        ("spec_at_sens095", "especificidad a sensibilidad 0.95"),
    ):
        rows.append(f"| {label} | {ci(c, key)} |")
    cal = test["calibration"]
    raw = test["calibration_raw"]
    rows += [
        f"| ECE uniforme / cuantiles (Platt congelada) | {cal['ece_uniform']['ece']:.4f} / {cal['ece_quantile']['ece']:.4f} (crudas: {raw['ece_uniform']['ece']:.4f}) |",
        f"| Brier (Platt congelada) | {cal['brier']:.4f} (crudas: {raw['brier']:.4f}) |",
        f"| prevalencia observada | {m['prevalence']:.4f} ({test['n_positives']} de {test['n_images']:,}, {test['n_patients']} pacientes) |",
    ]
    return rows + [
        "",
        "![ROC test](figures/f4_test_roc.png) ![PR test](figures/f4_test_pr.png) ![fiabilidad test](figures/f4_test_reliability.png)",
    ]


def section_operating_point(test: dict | None) -> list[str]:
    if not test:
        return [PENDING]
    out = []
    for k, label in (("confusion_tau_95", "τ95"), ("confusion_tau_90", "τ90")):
        c = test[k]
        out += [
            f"**{label} = {c['threshold']:.4f}**",
            "",
            "| | pred. benigno | pred. melanoma |",
            "|:--|--:|--:|",
            f"| real benigno | {c['tn']} | {c['fp']} |",
            f"| real melanoma | {c['fn']} | {c['tp']} |",
            "",
            f"sensibilidad {c['sensitivity']:.3f} · especificidad {c['specificity']:.3f} · VPP {c['ppv']:.3f} · VPN {c['npv']:.4f} a la prevalencia observada ({test['metrics']['prevalence']:.4f})",
            "",
        ]
    out += [
        "**Proyección de τ95 a otras prevalencias** (aritmética desde sensibilidad y especificidad del test):",
        "",
        "| prevalencia | VPP | VPN | fracción referida |",
        "|--:|--:|--:|--:|",
    ]
    for p in test["ppv_npv_projection_tau_95"]:
        out.append(
            f"| {100 * p['prevalence']:.0f} % | {p['ppv']:.3f} | {p['npv']:.4f} | {p['referred_fraction']:.3f} |"
        )
    return out


def section_gap(cal: dict, test: dict | None) -> list[str]:
    if not test:
        return [PENDING]
    mv, mt = cal["metrics_platt"], test["metrics"]
    rows = ["| métrica | validación (gastada) | test (limpio) | brecha |", "|:--|--:|--:|--:|"]
    for key, label in (
        ("auroc", "AUC-ROC"),
        ("auprc", "AUPRC"),
        ("spec_at_sens090", "spec@sens 0.90"),
        ("spec_at_sens095", "spec@sens 0.95"),
    ):
        rows.append(f"| {label} | {mv[key]:.3f} | {mt[key]:.3f} | {mt[key] - mv[key]:+.3f} |")
    return rows + [
        "",
        "**Advertencia de selección.** Sobre validación se eligieron la línea base, el desbalance, la arquitectura, la resolución, la época de cada corrida, la calibración y el umbral; sus métricas están sesgadas al alza. El test es la única estimación limpia; una brecha negativa es la brecha de generalización, no un bug.",
    ]


def section_subgroups(test: dict | None) -> list[str]:
    if not test:
        return [PENDING]
    rows = [
        "| dimensión | grupo | n | melanomas | AUC-ROC [IC] | sens@τ95 | spec@τ95 |",
        "|:--|:--|--:|--:|:--|--:|--:|",
    ]
    for r in test["subgroups"]:
        rows.append(
            f"| {r['dimension']} | {r['group']} | {r['n']} | {r['positives']} | {fmt(r['auroc'])} [{fmt(r['auroc_lo'])}, {fmt(r['auroc_hi'])}] | {fmt(r['sensitivity_at_tau95'])} | {fmt(r['specificity_at_tau95'])} |"
        )
    return rows + [
        "",
        "**Exploratorio.** Con 86 melanomas en test, los subgrupos tienen entre 5 y 30 positivos y los intervalos son anchos: este análisis detecta diferencias grandes, no confirma diferencias pequeñas. No se concluye nada de una diferencia cuyo intervalo cruce el del grupo de referencia.",
    ]


def section_ddi(ddi: dict | None) -> list[str]:
    if not ddi:
        return ["_pendiente_"]
    out = [
        f"Corrido en local, en CPU, el {ddi['date'][:10]} (commit `{ddi['git_sha']}`, checkpoint `{ddi['checkpoint_sha256'][:16]}…`). {ddi['n_images']} imágenes clínicas con biopsia; {ddi['n_malignant']} malignas (26 %); {ddi['n_melanoma']} melanomas en `disease` ({', '.join(ddi['melanoma_diagnoses'])}). Misma transformación que validación (lado corto + CenterCrop 224), calibración Platt de ISIC tal cual, umbral τ95 de ISIC tal cual. Bootstrap a nivel imagen: DDI no tiene id de paciente (limitación declarada).",
        "",
        "**Dos cambios a la vez, declarados:** dermatoscopía → foto clínica, y melanoma-vs-resto → maligno-vs-benigno (78 diagnósticos). La tarea primaria es un cambio de tarea respecto al entrenamiento.",
        "",
        "| tarea | grupo | n | positivos | AUC-ROC [IC] | AUPRC | sens@τ95 | spec@τ95 |",
        "|:--|:--|--:|--:|:--|--:|--:|--:|",
    ]
    for task, label in (
        ("malignant_vs_benign", "maligno vs benigno (primaria)"),
        ("melanoma_vs_rest", "melanoma vs resto (secundaria)"),
    ):
        t = ddi["tasks"][task]
        for g, v in (("todas", t["all"]), *t["by_group"].items()):
            gl = {"fst_12": "FST I–II", "fst_34": "FST III–IV", "fst_56": "FST V–VI"}.get(g, g)
            out.append(
                f"| {label if g == 'todas' else ''} | {gl} | {v['n']} | {v['positives']} | {fmt(v['auroc'])} [{fmt(v['auroc_lo'])}, {fmt(v['auroc_hi'])}] | {fmt(v['auprc'])} | {fmt(v['at_tau_95']['sensitivity'])} | {fmt(v['at_tau_95']['specificity'])} |"
            )
    c = ddi["calibration_isic_platt_on_ddi"]
    mb = ddi["tasks"]["malignant_vs_benign"]
    out += [
        "",
        f"**Calibración de ISIC aplicada a DDI:** ECE uniforme {c['ece_uniform']['ece']:.3f}, Brier {c['brier']:.3f}. Con prevalencia 15 veces mayor y otro dominio, la calibración no se transfiere: es un resultado, no un error. El umbral τ95 de ISIC refiere casi todo (especificidad {mb['all']['at_tau_95']['specificity']:.3f}): fuera del dominio el punto de operación tampoco vale.",
        "",
        "**Anclaje (verificado en fuente el 2026-09-18, texto completo en PMC9374341):** Daneshjou, R., Vodrahalli, K., Liang, W., Novoa, R. A., Jenkins, M., Rotemberg, V., Ko, J., Swetter, S. M., Bailey, E. E., Gevaert, O., Mukherjee, P., Phung, M., Yekrang, K., Fong, B., Sahasrabudhe, R., Zou, J., Chiou, A. S. (2022). *Disparities in dermatology AI performance on a diverse, curated clinical image set.* Science Advances, 8(31), eabq6147. <https://doi.org/10.1126/sciadv.abq6147>. Sobre las mismas 656 imágenes (208 / 241 / 207 por grupo, con 49 / 74 / 48 malignas), tres algoritmos del estado del arte cayeron de 0.88–0.94 en sus conjuntos originales a «ModelDerm had an ROC-AUC of 0.65 [95% confidence interval (CI), 0.61 to 0.70], DeepDerm had an ROC-AUC of 0.56 (0.51 to 0.61), and HAM10000 had an ROC-AUC of 0.67 (95% CI, 0.62 to 0.71)», con peor desempeño en FST V–VI (por ejemplo HAM10000: 0.72 en FST I–II frente a 0.57 en FST V–VI).",
        "",
        f"**Situación de este modelo:** AUC-ROC {mb['all']['auroc']:.3f} [{mb['all']['auroc_lo']:.3f}, {mb['all']['auroc_hi']:.3f}] en maligno vs benigno, dentro del rango 0.56–0.67 de los tres algoritmos del paper y en el extremo bajo, como corresponde a un modelo entrenado solo en dermatoscopía de ISIC 2020 y sin fotos clínicas. Por tono: {mb['by_group']['fst_12']['auroc']:.3f} / {mb['by_group']['fst_34']['auroc']:.3f} / {mb['by_group']['fst_56']['auroc']:.3f} (I–II / III–IV / V–VI), con intervalos que se traslapan por completo: este modelo **no** muestra la caída en piel oscura que reporta el paper. Con 48–74 positivos por grupo el intervalo de cada AUC mide ±0.09, así que lo que puede decirse es que no hay una diferencia grande, no que no haya ninguna; y un modelo que discrimina poco en todos los grupos tiene poco margen para discriminar peor en uno. Melanoma vs resto (21 positivos, 7 por grupo) no permite concluir nada.",
    ]
    return out


V1_TABLE = """| | v1 (2024) | v2 (este trabajo) |
|---|---|---|
| Conjunto reportado | Validación, contaminada | Test bloqueado, un acceso |
| Prevalencia | 48 % | 1.64 % |
| Métrica principal | Accuracy 90.39 % | AUC-ROC y AUPRC con IC |
| Umbral | 0.5 por omisión | τ95 por bootstrap sobre validación |
| Calibración | Ninguna | Platt sobre validación |
| Fugas verificadas | No | Sí, por paciente |
| Sesgo por tono de piel | No evaluado | DDI, tres grupos |
| Dominio fuera de dermatoscopía | Una foto sin diagnóstico | 656 imágenes con biopsia |"""


def main() -> None:
    cal = json.loads(CAL.read_text())
    test = json.loads(TEST.read_text()) if TEST.exists() else None
    ddi = json.loads(DDI.read_text()) if DDI.exists() else None
    log_lines = LOG.read_text().strip().splitlines() if LOG.exists() else []
    limits = (
        LIMITS.read_text().strip()
        if LIMITS.exists()
        else "_pendiente: las escribe Edgar en `docs/f4_limitations.md`_"
    )
    lines = [
        "# F4 — Evaluación clínica",
        "",
        "Generado por `scripts/f4_report.py`. Modelo: A1 (`tf_efficientnetv2_s.in21k_ft_in1k` a 224 px, semilla 0), checkpoint `melanoma-f3-final`. Nada de esta fase cambia el modelo.",
        "",
        "## 1. Protocolo congelado",
        "",
        f"`docs/f4_protocol.yaml`, commiteado en `{protocol_commit()}`, antes del script de evaluación del test. Copia literal:",
        "",
        "```yaml",
        PROTOCOL.read_text().rstrip(),
        "```",
        "",
        "## 2. Bitácora del acceso al test",
        "",
        *(
            [
                f"```\n{chr(10).join(log_lines)}\n```",
                f"{len(log_lines)} línea(s) en `logs/test_set_access.log` (presupuesto: 2).",
            ]
            if log_lines
            else ["`logs/test_set_access.log` está vacío: el test no se ha abierto."]
        ),
        "",
        "## 3. Calibración sobre validación",
        "",
        *section_calibration(cal),
        "",
        "### Umbrales por bootstrap (F4.2)",
        "",
        *section_thresholds(cal),
        "",
        "## 4. Resultados sobre el test",
        "",
        *section_test(test),
        "",
        "## 5. Punto de operación",
        "",
        *section_operating_point(test),
        "",
        "## 6. Validación vs test",
        "",
        *section_gap(cal, test),
        "",
        "## 7. Subgrupos (exploratorio)",
        "",
        *section_subgroups(test),
        "",
        "## 8. Evaluación externa en DDI",
        "",
        *section_ddi(ddi),
        "",
        "## 9. Comparación con la v1",
        "",
        V1_TABLE,
        "",
        "## 10. Limitaciones",
        "",
        limits,
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"test: {'sí' if test else 'pendiente'} · DDI: {'sí' if ddi else 'pendiente'} · bitácora: {len(log_lines)} líneas → {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
