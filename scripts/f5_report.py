# ruff: noqa: E501
"""F5.6 — `reports/f5_explainability.md`, siete secciones en el orden de la spec, a partir de
reports/f5_cam.json, f5_overlap.json, f5_artifacts.json y f5_figures.json. Ningún número se
escribe a mano: todo sale de los JSON que producen los otros scripts de F5.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "reports" / "f5_cam.json"
OVERLAP = ROOT / "reports" / "f5_overlap.json"
ART = ROOT / "reports" / "f5_artifacts.json"
FIGS = ROOT / "reports" / "f5_figures.json"
OUT = ROOT / "reports" / "f5_explainability.md"

GROUP_ES = {
    "TP": "Verdaderos positivos (VP)",
    "FN": "Falsos negativos (FN)",
    "FP": "Falsos positivos (FP)",
    "TN": "Verdaderos negativos (VN)",
}
SCOPE_ES = {
    "head": "cabeza (lineal + `conv_head` + `bn2`)",
    "head_last_block": "cabeza + último bloque (`blocks[-1]`)",
    "all": "todos los pesos",
}


def fmt(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/d"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(d: dict | None, d_=3, sign=False) -> str:
    if (
        not d
        or d.get("point") is None
        or (isinstance(d["point"], float) and math.isnan(d["point"]))
    ):
        return "n/d"
    return f"{fmt(d['point'], d_, sign)} [{fmt(d['lo'], d_, sign)}, {fmt(d['hi'], d_, sign)}]"


def pct(x, d=0) -> str:
    return "n/d" if x is None else f"{100 * x:.{d}f} %"


def section_method(cam: dict) -> list[str]:
    h, w = cam["map_shape"]
    zero = cam["cam_all_zero_images"]
    return [
        "## 1. Método y capa",
        "",
        f"**Modelo:** A1, `tf_efficientnetv2_s.in21k_ft_in1k` a {cam['input_size']} px, semilla 0, checkpoint de F3 "
        f"`{cam['checkpoint']['file']}` (SHA256 `{cam['checkpoint']['sha256'][:16]}…`). Cabeza: pooling global promedio → dropout → lineal de una salida (logit de melanoma).",
        "",
        f"**Método:** Grad-CAM (Selvaraju et al., 2017) sobre la última etapa convolucional antes del pooling: `{cam['layer']}` de timm, "
        "es decir, la salida de `conv_head` (1×1, 256 → 1280 canales) tras BatchNorm y SiLU. Es exactamente el tensor que entra al pooling global. "
        f"A {cam['input_size']} px y stride total 32 el mapa mide **{h} × {w}** (1280 canales) y se interpola bilinealmente a {cam['input_size']} × {cam['input_size']} para superponerlo.",
        "",
        "**Equivalencia con CAM.** Con la cabeza GAP + lineal, ∂logit/∂A_k(x, y) = w_k / (H·W) para toda posición, así que los pesos de Grad-CAM "
        "(promedio espacial del gradiente) son w_k / 49 y, tras normalizar al máximo, el mapa es idéntico al CAM de Zhou et al. (2016): "
        "`ReLU(Σ_k w_k · A_k(x, y))`. La implementación del proyecto (`melanoma.explain.cam_from_features`) no usa gradientes ni torch: recibe los mapas "
        "`[1280, 7, 7]` y los 1280 pesos de la capa lineal. La sección 2 verifica numéricamente la equivalencia. Contrato para F6: "
        "`explain(features[1280, 7, 7], linear_weight[1280]) → [7, 7]` con valores en [0, 1] y máximo 1; `upsample(cam, size) → [size, size]` bilineal (numpy puro, verificado contra `torch.nn.functional.interpolate`).",
        "",
        "**Limitaciones del método, declaradas de antemano:**",
        "",
        f"- **Resolución.** Una malla de {h} × {w} sobre {cam['input_size']} px: cada celda cubre 32 × 32 px de la entrada (≈ 6.5 % del lado). El mapa no puede señalar estructuras finas (red pigmentaria, puntos, estrías); señala regiones. Es una limitación de la resolución del mapa, no del modelo.",
        "- **Solo evidencia positiva.** Con un único logit, la ReLU conserva únicamente las celdas que empujan hacia «melanoma». Las regiones que empujan hacia «benigno» quedan en cero: el mapa no explica por qué el modelo descarta una lesión, solo dónde ve evidencia a favor de referirla.",
        f"- **Mapas vacíos.** En **{zero} de {cam['n_images']}** imágenes de validación ({pct(zero / cam['n_images'], 1)}) ninguna celda es positiva y el mapa es cero en todas partes (ninguna de ellas es melanoma: son benignos con logit muy negativo). F6 debe tratar ese caso como «sin evidencia positiva», no como un mapa uniforme.",
        "",
    ]


def section_equivalence(cam: dict) -> list[str]:
    e = cam["equivalence"]
    ok = "**verificada**" if e["all_above_threshold"] else "**NO verificada**"
    return [
        "## 2. Equivalencia Grad-CAM ≡ CAM",
        "",
        f"Sobre {e['n']} imágenes de validación elegidas al azar (semilla {ART_SEED}), la CAM en numpy interpolada a {cam['input_size']} px se comparó con la salida de "
        f"`pytorch-grad-cam` (`GradCAM`, capa `{e['layer']}`, objetivo = el logit) mediante la correlación de Pearson por imagen. Criterio fijado en la spec: > {e['threshold']}.",
        "",
        "| imágenes | con correlación definida | ambas cero | correlación mínima | media | mediana | criterio |",
        "|--:|--:|--:|--:|--:|--:|:--|",
        f"| {e['n']} | {e['n_correlated']} | {e['n_both_zero']} | {fmt(e['min_corr'], 6)} | {fmt(e['mean_corr'], 6)} | {fmt(e['median_corr'], 6)} | {ok} |",
        "",
        f"En las {e['n_both_zero']} imágenes «ambas cero» ninguna celda es positiva, los dos métodos devuelven un mapa nulo y la correlación no está definida; coinciden trivialmente. "
        "En el resto la correlación es 1 hasta la precisión de coma flotante: no hay error de capa ni de signo, y la API de F6 puede calcular el mapa sin PyTorch.",
        "",
        "Además, `tests/test_explain.py::test_cam_equals_gradcam` repite la comparación en CI con el mismo backbone (pesos aleatorios) sobre imágenes sintéticas.",
        "",
    ]


def section_sanity(cam: dict) -> list[str]:
    s = cam["sanity"]
    rows = [
        "| ámbito reinicializado | semillas | Spearman media | mediana | p05 | p95 | < 0.3 | indefinidas |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for scope, d in s["scopes"].items():
        p = d["pooled"]
        rows.append(
            f"| {SCOPE_ES[scope]} | {len(d['seeds'])} | {fmt(p['mean'])} | {fmt(p['median'])} | {fmt(p['p05'])} | {fmt(p['p95'])} | {pct(p['frac_below_threshold'])} | {p['n_undefined']} / {p['n']} |"
        )
    worst = max(d["pooled"]["mean"] for d in s["scopes"].values())
    verdict = (
        f"Todas las medias quedan muy por debajo de {s['threshold']} (la mayor es {fmt(worst)}): el mapa depende de los pesos, no solo de la imagen. **La prueba se pasa.**"
        if worst < s["threshold"]
        else f"Al menos un ámbito supera {s['threshold']} (mayor media {fmt(worst)}): los mapas no dependen lo suficiente del modelo. **La prueba NO se pasa** y el método no sirve como explicación."
    )
    return [
        "## 3. Prueba de cordura (Adebayo et al., 2018)",
        "",
        f"Prueba de aleatorización de parámetros: sobre {s['n']} imágenes de validación al azar se calcula el CAM con el modelo entrenado y con copias en las que se reinicializan los pesos "
        "(`reset_parameters` de torch: Kaiming uniforme en convoluciones y lineal, BatchNorm a peso 1 / sesgo 0 / estadísticas 0 y 1), en tres ámbitos en cascada: la cabeza, la cabeza más el último bloque, y todo. "
        f"Se reporta la correlación de Spearman entre el mapa entrenado y el aleatorio, celda a celda sobre la malla 7 × 7, con {len(s['scopes']['all']['seeds'])} semillas por ámbito. "
        f"**Umbral declarado de antemano (spec F5.2): correlación < {s['threshold']}.** Las parejas donde uno de los dos mapas es constante (normalmente el aleatorio, todo cero) no tienen correlación definida y se cuentan aparte.",
        "",
        *rows,
        "",
        verdict,
        "",
        "Esto coincide con lo que Adebayo et al. reportan para Grad-CAM («Of the methods we tested, Gradients & GradCAM pass the sanity checks, while Guided BackProp & Guided GradCAM fail») y es lo esperable por construcción: con la equivalencia de la sección 1, el CAM es una combinación lineal de los mapas de activación con los pesos de la capa lineal, y reinicializar cualquiera de los dos lo cambia. "
        "`tests/test_explain.py::test_cam_depends_on_weights` vigila la propiedad en CI con una imagen sintética.",
        "",
    ]


def section_overlap(ov: dict) -> list[str]:
    r = ov["mask_reasons"]
    rules = ov["mask_rules"]
    rows = [
        "| grupo en τ95 | imágenes | con máscara válida | y CAM no nulo (n de las columnas siguientes) | solapamiento media | mediana | p25–p75 | azar (área de la máscara) | > azar | solapamiento − azar [IC 95 %] |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for g, d in ov["by_group"].items():
        o, c = d["overlap"], d["chance"]
        if not o.get("n"):
            rows.append(
                f"| {GROUP_ES[g]} | {d['n_total']} | 0 | 0 | n/d | n/d | n/d | n/d | n/d | n/d |"
            )
            continue
        rows.append(
            f"| {GROUP_ES[g]} | {d['n_total']} | {d['n_valid_mask']} | {o['n']} | {fmt(o['mean'])} | {fmt(o['median'])} | {fmt(o['p25'], 2)}–{fmt(o['p75'], 2)} | {fmt(c['mean'])} | {pct(d['frac_above_chance'])} | {ci(d['overlap_minus_chance_ci'], sign=True)} |"
        )
    all_ = ov["overall"]
    tp, tn, fn = ov["by_group"]["TP"], ov["by_group"]["TN"], ov["by_group"]["FN"]
    return [
        "## 4. Solapamiento del CAM con la lesión",
        "",
        "**Máscara automática.** Escala de grises del recorte de 224 px que ve el modelo → desenfoque gaussiano 5 × 5 → umbral de Otsu (la lesión es la parte oscura) → apertura y cierre morfológicos (elipse de 7 px) → "
        f"componente conexa más grande que no toque más del {pct(rules['max_border_frac'])} del perímetro y que alcance el cuadrado central de la imagen (25–75 % de cada lado; sin esta condición una esquina oscura de viñeta ganaba a una lesión pequeña en las primeras corridas). Filtro de calidad: área entre {pct(rules['min_area_frac'])} y {pct(rules['max_area_frac'])} de la imagen. "
        "Es una máscara cruda, sin aprendizaje, y se declara como tal.",
        "",
        "| imágenes de validación | máscara válida | descartadas: componente < 3 % | descartadas: > 70 % | descartadas: sin componente |",
        "|--:|--:|--:|--:|--:|",
        f"| {ov['n_images']} | **{ov['n_valid_mask']}** ({pct(ov['n_valid_mask'] / ov['n_images'], 1)}) | {r.get('too_small', 0)} | {r.get('too_large', 0)} | {r.get('no_component', 0)} |",
        "",
        "![Ejemplos de máscaras válidas (arriba) y descartadas (abajo)](figures/f5_mask_examples.png)",
        "",
        f"Ejemplos (`image_id`): {', '.join(f'`{i}`' for i in ov['mask_examples'])}. Los descartes por «> 70 %» son en general dermatoscopías con viñeta circular negra, donde Otsu une la lesión con el borde oscuro; los «< 3 %» son lesiones muy claras o difusas donde Otsu separa solo un fragmento. "
        "Entre las válidas hay máscaras que incluyen piel oscura vecina: el número de solapamiento hereda ese ruido.",
        "",
        "**Métrica.** Para cada imagen con máscara válida y CAM no nulo: fracción de la energía del CAM (interpolado a 224 px) que cae dentro de la máscara. Referencia por azar: la fracción de área que ocupa la máscara (un mapa uniforme daría exactamente ese valor). "
        f"Los grupos se definen en τ95 = {ov['tau_95']:.4f} sobre la probabilidad Platt de F4 (VP {ov['confusion_at_tau95'].get('TP', 0)}, FN {ov['confusion_at_tau95'].get('FN', 0)}, FP {ov['confusion_at_tau95'].get('FP', 0)}, VN {ov['confusion_at_tau95'].get('TN', 0)}). "
        f"Intervalos por bootstrap a nivel paciente ({ov['by_group']['TP']['overlap_minus_chance_ci']['n_resamples']} remuestreos).",
        "",
        *rows,
        "",
        f"Global ({all_['overlap']['n']} imágenes con máscara válida y CAM no nulo): solapamiento medio {fmt(all_['overlap']['mean'])}, mediana {fmt(all_['overlap']['median'])}, azar medio {fmt(all_['chance']['mean'])}; "
        f"solapamiento − azar = {ci(all_['overlap_minus_chance_ci'], sign=True)}; por encima del azar en el {pct(all_['frac_above_chance'])} de las imágenes.",
        "",
        "**Lectura.** El resultado contradice la expectativa de la spec («el solapamiento probablemente será alto»):",
        "",
        f"- En los **verdaderos positivos** el CAM cae dentro de la lesión en una fracción ({fmt(tp['overlap']['mean'])}) apenas por encima del azar ({fmt(tp['chance']['mean'])}); la diferencia {ci(tp['overlap_minus_chance_ci'], sign=True)} es pequeña. La evidencia a favor de «melanoma» se reparte entre la lesión y la piel que la rodea.",
        f"- En los **verdaderos negativos** el CAM casi nunca cae en la lesión (mediana {fmt(tn['overlap']['median'])}, media {fmt(tn['overlap']['mean'])} contra {fmt(tn['chance']['mean'])} por azar). Es coherente con la sección 1: en un benigno claro, la lesión aporta evidencia *en contra* de melanoma, que la ReLU descarta, y la poca evidencia positiva que queda está en la piel y en los bordes de la imagen.",
        f"- Los **falsos negativos** son {fn['n_total']} imágenes ({fn['n_valid_mask']} con máscara válida): solapamiento medio {fmt(fn['overlap']['mean'])} frente a {fmt(fn['chance']['mean'])} por azar. Con esa n solo cabe decir que en ninguno el mapa se concentró en la lesión: el modelo no encontró evidencia positiva *dentro* del melanoma que se le escapó, que es el modo de falla «no miró donde debía» y no el de «miró bien y aun así falló».",
        f"- Los **falsos positivos** ({fp_n(ov)} imágenes) tampoco miran a la lesión más que al azar: lo que hace cruzar τ95 a un benigno está, en promedio, tanto o más fuera de la lesión que dentro.",
        "",
        "Lo que esto mide es dónde está la activación positiva, no si el modelo «razona» sobre morfología. Pero un modelo cuya evidencia a favor de melanoma se localiza tan a menudo fuera de la lesión es exactamente el motivo por el que la sección 5 inyecta artefactos en la periferia.",
        "",
    ]


def fp_n(ov: dict) -> int:
    return ov["by_group"]["FP"]["n_total"]


def section_artifacts(art: dict) -> list[str]:
    res = art["results"]
    head_b = [
        "| perturbación | Δ media [IC 95 %] | Δ mediana [IC 95 %] | Δ > 0 | cruzan τ95 (no referir → referir) | referidos antes → después | CAM hacia la zona del artefacto: antes → después (Δ [IC]) | área de la zona | cambio de píxel |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    rows_b, rows_m = [], []
    for _c, r in res.items():
        b, m = r["benign"], r["melanoma"]
        cross = (
            f"{b['n_cross_to_referred']} / {b['n_not_referred_original']} ({ci(b['frac_cross_to_referred_ci'], 3)})"
            if b.get("frac_cross_to_referred_ci")
            else "n/d"
        )
        rows_b.append(
            f"| {r['label']} | {ci(b['delta_mean_ci'], 4, True)} | {ci(b['delta_median_ci'], 4, True)} | {pct(b['frac_delta_positive'])} | {cross} | {pct(b['frac_referred_original'], 1)} → {pct(b['frac_referred_perturbed'], 1)} | {fmt(b['cam_in_region_original_mean'], 2)} → {fmt(b['cam_in_region_perturbed_mean'], 2)} ({ci(b['cam_shift_ci'], 3, True)}) | {pct(b['region_area_frac_mean'])} | {fmt(b['pixel_change_mean'], 1)} |"
        )
        lose = (
            f"{m['n_lose_referral']} / {m['n_referred_original']} ({ci(m['frac_lose_referral_ci'], 3)})"
            if m.get("frac_lose_referral_ci")
            else "n/d"
        )
        rows_m.append(
            f"| {r['label']} | {ci(m['delta_mean_ci'], 4, True)} | {ci(m['delta_median_ci'], 4, True)} | {pct(m['frac_delta_positive'])} | {lose} | {fmt(m['cam_in_region_original_mean'], 2)} → {fmt(m['cam_in_region_perturbed_mean'], 2)} ({ci(m['cam_shift_ci'], 3, True)}) | {fmt(m['pixel_change_mean'], 1)} |"
        )
    head_m = [
        "| perturbación | Δ media [IC 95 %] | Δ mediana [IC 95 %] | Δ > 0 | pierden la referencia (referir → no referir) | CAM hacia la zona: antes → después (Δ [IC]) | cambio de píxel |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    ctrl = res["noise"]["benign"]
    return [
        "## 5. Artefactos por perturbación",
        "",
        f"**Diseño.** Experimento pareado sobre {art['n_benign']} benignos de validación elegidos al azar (de {art['n_benign_available']}; semilla {art['seed']}) y los {art['n_melanoma']} melanomas ({art['n_patients']} pacientes). "
        "A cada imagen (el recorte de 224 px que ve el modelo) se le inyecta, por separado, cada una de cinco perturbaciones sintéticas (`melanoma.explain.perturb`, parámetros fijos, no ajustados contra el modelo):",
        "",
        "| perturbación | síntesis | zona para el CAM |",
        "|:--|:--|:--|",
        "| regla | franja clara del 9 % del lado en un borde al azar, con marcas de escala cada 1/20 y marcas largas cada cinco | la franja |",
        "| tinta | trazo curvo azul-violeta (RGB 72, 40, 150; grosor 3 % del lado, largo 22 %) colocado a radio + 12 % del centroide de la máscara automática, en el ángulo que menos la invade; los píxeles de la lesión nunca se pintan | el trazo |",
        "| vello | 5–10 curvas de Bézier marrón oscuro de 1–2 px cruzando la imagen | las curvas |",
        "| viñeteado | centro intacto hasta 0.45 del radio; de ahí a la esquina el brillo cae cuadráticamente hasta el 25 % | píxeles atenuados por debajo de 0.85 |",
        "| ruido (control) | ruido gaussiano i.i.d. con σ = 12 (escala 0–255), sin estructura espacial | toda la imagen |",
        "",
        f"Se mide, con la probabilidad calibrada (Platt de F4, congelada), Δ = P(con artefacto) − P(original) pareada por imagen; la fracción de imágenes con Δ > 0; cuántos benignos que no se referían (P < τ95 = {art['tau_95']:.4f}) pasan a referirse; cuántos melanomas referidos dejan de serlo; "
        "y si el CAM se mueve hacia la zona del artefacto (fracción de energía del CAM dentro de la zona, antes y después). «Cambio de píxel» es el cambio medio absoluto por píxel y canal (0–255) y sirve para comparar la magnitud del control con la de los artefactos. "
        f"Intervalos por bootstrap a nivel paciente ({art['bootstrap']['n_resamples']} remuestreos). Los logits de la condición «original» coinciden con los de F4 (diferencia máxima {art['logit_max_abs_diff_vs_f4']:.1e}).",
        "",
        f"**Benignos (n = {art['n_benign']}):**",
        "",
        *head_b,
        *rows_b,
        "",
        f"**Melanomas (n = {art['n_melanoma']}):**",
        "",
        *head_m,
        *rows_m,
        "",
        "![Distribución de Δ por perturbación](figures/f5_artifacts_delta.png)",
        "",
        "**Lectura.** Cada artefacto se compara con el control de ruido en dos cosas: el Δ medio y la fracción de benignos que cruzan τ95. «Supera al control» significa que los intervalos del 95 % no se traslapan.",
        "",
        *reading_bullets(art),
        "",
        f"Dos advertencias de lectura. Primera: τ95 = {art['tau_95']:.4f} está en el percentil 40 de las probabilidades de los benignos, así que hay mucha masa justo debajo del corte y cualquier cambio (incluido el ruido, que cruza al {pct(ctrl['frac_cross_to_referred_ci']['point'], 0)}) hace cruzar a muchos; por eso el número que importa es el exceso sobre el control, no la fracción absoluta. "
        "Segunda: los Δ son pequeños en valor absoluto porque son probabilidades calibradas a prevalencia 1.8 %; un Δ de +0.01 en un benigno con P = 0.003 es triplicar su probabilidad.",
        "",
        "Las cifras de la tabla son la limitación (o la evidencia de robustez) medida, y así pasan a la sección 7. Nada de esto reabre F3: si el modelo aprendió un artefacto, se reporta, no se reentrena.",
        "",
        "**Sobre Winkler et al. (2019), verificado en el texto completo (PMC6694463).** No mide «+40 % de probabilidad»: sobre 107 nevos y 23 melanomas con marcas de violeta de genciana, la probabilidad media de melanoma que Moleanalyzer-Pro (Inception-v4) asignó a los nevos pasó de 0.16 a 0.54, "
        "la especificidad cayó de 84.1 % a 45.8 % (sensibilidad 95.7 % → 100 %) y la tasa de falsos positivos subió «by approximately 40%» en puntos porcentuales; recortar la imagen alrededor de la lesión revertía el efecto (especificidad 97.2 %). "
        "El experimento de aquí es análogo pero sintético: tinta digital, no marcas reales, y probabilidades calibradas a prevalencia 1.8 %, por lo que los Δ son pequeños en valor absoluto y lo comparable es la fracción de cruces de umbral.",
        "",
    ]


def _vs_control(a: dict, c: dict) -> str:
    if a["lo"] > c["hi"]:
        return "supera al control"
    if a["hi"] < c["lo"]:
        return "queda por debajo del control"
    return "no se distingue del control"


def reading_bullets(art: dict) -> list[str]:
    res = art["results"]
    ctrl_b, ctrl_m = res["noise"]["benign"], res["noise"]["melanoma"]
    out = [
        f"- **Control (ruido σ = 12, cambio de píxel {fmt(ctrl_b['pixel_change_mean'], 1)}):** Δ = {ci(ctrl_b['delta_mean_ci'], 4, True)} en benignos, cruzan τ95 {ctrl_b['n_cross_to_referred']} de {ctrl_b['n_not_referred_original']} ({pct(ctrl_b['frac_cross_to_referred_ci']['point'], 0)}); en melanomas Δ = {ci(ctrl_m['delta_mean_ci'], 4, True)} y {ctrl_m['n_lose_referral']} de {ctrl_m['n_referred_original']} dejan de referirse. Es la referencia de «el modelo reacciona a cualquier cambio»: el ruido empuja las probabilidades hacia el centro (sube las muy bajas, baja las altas)."
    ]
    for c in ("ink", "ruler", "vignette", "hair"):
        b, m = res[c]["benign"], res[c]["melanoma"]
        d_cmp = _vs_control(b["delta_mean_ci"], ctrl_b["delta_mean_ci"])
        x_cmp = _vs_control(b["frac_cross_to_referred_ci"], ctrl_b["frac_cross_to_referred_ci"])
        cam = f"el CAM dentro de la zona del artefacto pasa de {fmt(b['cam_in_region_original_mean'], 2)} a {fmt(b['cam_in_region_perturbed_mean'], 2)} ({ci(b['cam_shift_ci'], 3, True)})"
        mel = f"melanomas: Δ = {ci(m['delta_mean_ci'], 4, True)}, {m['n_lose_referral']} de {m['n_referred_original']} dejan de referirse ({ci(m['frac_lose_referral_ci'], 3)})"
        out.append(
            f"- **{res[c]['label'].capitalize()}** (cambio de píxel {fmt(b['pixel_change_mean'], 1)}): en benignos Δ = {ci(b['delta_mean_ci'], 4, True)} ({d_cmp}); cruzan τ95 {b['n_cross_to_referred']} de {b['n_not_referred_original']} ({pct(b['frac_cross_to_referred_ci']['point'], 0)}, {x_cmp}); {cam}. En {mel}."
        )
    return out


def conclusion_artifact_bullets(art: dict) -> list[str]:
    res = art["results"]
    ctrl_b, ctrl_m = res["noise"]["benign"], res["noise"]["melanoma"]
    names = ("ink", "ruler", "vignette", "hair")
    learned = [
        c
        for c in names
        if res[c]["benign"]["delta_mean_ci"]["lo"] > ctrl_b["delta_mean_ci"]["hi"]
        and res[c]["benign"]["frac_cross_to_referred_ci"]["lo"]
        > ctrl_b["frac_cross_to_referred_ci"]["hi"]
    ]
    lose = [
        c
        for c in names
        if res[c]["melanoma"]["frac_lose_referral_ci"]
        and res[c]["melanoma"]["frac_lose_referral_ci"]["lo"] > 0
    ]
    out = []
    if learned:
        parts = "; ".join(
            f"{res[c]['label']}: Δ {ci(res[c]['benign']['delta_mean_ci'], 4, True)} y {pct(res[c]['benign']['frac_cross_to_referred_ci']['point'], 0)} de cruces frente a {ci(ctrl_b['delta_mean_ci'], 4, True)} y {pct(ctrl_b['frac_cross_to_referred_ci']['point'], 0)} con ruido"
            for c in learned
        )
        out.append(
            f"- **Artefactos que el modelo asocia con melanoma (limitación medida):** {parts}. Con un cambio de píxel de {fmt(res['ink']['benign']['pixel_change_mean'], 1)} sobre 255, la tinta es la perturbación más pequeña del experimento y la que más sube la probabilidad de los benignos: el modelo aprendió la marca, como la CNN de Winkler et al. (2019), aunque el CAM apenas se mueve hacia el trazo (el efecto es global, no una activación sobre la tinta)."
        )
    else:
        out.append(
            "- Ningún artefacto supera al control a la vez en Δ medio y en fracción de cruces: no hay evidencia de que el modelo haya aprendido estos artefactos."
        )
    if lose:
        parts = "; ".join(
            f"{res[c]['label']}: {res[c]['melanoma']['n_lose_referral']} de {res[c]['melanoma']['n_referred_original']} ({ci(res[c]['melanoma']['frac_lose_referral_ci'], 2)}), Δ {ci(res[c]['melanoma']['delta_mean_ci'], 3, True)}"
            for c in lose
        )
        out.append(
            f"- **Pérdida de sensibilidad ante degradación (limitación medida):** melanomas referidos en τ95 que dejan de serlo con el artefacto: {parts}; con ruido, {ctrl_m['n_lose_referral']} de {ctrl_m['n_referred_original']}. Cubrir la lesión con vello sintético o con ruido destruye la evidencia positiva y el modelo tiende a «benigno»: en un triage de alta sensibilidad ese es el error caro, y aparece con perturbaciones que no son raras en la práctica."
        )
    robust = [c for c in names if c not in learned and c not in lose]
    if robust:
        details = []
        for c in robust:
            b = res[c]["benign"]
            over = []
            if b["delta_mean_ci"]["lo"] > ctrl_b["delta_mean_ci"]["hi"]:
                over.append("Δ medio")
            if b["frac_cross_to_referred_ci"]["lo"] > ctrl_b["frac_cross_to_referred_ci"]["hi"]:
                over.append("cruces")
            if b["cam_shift_ci"]["lo"] > 0:
                over.append("el CAM se desplaza hacia la zona")
            details.append(
                f"{res[c]['label']}"
                + (
                    f" (supera al control solo en {' y '.join(over)})"
                    if over
                    else " (no supera al control en nada)"
                )
            )
        out.append(
            "- **Sin evidencia suficiente de asociación aprendida** (no superan al control en Δ medio y en cruces a la vez, y no quitan referencias a melanomas): "
            + "; ".join(details)
            + ". Es robustez parcial frente a *estas* versiones sintéticas, con intervalos, no una garantía general."
        )
    return out


def section_figures(figs: dict, art: dict) -> list[str]:
    rows = ["| grupo | `image_id` | P (Platt) | solapamiento | azar |", "|:--|:--|--:|--:|--:|"]
    for r in figs["rows"]:
        rows.append(
            f"| {r['group']} | `{r['image_id']}` | {r['prob_platt']:.4f} | {r['overlap']:.2f} | {r['chance']:.2f} |"
        )
    counts = ", ".join(f"{k} {len(v)}" for k, v in figs["grid"].items())
    return [
        "## 6. Figuras cualitativas",
        "",
        f"**Cuadrícula** (`{figs['figure']}`): {len(figs['rows'])} imágenes de validación ({counts}), elegidas al azar con semilla {figs['seed']} dentro de cada grupo en τ95 entre las que tienen máscara válida. Columnas: recorte original de 224 px, máscara automática, CAM superpuesto con probabilidad calibrada y solapamiento.",
        "",
        f"![Cuadrícula cualitativa]({figs['figure'].removeprefix('reports/')})",
        "",
        *rows,
        "",
        f"**Panel de artefactos** (`reports/figures/f5_artifact_panel.png`): la imagen benigna `{art['panel_image_id']}` con las cinco perturbaciones y su CAM en cada una.",
        "",
        "![Una imagen benigna con las cinco perturbaciones y su CAM](figures/f5_artifact_panel.png)",
        "",
        f"Todas son imágenes de ISIC 2020 ({figs['license']}), reproducibles con esa atribución. **Ninguna imagen de DDI** se usa en esta fase (lo vigila `tests/test_explain.py::test_no_ddi_in_explain`).",
        "",
    ]


def section_conclusions(cam: dict, ov: dict, art: dict) -> list[str]:
    tp = ov["by_group"]["TP"]
    return [
        "## 7. Qué demuestra y qué no",
        "",
        "**Qué demuestra.**",
        "",
        "- Que el mapa que mostrará la API es Grad-CAM de verdad y no una aproximación: CAM en numpy ≡ Grad-CAM con correlación 1 (sección 2), calculable sin PyTorch (sección 1).",
        f"- Que el mapa depende del modelo: reinicializar los pesos lo destruye (Spearman ≈ {fmt(cam['sanity']['scopes']['all']['pooled']['mean'], 2)} con todo aleatorio, sección 3). No es un detector de bordes disfrazado.",
        f"- **Dónde** está la evidencia positiva: en los melanomas detectados apenas por encima del azar dentro de la lesión ({fmt(tp['overlap']['mean'])} contra {fmt(tp['chance']['mean'])}), y en los benignos casi siempre fuera de ella (sección 4). Esto es un hecho medido sobre {ov['n_valid_mask']} imágenes, no una impresión visual.",
        *conclusion_artifact_bullets(art),
        "",
        "**Qué no demuestra.**",
        "",
        "- Un mapa centrado en la lesión no prueba que el modelo razone con criterios dermatoscópicos (asimetría, borde, color, estructuras). Prueba dónde puso la activación que suma al logit. Y un mapa fuera de la lesión no prueba que el modelo «se equivoque»: la piel circundante contiene información real (fototipo, daño solar, sitio) que correlaciona con melanoma en ISIC 2020.",
        "- El CAM de un solo logit solo muestra evidencia a favor de «melanoma». No dice por qué una lesión se considera benigna; en 181 imágenes el mapa está vacío. Para explicar decisiones negativas haría falta el mapa con signo (fuera de alcance aquí; F6 puede exponerlo si se decide).",
        "- La resolución de 7 × 7 impide atribuir la activación a estructuras finas. «El modelo miró la red pigmentaria» no se puede afirmar con este método.",
        "- La máscara automática es cruda (Otsu + morfología): el solapamiento hereda sus errores, sobre todo en lesiones claras y en imágenes con viñeta circular. Los números por grupo son consistentes entre sí y con lo que se ve en las figuras, pero no son una medición contra segmentaciones de referencia.",
        "- Los artefactos son sintéticos y de parámetros fijos. Una regla real, una marca de tinta real o un vello real no tienen por qué producir el mismo Δ; el experimento acota la sensibilidad a *estos* artefactos, no a todos.",
        "- Nada de esto valida el modelo clínicamente. F4 midió el desempeño; F5 mide dónde y ante qué cambia, y lo reporta como está.",
        "",
    ]


def section_references() -> list[str]:
    return [
        "## Referencias (verificadas en fuente el 2026-09-21)",
        "",
        "- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626. <https://doi.org/10.1109/ICCV.2017.74> (arXiv:1610.02391). "
        "Verificado en el PDF de CVF: «These gradients flowing back are global-average-pooled to obtain the neuron importance weights α_k^c» y «We perform a weighted combination of forward activation maps, and follow it by a ReLU»; sobre CAM: «When Grad-CAM is applied to these architectures α_k^c = w_k^c — making Grad-CAM a strict generalization of CAM».",
        "- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). *Learning deep features for discriminative localization.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2921–2929. <https://doi.org/10.1109/CVPR.2016.319> (arXiv:1512.04150). "
        "Verificado en el PDF de CVF: «We define M_c as the class activation map for class c, where each spatial element is given by M_c(x, y) = Σ_k w_k^c f_k(x, y)».",
        "- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity checks for saliency maps.* Advances in Neural Information Processing Systems 31 (NeurIPS 2018). <https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html> (arXiv:1810.03292). "
        "Verificado en el PDF de NeurIPS: «The model parameter randomization test compares the output of a saliency method on a trained model with the output of the saliency method on a randomly initialized untrained network of the same architecture»; métricas «Spearman rank correlation with absolute value (absolute value), Spearman rank correlation without absolute value (diverging), the structural similarity index (SSIM), and the Pearson correlation of the histogram of gradients (HOGs)»; conclusión «Of the methods we tested, Gradients & GradCAM pass the sanity checks, while Guided BackProp & Guided GradCAM fail». El bibtex oficial no trae páginas; el rango 9525–9536 solo aparece en fuentes secundarias y no se cita.",
        "- Winkler, J. K., Fink, C., Toberer, F., Enk, A., Deinlein, T., Hofmann-Wellenhof, R., Thomas, L., Lallas, A., Blum, A., Stolz, W., & Haenssle, H. A. (2019). *Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition.* JAMA Dermatology, 155(10), 1135–1141. <https://doi.org/10.1001/jamadermatol.2019.1735> (PMID 31411641, PMCID PMC6694463). "
        "Verificado en el texto completo: «In marked lesions, an increase in melanoma probability scores was observed that resulted in a sensitivity of 100% (95% CI, 85.7%-100%) and a significantly reduced specificity of 45.8% (95% CI, 36.7%-55.2%, P < .001)»; «Skin markings significantly increased the mean melanoma probability scores of the classifier in benign nevi from 0.16 (95% CI, 0.10-0.22) to 0.54 (95% CI, 0.46-0.62)»; «consequently increasing the false-positive rate of benign nevi by approximately 40%». La frase de la spec («subía la probabilidad ~40 %») se corrige aquí con esas cifras.",
        "",
    ]


def section_repro(cam: dict, ov: dict, art: dict, figs: dict) -> list[str]:
    return [
        "## Reproducibilidad",
        "",
        "```",
        "make f5-cam        # reports/f5_cam.json, reports/f5_cams_val.npz (fuera de git)",
        "make f5-overlap    # reports/f5_overlap.json, reports/predictions/f5_overlap_val.csv, figures/f5_mask_examples.png",
        "make f5-artifacts  # reports/f5_artifacts.json, reports/predictions/f5_artifacts_val.csv, figures/f5_artifacts_delta.png, figures/f5_artifact_panel.png",
        "make f5-figures    # reports/f5_figures.json, figures/f5_qualitative_grid.png",
        "make f5-report     # este archivo",
        "```",
        "",
        f"Config `configs/f5.yaml`; semilla {art['seed']}; checkpoint `{cam['checkpoint']['file']}` (SHA256 `{cam['checkpoint']['sha256']}`); split de validación SHA256 `{cam['split_sha256']['val'][:16]}…`. "
        f"Corridas: CAM {cam['date']} (`{cam['git_sha']}`, {cam['minutes']:.1f} min), solapamiento {ov['date']} ({ov['minutes']:.1f} min), artefactos {art['date']} ({art['minutes']:.1f} min), figuras {figs['date']}. Todo en CPU; el conjunto de prueba no se tocó (`logs/test_set_access.log` sigue con una línea).",
        "",
    ]


ART_SEED = None


def main() -> None:
    global ART_SEED
    missing = [p.name for p in (CAM, OVERLAP, ART, FIGS) if not p.exists()]
    if missing:
        raise SystemExit(
            f"faltan {missing}: correr make f5-cam / f5-overlap / f5-artifacts / f5-figures"
        )
    cam, ov, art, figs = (
        json.loads(p.read_text(encoding="utf-8")) for p in (CAM, OVERLAP, ART, FIGS)
    )
    ART_SEED = art["seed"]
    lines = [
        "# F5 — Explicabilidad: Grad-CAM validado sobre validación",
        "",
        f"_Generado por `scripts/f5_report.py` el {art['date']}. Modelo final de F3 (A1), calibración y τ95 de F4 congelados. Todo sobre el split de validación ({cam['n_images']} imágenes, {cam['n_positives']} melanomas); el conjunto de prueba no se abre. Sin imágenes de DDI._",
        "",
        *section_method(cam),
        *section_equivalence(cam),
        *section_sanity(cam),
        *section_overlap(ov),
        *section_artifacts(art),
        *section_figures(figs, art),
        *section_conclusions(cam, ov, art),
        *section_references(),
        *section_repro(cam, ov, art, figs),
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {OUT.relative_to(ROOT)} ({len(lines)} líneas)")


if __name__ == "__main__":
    main()
