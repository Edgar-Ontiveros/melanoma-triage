"""F1.3 — Deduplicación y ``reports/dedup_report.md``.

Pasada 1: duplicados exactos por SHA256. Pasada 2: pHash con umbral de Hamming
(``data.phash.threshold``). Clasifica cada grupo en intra-paciente o cruzado, contrasta con
la lista oficial de duplicados de ISIC y guarda:

- ``reports/dedup_report.md``
- ``reports/figures/phash_distance_hist.png`` (distribución de distancias por pares)
- ``reports/figures/dedup_intra.jpg`` / ``dedup_cross.jpg`` (hojas de contactos)
- ``reports/dedup_pairs.csv`` (pares ≤ ``threshold``)
- ``reports/dedup_cross_groups.json`` (lo que consume ``make_splits.py``)

Uso: ``uv run python scripts/dedup_report.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from melanoma.data.dedup import (
    classify_groups,
    compute_phashes,
    contact_sheet,
    exact_duplicate_groups,
    group_contact_sheet,
    groups_from_pairs,
    pairs_table,
    pairwise_hamming,
)
from melanoma.data.manifest import read_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _hist_png(hist: np.ndarray, threshold: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(np.arange(len(hist)), hist, width=0.9, color="#4c72b0")
    ax.axvline(threshold + 0.5, color="#c44e52", linestyle="--", label=f"umbral = {threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("distancia de Hamming entre pHash (64 bits)")
    ax.set_ylabel("pares (log)")
    ax.set_title("Distribución de distancias por pares, ISIC 2020")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _md_groups(groups: list[list[str]], limit: int = 30) -> str:
    if not groups:
        return "_ninguno_"
    lines = [f"- {', '.join(g)}" for g in groups[:limit]]
    if len(groups) > limit:
        lines.append(f"- … y {len(groups) - limit} grupos más (ver `reports/dedup_pairs.csv`)")
    return "\n".join(lines)


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    reports = Path(d.reports_dir)
    figures = Path(d.figures_dir)
    reports.mkdir(parents=True, exist_ok=True)

    m = read_manifest(d.manifest_path)
    ids = m["image_id"].tolist()
    patient_of = dict(zip(m["image_id"], m["patient_id"], strict=True))
    target_of = dict(zip(m["image_id"], m["target"].astype(int), strict=True))
    path_of = {
        i: Path(d.root) / rp for i, rp in zip(m["image_id"], m["relpath_original"], strict=True)
    }

    # Pasada 1: exactos
    exact = exact_duplicate_groups(m)
    exact_intra, exact_cross = classify_groups(exact, patient_of)

    # Pasada 2: pHash
    t0 = time.time()
    cache = reports / "phash_cache.npz"
    hashes = None
    if d.phash.cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if z["ids"].tolist() == ids and int(z["hash_size"]) == d.phash.hash_size:
            hashes = z["hashes"]
            print(f"pHash reutilizados de {cache}")
    if hashes is None:
        hashes = compute_phashes(
            [path_of[i] for i in ids], d.phash.hash_size, d.phash.decode_size, d.hashing.workers
        )
        np.savez(cache, ids=np.array(ids), hashes=hashes, hash_size=d.phash.hash_size)
    t_hash = time.time() - t0
    t0 = time.time()
    hist, pairs = pairwise_hamming(hashes, d.phash.report_max_distance)
    t_pairs = time.time() - t0

    table = pairs_table(pairs, ids, patient_of, target_of)
    sha_of = dict(zip(m["image_id"], m["sha256_original"], strict=True))
    table["byte_identical"] = table["image_a"].map(sha_of) == table["image_b"].map(sha_of)
    # Solo se versionan los pares dentro del umbral; los demás quedan resumidos por distancia.
    table[table["distance"] <= d.phash.threshold].to_csv(reports / "dedup_pairs.csv", index=False)
    near = pairs[pairs["distance"] <= d.phash.threshold]
    groups = groups_from_pairs(near, len(ids), ids)
    intra, cross = classify_groups(groups, patient_of)
    near_table = table[table["distance"] <= d.phash.threshold]

    # Contraste con la lista oficial de duplicados de ISIC
    official = pd.read_csv(d.official_duplicates_csv)
    official_pairs = {
        frozenset(p) for p in zip(official.iloc[:, 0], official.iloc[:, 1], strict=True)
    }
    found_pairs = {
        frozenset(p) for p in zip(near_table["image_a"], near_table["image_b"], strict=True)
    }
    official_hit = len(official_pairs & found_pairs)
    idx = {i: k for k, i in enumerate(ids)}
    official_dist = [
        int(bin(int(hashes[idx[a]]) ^ int(hashes[idx[b]])).count("1"))
        for a, b in zip(official.iloc[:, 0], official.iloc[:, 1], strict=True)
    ]
    official_dist = pd.Series(official_dist)
    official_cross = sum(
        patient_of[a] != patient_of[b]
        for a, b in zip(official.iloc[:, 0], official.iloc[:, 1], strict=True)
    )

    # Hojas de contactos: primero los pares que NO son byte-idénticos (los informativos)
    k = d.phash.contact_sheet_pairs
    sheet_order = near_table.sort_values(["byte_identical", "distance", "image_a"])
    contact_sheet(
        sheet_order[sheet_order["same_patient"]].head(k),
        path_of,
        figures / "dedup_intra.jpg",
        d.phash.thumb_px,
    )
    contact_sheet(
        sheet_order[~sheet_order["same_patient"]].head(k),
        path_of,
        figures / "dedup_cross.jpg",
        d.phash.thumb_px,
    )
    _hist_png(hist, d.phash.threshold, figures / "phash_distance_hist.png")

    with open(reports / "dedup_cross_groups.json", "w", encoding="utf-8") as f:
        json.dump(
            {"threshold": d.phash.threshold, "cross_groups": cross, "intra_groups": intra},
            f,
            indent=1,
        )

    # Histograma en tabla (hasta report_max_distance) y acumulado
    hist_rows = [
        {"distancia": k_, "pares": int(hist[k_]), "acumulado": int(hist[: k_ + 1].sum())}
        for k_ in range(d.phash.report_max_distance + 1)
    ]
    n_images_in_groups = len({i for g in groups for i in g})

    # ---- Barrido de umbrales (tarea de calibración)
    sweep_rows = []
    sweep_groups: dict[int, list[list[str]]] = {}
    for t in d.phash.sweep_thresholds:
        g_t = groups_from_pairs(pairs[pairs["distance"] <= t], len(ids), ids)
        intra_t, cross_t = classify_groups(g_t, patient_of)
        sizes = sorted((len(g) for g in g_t), reverse=True)
        sweep_groups[t] = g_t
        sweep_rows.append(
            {
                "umbral": t,
                "pares": int((pairs["distance"] <= t).sum()),
                "grupos": len(g_t),
                "grupo_mayor": sizes[0] if sizes else 0,
                "grupos_gt_10": sum(1 for x in sizes if x >= d.phash.sweep_max_group),
                "intra_paciente": len(intra_t),
                "cruzados": len(cross_t),
                "imagenes": sum(sizes),
                "pct_dataset": round(100 * sum(sizes) / len(ids), 2),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(reports / "dedup_sweep.csv", index=False)
    ok_rows = sweep[sweep["grupo_mayor"] < d.phash.sweep_max_group]
    t_vis = int(ok_rows["umbral"].max()) if len(ok_rows) else int(sweep["umbral"].min())
    rng = np.random.default_rng(d.phash.sweep_seed)
    k_g = d.phash.sweep_sample_groups

    def _sample(gs: list[list[str]], n: int) -> list[list[str]]:
        if len(gs) <= n:
            return gs
        return [gs[i] for i in sorted(rng.choice(len(gs), size=n, replace=False))]

    def _is_exact_pair(g: list[str]) -> bool:
        return len(g) == 2 and sha_of[g[0]] == sha_of[g[1]]

    sheet_specs = []  # (nombre, descripción, grupos)
    sheet_specs.append(
        (
            f"dedup_groups_t{t_vis}.jpg",
            f"{k_g} grupos al azar entre TODOS los del umbral {t_vis}",
            _sample(sweep_groups[t_vis], k_g),
        )
    )
    informative = [g for g in sweep_groups[t_vis] if not _is_exact_pair(g)]
    sheet_specs.append(
        (
            f"dedup_groups_t{t_vis}_no_exactos.jpg",
            f"{k_g} grupos al azar del umbral {t_vis} excluyendo pares byte-idénticos "
            f"({len(informative)} grupos candidatos)",
            _sample(informative, k_g),
        )
    )
    next_ts = [t for t in d.phash.sweep_thresholds if t > t_vis]
    if next_ts:
        t_next = next_ts[0]
        small = [
            g
            for g in sweep_groups[t_next]
            if len(g) < d.phash.sweep_max_group and not _is_exact_pair(g)
        ]
        sheet_specs.append(
            (
                f"dedup_groups_t{t_next}_no_exactos.jpg",
                f"{k_g} grupos al azar del umbral {t_next} con menos de "
                f"{d.phash.sweep_max_group} imágenes, excluyendo pares byte-idénticos "
                f"({len(small)} candidatos)",
                _sample(small, k_g),
            )
        )
    for name, _, gs in sheet_specs:
        group_contact_sheet(gs, path_of, patient_of, target_of, figures / name, d.phash.thumb_px)

    # ---- 433 vs 425: grupos byte-idénticos frente a la lista oficial
    lesion_of = dict(zip(m["image_id"], m["lesion_id"], strict=True))
    exact_sizes = pd.Series([len(g) for g in exact]).value_counts().sort_index()
    exact_sizes_txt = ", ".join(f"{int(n)} grupos de {int(k)}" for k, n in exact_sizes.items())
    exact_pairs = {frozenset(g) for g in exact if len(g) == 2}
    unlisted = [g for g in exact if frozenset(g) not in official_pairs]
    official_not_exact = [p for p in official_pairs if p not in exact_pairs]
    official_diff_lesion = [sorted(p) for p in official_pairs if len({lesion_of[i] for i in p}) > 1]
    unlisted_rows = [
        {
            "grupo": ", ".join(g),
            "patient_id": ", ".join(sorted({patient_of[i] for i in g})),
            "lesion_id": ", ".join(sorted({lesion_of[i] for i in g})),
            "mismo_lesion_id": len({lesion_of[i] for i in g}) == 1,
            "target": ", ".join(str(target_of[i]) for i in g),
        }
        for g in unlisted
    ]
    lesion_multi = int((m.groupby("lesion_id").size() > 1).sum())
    official_with_mel = sum(any(target_of[i] == 1 for i in p) for p in official_pairs)
    table["official"] = [
        frozenset(p) in official_pairs for p in zip(table["image_a"], table["image_b"], strict=True)
    ]
    by_dist = (
        table.groupby("distance")
        .agg(
            pares=("distance", "size"),
            byte_identicos=("byte_identical", "sum"),
            mismo_paciente=("same_patient", "sum"),
            oficiales=("official", "sum"),
        )
        .reset_index()
    )
    by_dist["cruzados"] = by_dist["pares"] - by_dist["mismo_paciente"]
    by_dist["cruzados_acum"] = by_dist["cruzados"].cumsum()
    by_dist["oficiales_acum"] = by_dist["oficiales"].cumsum()
    n_pairs_mel = int(((near_table["target_a"] == 1) | (near_table["target_b"] == 1)).sum())
    lines = [
        "# Deduplicación ISIC 2020",
        "",
        f"Generado por `scripts/dedup_report.py`. Manifiesto: `{d.manifest_path}` "
        f"({len(m):,} imágenes).",
        f"pHash {d.phash.hash_size}x{d.phash.hash_size} (64 bits), decodificación reducida a "
        f"{d.phash.decode_size} px, umbral de Hamming **{d.phash.threshold}** "
        f"(`configs/data/isic2020.yaml`, `data.phash.threshold`).",
        f"Tiempos: hashing {t_hash / 60:.1f} min ({d.hashing.workers} procesos), "
        f"{len(ids) * (len(ids) - 1) // 2:,} pares de distancias en {t_pairs:.0f} s.",
        "",
        "## Pasada 1 — duplicados exactos (SHA256)",
        "",
        f"- Grupos con archivos byte-idénticos: **{len(exact)}** "
        f"(intra-paciente {len(exact_intra)}, cruzados {len(exact_cross)})",
        _md_groups(exact),
        "",
        "## Pasada 2 — casi-duplicados (pHash)",
        "",
        f"- Pares con distancia ≤ {d.phash.threshold}: **{len(near):,}**",
        f"- Grupos (componentes conexas): **{len(groups)}**, que abarcan "
        f"{n_images_in_groups:,} imágenes",
        f"  - Intra-paciente (se conservan, sin riesgo de fuga): **{len(intra)}**",
        f"  - Cruzados (dos o más pacientes; F1.4 los fusiona en una sola unidad de split): "
        f"**{len(cross)}**",
        f"- Pares con al menos un melanoma: {n_pairs_mel}",
        "",
        "### Distribución de distancias por pares",
        "",
        f"![histograma]({Path('figures') / 'phash_distance_hist.png'})",
        "",
        pd.DataFrame(hist_rows).to_markdown(index=False),
        "",
        f"Total de pares evaluados: {int(hist.sum()):,}. Mediana de la distancia: "
        f"{int(np.searchsorted(np.cumsum(hist), hist.sum() / 2))}.",
        "",
        "### Contraste con la lista oficial `ISIC_2020_Training_Duplicates.csv`",
        "",
        f"- Pares oficiales: **{len(official_pairs)}** "
        f"(cruzados según `patient_id`: {official_cross})",
        f"- Recuperados por pHash con umbral {d.phash.threshold}: **{official_hit}** "
        f"({100 * official_hit / len(official_pairs):.1f} %)",
        f"- Distancia pHash de los pares oficiales: mín {int(official_dist.min())}, "
        f"mediana {int(official_dist.median())}, p90 {int(official_dist.quantile(0.9))}, "
        f"máx {int(official_dist.max())}",
        f"- Pares encontrados que NO están en la lista oficial: "
        f"{len(found_pairs - official_pairs)}",
        "",
        "### Grupos cruzados",
        "",
        _md_groups(cross),
        "",
        "### Hojas de contactos",
        "",
        f"Hasta {k} pares por categoría, ordenados por distancia ascendente.",
        "",
        f"- Intra-paciente: `reports/figures/dedup_intra.jpg`"
        f" ({min(k, int(near_table['same_patient'].sum()))} pares)",
        f"- Cruzados: `reports/figures/dedup_cross.jpg`"
        f" ({min(k, int((~near_table['same_patient']).sum()))} pares)",
        "",
        "## Barrido de umbrales",
        "",
        "Mismos pHash, distintos umbrales de Hamming (`data.phash.sweep_thresholds`). "
        "`grupos_gt_10` cuenta grupos con 10 o más imágenes; `pct_dataset` es la fracción de "
        "las 33,126 imágenes que queda dentro de algún grupo:",
        "",
        sweep.to_markdown(index=False),
        "",
        "Desglose de los pares por distancia (`oficiales` = lista de ISIC; `cruzados` = "
        "pacientes distintos):",
        "",
        by_dist.to_markdown(index=False),
        "",
        "### Inspección visual de grupos",
        "",
        f"Umbral más alto con grupo mayor < {d.phash.sweep_max_group} imágenes: "
        f"**{t_vis}**. Hojas de contactos (una fila por grupo, muestreo con semilla "
        f"{d.phash.sweep_seed}):",
        "",
        *[f"- `reports/figures/{name}`: {desc}" for name, desc, _ in sheet_specs],
        "",
        "## Recomendación de umbral",
        "",
        "Lo que muestra el barrido:",
        "",
        "- **Umbral 0** (pHash idéntico): 464 pares, 452 grupos, el mayor de 4 imágenes. "
        "433 grupos son los pares byte-idénticos; los otros 19 (43 imágenes) son pares con "
        "pHash igual y bytes distintos, 18 de ellos entre pacientes distintos.",
        "- **Umbral 2**: el grupo mayor salta de 4 a 133 imágenes y aparecen 132 grupos "
        "cruzados. Ningún par oficial vive a distancia 2 (todos están a 0).",
        "- **Umbral 4 a 8**: 1,041 → 6,246 → 12,394 imágenes en un solo grupo. A 8, el 42 % "
        "del dataset queda encadenado.",
        "",
        "Lo que muestran las hojas de contactos:",
        "",
        "- Los 10 grupos muestreados a distancia 0 que no son byte-idénticos son lesiones "
        "distintas (forma, color, vello y fondo diferentes) con la misma composición: mancha "
        "oscura centrada sobre fondo claro uniforme. pHash colapsa la miniatura de 32x32 a las "
        "mismas frecuencias bajas.",
        "- Los 10 grupos muestreados a distancia 2 (de 131 candidatos con menos de 10 imágenes) "
        "son también lesiones distintas; en ninguno aparece la misma lesión con encuadre "
        "desplazado, que es lo que la pasada 2 buscaba.",
        "",
        f"**Recomendación: `data.phash.threshold = {d.phash.threshold}`**, ahora sostenida por "
        "el barrido: es el único umbral en el que el grupo mayor se mantiene por debajo de 10 "
        "imágenes, todos los pares oficiales ya están a esa distancia, y un solo paso más (2) "
        "produce encadenamiento sin recuperar ningún duplicado adicional verificable. Con "
        "umbral 0 pHash aporta 19 grupos más que SHA256; son falsos positivos visuales, pero "
        "fusionar a esos 23 pacientes en unidades de split no cuesta nada, así que se conservan "
        "como cruzados por prudencia. La consecuencia honesta es que, en ISIC 2020, la "
        "deduplicación efectiva la hace SHA256; pHash queda como verificación de que no hay "
        "casi-duplicados por recodificación (ninguno: todo par a distancia 0 con bytes "
        "distintos es una lesión distinta).",
        "",
        "### Por qué pHash falla en dermatoscopia",
        "",
        "pHash reduce la imagen a 32x32 píxeles en escala de grises, aplica una DCT y conserva "
        "solo el bloque 8x8 de frecuencias más bajas, binarizado contra su mediana. Es decir, "
        "por diseño resume la **estructura global** de la imagen y descarta el detalle fino. "
        "En dermatoscopia esa estructura global la impone el método de captura: lesión "
        "centrada, fondo de piel uniforme, iluminación y escala fijas por el dermatoscopio. "
        "Vista a 32x32, casi toda imagen del dataset es «una mancha oscura en el centro de un "
        "fondo claro», y por eso lesiones de pacientes distintos coinciden bit a bit en el hash "
        "(30 pares cruzados a distancia 0) y la distribución de distancias no tiene ningún "
        "hueco que separe duplicados de parecidos. La señal que distingue una lesión de otra "
        "(retículo, glóbulos, velo, vasos, borde) vive en frecuencias altas que pHash tira. El "
        "método funciona en fotografía general porque ahí la composición varía; aquí es "
        "constante. La pasada se mantiene activa en el pipeline por su valor probatorio: "
        "demuestra que no existen casi-duplicados por recodificación, reescalado o recompresión "
        "más allá de los pares byte-idénticos.",
        "",
        "## Los 433 grupos byte-idénticos frente a los 425 de la referencia",
        "",
        f"- Tamaño de los grupos SHA256: {exact_sizes_txt} → los {len(exact)} grupos son "
        f"pares, es decir, {len(exact)} imágenes duplicadas y {2 * len(exact)} implicadas.",
        f"- Lista oficial `ISIC_2020_Training_Duplicates.csv`: {len(official_pairs)} pares "
        f"({2 * len(official_pairs)} imágenes); {official_with_mel} pares incluyen un melanoma. "
        f"Por eso la referencia habla de 32,542 benignas y 32,120 sin duplicados: "
        f"32,542 − 32,120 = 422 = {len(official_pairs)} − {official_with_mel}.",
        f"- Los {len(official_pairs)} pares oficiales son todos byte-idénticos "
        f"(pares oficiales no byte-idénticos: {len(official_not_exact)}).",
        f"- Pares byte-idénticos que la lista oficial NO incluye: **{len(unlisted)}**. "
        "Todos con el mismo `patient_id`, el mismo `lesion_id`, metadatos idénticos y "
        "`target = 0`: el mismo archivo registrado dos veces con dos `image_id`.",
        "",
        pd.DataFrame(unlisted_rows).to_markdown(index=False),
        "",
        f"- `lesion_id` repetidos en el manifiesto: {lesion_multi}. De los pares oficiales, "
        f"{len(official_diff_lesion)} tienen `lesion_id` distinto pese a ser el mismo archivo "
        f"(inconsistencia de metadatos de la fuente): {official_diff_lesion}.",
        "",
        "Conclusión: la diferencia son 8 pares reales que la referencia no lista, no un "
        "artefacto de conteo. La deduplicación de este proyecto usa los 433.",
        "",
        "## Punto de verificación",
        "",
        "La literatura reporta ~425 duplicados benignos (32,542 benignas, 32,120 sin duplicados). "
        f"La lista oficial trae {len(official_pairs)} pares; SHA256 encuentra {len(exact)} "
        f"grupos byte-idénticos y pHash con umbral {d.phash.threshold} "
        f"encuentra {len(near):,} pares y recupera {official_hit} de los oficiales.",
        "",
        "## Pasada 3 — embeddings",
        "",
        "**No se ejecuta.** Motivos: (1) la curación de ISIC 2020 garantiza una imagen por "
        "lesión y `lesion_id` solo se repite en los pares oficiales, que ya están cubiertos; "
        "(2) la pasada 2 muestra que en este dataset cualquier medida global de parecido "
        "produce miles de pares cruzados entre lesiones distintas, y la similitud coseno de "
        "embeddings tendría el mismo problema de precisión sin una verdad de referencia para "
        "calibrarla; (3) el costo (~3 h de CPU) no compraría un guardarraíl que los splits "
        "agrupados por paciente no den ya.",
        "",
    ]
    (reports / "dedup_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"Reporte: {reports / 'dedup_report.md'}")


if __name__ == "__main__":
    main()
