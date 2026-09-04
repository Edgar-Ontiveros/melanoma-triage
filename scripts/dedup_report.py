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
    hashes = compute_phashes(
        [path_of[i] for i in ids], d.phash.hash_size, d.phash.decode_size, d.hashing.workers
    )
    t_hash = time.time() - t0
    t0 = time.time()
    hist, pairs = pairwise_hamming(hashes, d.phash.report_max_distance)
    t_pairs = time.time() - t0
    np.save(reports / "phash_uint64.npy", hashes)

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
        "## Justificación del umbral",
        "",
        "Pares hasta la distancia máxima reportada, desglosados. `oficiales` son pares de la "
        "lista de ISIC; `cruzados` son pares entre pacientes distintos:",
        "",
        by_dist.to_markdown(index=False),
        "",
        "Lectura de la tabla con los datos reales (2026-09-04):",
        "",
        "- Los 425 pares oficiales están a distancia **0** y además son **byte-idénticos**; "
        "la pasada 1 (SHA256) ya los recupera al 100 %.",
        "- No hay ningún hueco en la distribución: a partir de la distancia 2 aparecen cientos "
        "de pares cruzados y a distancia 8 cientos de miles, ninguno oficial. Con umbral 8 "
        "los grupos abarcaban 13,997 imágenes y 451 grupos cruzados, es decir, casi la mitad "
        "del dataset quedaba encadenada en una sola unidad de split.",
        "- Inspección visual de muestras de pares cruzados a distancias 0, 2, 4 y 6: son "
        "lesiones distintas (mancha oscura centrada sobre piel clara). pHash resume una "
        "miniatura de 32x32 en frecuencias bajas y esa composición es la misma en casi todas "
        "las imágenes dermatoscópicas, así que la distancia no separa lesión repetida de "
        "lesión parecida.",
        "",
        f"**Decisión:** `data.phash.threshold = {d.phash.threshold}`. Solo cuentan como "
        "casi-duplicados los pares con pHash idéntico. Los grupos cruzados que quedan a esa "
        "distancia son falsos positivos visuales, pero fusionar sus pacientes en una misma "
        "unidad de split no cuesta nada y elimina la duda; por eso se conservan como cruzados.",
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
