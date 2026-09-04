"""Deduplicación: duplicados exactos (SHA256) y casi-duplicados (pHash, distancia de Hamming).

Los grupos se clasifican en *intra-paciente* (todas las imágenes comparten ``patient_id``,
sin riesgo de fuga con splits agrupados) y *cruzados* (dos o más pacientes, riesgo de fuga
que F1.4 resuelve fusionando esos pacientes en una misma unidad de split).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- pHash


def phash_of(path: Path | str, hash_size: int, decode_size: int) -> int:
    """pHash de una imagen como entero de ``hash_size**2`` bits.

    ``Image.draft`` pide al decodificador JPEG una escala reducida (1/2, 1/4, 1/8): pHash
    trabaja sobre una miniatura de 32x32, así que decodificar los 24 MP completos sería
    tiempo perdido.
    """
    with Image.open(path) as img:
        img.draft("RGB", (decode_size, decode_size))
        h = imagehash.phash(img.convert("RGB"), hash_size=hash_size)
    bits = h.hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _phash_worker(args: tuple[str, int, int]) -> int:
    return phash_of(*args)


def compute_phashes(
    paths: Sequence[Path | str], hash_size: int, decode_size: int, workers: int
) -> np.ndarray:
    """pHash de muchas imágenes en procesos paralelos. Devuelve ``uint64`` (hash_size ≤ 8)."""
    if hash_size > 8:
        raise ValueError("hash_size > 8 no cabe en uint64")
    jobs = [(str(p), hash_size, decode_size) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vals = list(pool.map(_phash_worker, jobs, chunksize=64))
    return np.array(vals, dtype=np.uint64)


def _hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancias de Hamming entre cada elemento de ``a`` y cada uno de ``b`` (uint64)."""
    x = (a[:, None] ^ b[None, :]).view(np.uint8).reshape(len(a), len(b), 8)
    return _POPCOUNT[x].sum(axis=2)


def pairwise_hamming(
    hashes: np.ndarray, max_distance: int, block: int = 512
) -> tuple[np.ndarray, pd.DataFrame]:
    """Histograma de todas las distancias por pares y lista de pares ≤ ``max_distance``.

    Recorre solo el triángulo superior (i < j). El histograma tiene 65 posiciones (0..64).
    """
    n = len(hashes)
    hist = np.zeros(65, dtype=np.int64)
    rows: list[tuple[int, int, int]] = []
    for start in range(0, n, block):
        a = hashes[start : start + block]
        d = _hamming_block(a, hashes)  # (len(a), n)
        # máscara del triángulo superior: j > i
        i_idx = np.arange(start, start + len(a))[:, None]
        j_idx = np.arange(n)[None, :]
        upper = j_idx > i_idx
        hist += np.bincount(d[upper].ravel(), minlength=65)[:65]
        close = np.argwhere(upper & (d <= max_distance))
        for ai, j in close:
            rows.append((start + int(ai), int(j), int(d[ai, j])))
    pairs = pd.DataFrame(rows, columns=["i", "j", "distance"])
    return hist, pairs


# ---------------------------------------------------------------- grupos


class UnionFind:
    """Unión-búsqueda mínima para agrupar pares en componentes."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        by_root: dict[int, list[int]] = {}
        for x in range(len(self.parent)):
            by_root.setdefault(self.find(x), []).append(x)
        return [g for g in by_root.values() if len(g) > 1]


def exact_duplicate_groups(manifest: pd.DataFrame) -> list[list[str]]:
    """Grupos de ``image_id`` con el mismo ``sha256_original`` (archivos byte-idénticos)."""
    dup = manifest[manifest.duplicated("sha256_original", keep=False)]
    return [sorted(g["image_id"].tolist()) for _, g in dup.groupby("sha256_original")]


def groups_from_pairs(pairs: pd.DataFrame, n: int, ids: Sequence[str]) -> list[list[str]]:
    """Componentes conexas del grafo de pares (índices) → listas de ``image_id``."""
    uf = UnionFind(n)
    for i, j in zip(pairs["i"], pairs["j"], strict=True):
        uf.union(int(i), int(j))
    return [sorted(ids[k] for k in g) for g in uf.groups()]


def classify_groups(
    groups: list[list[str]], patient_of: dict[str, str]
) -> tuple[list[list[str]], list[list[str]]]:
    """Separa grupos en (intra-paciente, cruzados)."""
    intra, cross = [], []
    for g in groups:
        (intra if len({patient_of[i] for i in g}) == 1 else cross).append(g)
    return intra, cross


def pairs_table(
    pairs: pd.DataFrame, ids: Sequence[str], patient_of: dict[str, str], target_of: dict[str, int]
) -> pd.DataFrame:
    """Pares con ids, pacientes y etiquetas, para el reporte y la hoja de contactos."""
    out = pd.DataFrame(
        {
            "image_a": [ids[i] for i in pairs["i"]],
            "image_b": [ids[j] for j in pairs["j"]],
            "distance": pairs["distance"].to_numpy(),
        }
    )
    out["patient_a"] = out["image_a"].map(patient_of)
    out["patient_b"] = out["image_b"].map(patient_of)
    out["target_a"] = out["image_a"].map(target_of)
    out["target_b"] = out["image_b"].map(target_of)
    out["same_patient"] = out["patient_a"] == out["patient_b"]
    return out.sort_values(["distance", "image_a"]).reset_index(drop=True)


# ---------------------------------------------------------------- hoja de contactos


def contact_sheet(
    pairs: pd.DataFrame, path_of: dict[str, Path], out_png: Path | str, thumb_px: int
) -> None:
    """PNG con una fila por par: imagen A | imagen B, y el texto del par en el nombre."""
    from PIL import ImageDraw

    n = len(pairs)
    if n == 0:
        return
    pad = 4
    label_h = 18
    cell = thumb_px + pad
    sheet = Image.new("RGB", (2 * cell + pad, n * (cell + label_h) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(pairs.itertuples(index=False)):
        y = pad + r * (cell + label_h)
        for c, image_id in enumerate((row.image_a, row.image_b)):
            with Image.open(path_of[image_id]) as img:
                img.draft("RGB", (thumb_px * 2, thumb_px * 2))
                img = img.convert("RGB")
                img.thumbnail((thumb_px, thumb_px))
                sheet.paste(img, (pad + c * cell, y))
        text = (
            f"{row.image_a} ({row.patient_a}, t={row.target_a})  vs  "
            f"{row.image_b} ({row.patient_b}, t={row.target_b})  d={row.distance}"
        )
        draw.text((pad, y + thumb_px + 2), text, fill="black")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)  # el formato lo decide la extensión (PNG o JPEG)


def group_contact_sheet(
    groups: list[list[str]],
    path_of: dict[str, Path],
    patient_of: dict[str, str],
    target_of: dict[str, int],
    out_path: Path | str,
    thumb_px: int,
    max_per_row: int = 6,
) -> None:
    """Una fila por grupo con hasta ``max_per_row`` imágenes; el texto lista ids y pacientes."""
    from PIL import ImageDraw

    if not groups:
        return
    pad = 4
    label_h = 18
    cell = thumb_px + pad
    width = max_per_row * cell + pad
    sheet = Image.new("RGB", (width, len(groups) * (cell + label_h) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for r, group in enumerate(groups):
        y = pad + r * (cell + label_h)
        for c, image_id in enumerate(group[:max_per_row]):
            with Image.open(path_of[image_id]) as img:
                img.draft("RGB", (thumb_px * 2, thumb_px * 2))
                img = img.convert("RGB")
                img.thumbnail((thumb_px, thumb_px))
                sheet.paste(img, (pad + c * cell, y))
        patients = sorted({patient_of[i] for i in group})
        text = (
            f"[{len(group)} img, {len(patients)} pac, mel={sum(target_of[i] for i in group)}] "
            + "  ".join(f"{i}({patient_of[i]})" for i in group[:max_per_row])
        )
        if len(group) > max_per_row:
            text += f"  … +{len(group) - max_per_row}"
        draw.text((pad, y + thumb_px + 2), text, fill="black")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
