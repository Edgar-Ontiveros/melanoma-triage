"""Corridas en Kaggle desde la terminal, con la API (CLI 2.2.4). Nada por navegador.

    push <corrida|kernel> --sha SHA [--set K=V ...] [--wait]  empuja y ejecuta el notebook
    status <corrida>      estado de la última ejecución
    wait <corrida>        espera, reporta y baja la salida
    output <corrida>      baja la salida a reports/runs/<corrida>/
    logs <corrida>        log de la última ejecución
    list                  corridas de configs/kaggle.yaml y su estado
    dataset-images        publica data/processed/isic2020_512 como un solo zip
    sync-wandb <corrida>  sube a W&B las corridas offline bajadas con `output`
    batch <c1> <c2> ... --sha SHA   empuja varias corridas y las vigila en un solo bucle

`kernel-metadata.json` se genera desde `configs/kaggle.yaml`; los parámetros de la primera celda
del notebook (REPO_SHA, OVERRIDES, RUN_TAG, ...) se sustituyen con `melanoma.utils.notebook`.
Verificado el 2026-09-17: estados `KernelWorkerStatus.{QUEUED,RUNNING,COMPLETE,ERROR,...}`; la API
responde 429 si se consulta seguido, por eso el sondeo es lento y reintenta.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma.utils.notebook import (  # noqa: E402
    load_notebook,
    save_notebook,
    substitute_parameters,
)

KAGGLE = Path(sys.executable).parent / "kaggle"
TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def cfg() -> DictConfig:
    return OmegaConf.load(ROOT / "configs/kaggle.yaml")


def kaggle_cli(*args: str, check: bool = True, quiet: bool = False) -> str:
    """Corre el CLI de Kaggle; reintenta ante 429 (límite de tasa)."""
    for attempt in range(6):
        proc = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        if "429" in out and "Too Many Requests" in out:
            wait = int(cfg().retry_seconds)
            print(f"  [429 de la API; reintento {attempt + 1} en {wait} s]", flush=True)
            time.sleep(wait)
            continue
        if not quiet:
            print(out)
        if check and proc.returncode != 0:
            raise SystemExit(f"kaggle {' '.join(args)} falló ({proc.returncode})")
        return out
    raise SystemExit("la API de Kaggle sigue devolviendo 429")


# ---- resolución de corridas -------------------------------------------------------------
def resolve(name: str, c: DictConfig) -> tuple[str, DictConfig, dict[str, Any]]:
    """``(kernel_name, kernel_cfg, params)`` para una corrida de ``runs`` o un kernel suelto."""
    if name in c.runs:
        run = c.runs[name]
        return str(run.kernel), c.kernels[run.kernel], dict(OmegaConf.to_container(run.params))
    if name in c.kernels:
        return name, c.kernels[name], {}
    raise SystemExit(
        f"{name!r} no está en configs/kaggle.yaml "
        f"(runs: {list(c.runs)}, kernels: {list(c.kernels)})"
    )


def slug_of(name: str, kernel_name: str, k: DictConfig) -> str:
    if "slug" in k:
        return str(k.slug)
    return f"{k.slug_prefix}-{name}"


def ref_of(name: str, c: DictConfig) -> str:
    kernel_name, k, _ = resolve(name, c)
    return f"{c.username}/{slug_of(name, kernel_name, k)}"


def kernel_metadata(name: str, c: DictConfig, code_file: str) -> dict[str, Any]:
    kernel_name, k, _ = resolve(name, c)
    slug = slug_of(name, kernel_name, k)
    meta = {
        "id": f"{c.username}/{slug}",
        "title": slug,  # Kaggle deriva el slug del título: título == slug evita sorpresas
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if k.gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": str(c.gpu_machine_shape) if k.gpu else "",
        "dataset_sources": [str(c.datasets[d]) for d in k.datasets],
        "competition_sources": [str(c.competition)] if k.competition else [],
        "kernel_sources": [],
        "model_sources": [],
    }
    return meta


def stage_kernel(name: str, c: DictConfig, sha: str, extra: dict[str, Any]) -> Path:
    """Carpeta lista para ``kaggle kernels push``: notebook parametrizado + kernel-metadata.json."""
    kernel_name, k, params = resolve(name, c)
    params = {"REPO_SHA": sha, **params, **extra}
    nb = substitute_parameters(load_notebook(ROOT / k.notebook), params)
    folder = ROOT / "outputs" / "kaggle" / "kernels" / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    code_file = Path(k.notebook).name
    save_notebook(nb, folder / code_file)
    meta = kernel_metadata(name, c, code_file)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return folder


# ---- comandos ---------------------------------------------------------------------------
def cmd_push(args: argparse.Namespace) -> None:
    c = cfg()
    extra = {}
    for kv in args.set or []:
        key, _, val = kv.partition("=")
        try:
            extra[key] = json.loads(val)
        except json.JSONDecodeError:
            extra[key] = val
    folder = stage_kernel(args.name, c, args.sha, extra)
    print(f"kernel-metadata.json:\n{(folder / 'kernel-metadata.json').read_text()}")
    print(f"parámetros:\n{(folder / 'params.json').read_text()}")
    out = kaggle_cli("kernels", "push", "-p", str(folder))
    if "successfully pushed" not in out:
        raise SystemExit("el push no confirmó la versión")
    if args.wait:
        wait_and_collect(args.name, c)


def status_of(name: str, c: DictConfig) -> str:
    out = kaggle_cli("kernels", "status", ref_of(name, c), check=False, quiet=True)
    m = re.search(r'has status "([^"]+)"', out)
    if not m:
        return f"desconocido: {out.strip()[:200]}"
    msg = re.search(r'Failure message: "([^"]*)"', out)
    return m.group(1).replace("KernelWorkerStatus.", "") + (f" ({msg.group(1)})" if msg else "")


def cmd_status(args: argparse.Namespace) -> None:
    c = cfg()
    print(f"{ref_of(args.name, c)}: {status_of(args.name, c)}")


def wait_and_collect(name: str, c: DictConfig) -> None:
    ref = ref_of(name, c)
    poll = int(c.poll_seconds)
    start = time.time()
    while True:
        status = status_of(name, c)
        elapsed = (time.time() - start) / 60
        print(f"[{datetime.now():%H:%M:%S}] {ref}: {status}  ({elapsed:.0f} min)", flush=True)
        if any(t in status.upper() for t in TERMINAL):
            break
        time.sleep(poll)
    if "COMPLETE" in status.upper():
        save_log(name, c)
        try:
            download_output(name, c)
        except SystemExit as exc:
            print(f"salida no bajada: {exc}. El log quedó en reports/runs/{name}/kernel.log")
    else:
        path = save_log(name, c)
        print("\n===== LOG DE LA EJECUCIÓN (falló), cola =====")
        print(path.read_text(encoding="utf-8")[-6000:])
        raise SystemExit(f"{ref} terminó con estado {status}")


def cmd_wait(args: argparse.Namespace) -> None:
    wait_and_collect(args.name, cfg())


def download_output(name: str, c: DictConfig) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    _, k, _ = resolve(name, c)
    dest = ROOT / c.output_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    ref = ref_of(name, c)
    token, total = None, 0
    while True:
        for attempt in range(6):
            try:
                files, token = api.kernels_output(
                    ref,
                    str(dest),
                    file_pattern=str(k.output_pattern),
                    force=True,
                    quiet=True,
                    page_token=token,
                    page_size=100,
                )
                break
            except Exception as exc:  # 429 u otro error transitorio
                if "429" not in str(exc):
                    raise
                if attempt == 5:
                    raise SystemExit(
                        "la API responde 429 al listar la salida del kernel. Verificado el "
                        "2026-09-17: pasa cuando la salida tiene decenas de miles de archivos "
                        "(no es límite de tasa). Dejar los archivos grandes fuera de "
                        "/kaggle/working y usar `logs`."
                    ) from exc
                time.sleep(int(c.retry_seconds))
        total += len(files)
        for f in files:
            print(f"  ↓ {f}")
        if not token:
            break
    print(f"{total} archivos en {dest.relative_to(ROOT)}")
    return dest


def cmd_output(args: argparse.Namespace) -> None:
    download_output(args.name, cfg())


def fetch_log(name: str, c: DictConfig) -> str:
    """Texto plano (stdout+stderr) del log; el CLI devuelve una lista JSON de eventos."""
    raw = kaggle_cli("kernels", "logs", ref_of(name, c), check=False, quiet=True)
    start = raw.find("[")
    try:
        events = json.loads(raw[start:]) if start >= 0 else []
    except json.JSONDecodeError:
        return raw
    return "".join(
        e.get("data", "") for e in events if e.get("stream_name") in ("stdout", "stderr")
    )


def save_log(name: str, c: DictConfig) -> Path:
    dest = ROOT / c.output_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "kernel.log"
    path.write_text(fetch_log(name, c), encoding="utf-8")
    print(f"log en {path.relative_to(ROOT)}")
    return path


def cmd_logs(args: argparse.Namespace) -> None:
    print(fetch_log(args.name, cfg()))


def cmd_list(args: argparse.Namespace) -> None:
    c = cfg()
    for name in list(c.kernels) + list(c.runs):
        if name == "train":
            continue
        print(f"{name:14s} {ref_of(name, c):45s} {status_of(name, c)}")
        time.sleep(3)


def cmd_dataset_images(args: argparse.Namespace) -> None:
    c = cfg()
    d = c.images_dataset
    src = ROOT / d.source_dir
    jpgs = sorted(src.glob("*.jpg"))
    if not jpgs:
        raise SystemExit(f"no hay JPEG en {src}")
    staging = ROOT / d.staging_dir
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    dataset_id = f"{c.username}/{d.title}"
    verification = ROOT / d.verification_json
    if verification.exists():
        shutil.copy2(verification, staging / "resize_verification.json")
    meta = {
        "title": str(d.title),
        "id": dataset_id,
        "licenses": [{"name": str(d.license)}],
        "subtitle": str(d.subtitle),
        "description": (
            "Imagenes del SIIM-ISIC 2020 Challenge Dataset (CC-BY-NC 4.0, "
            "https://doi.org/10.34970/2020-ds01) redimensionadas al lado largo de 512 px con "
            f"JPEG calidad 95 por {c.repo_url} (data/processed/isic2020_512). Byte-identicas al "
            "redimensionado hecho en Kaggle: 33126/33126 huellas SHA256 coinciden (2026-09-17)."
        ),
        "keywords": ["medicine", "image"],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zip_path = staging / "isic2020_512.zip"
    print(f"empaquetando {len(jpgs)} JPEG en {zip_path.relative_to(ROOT)} (sin compresión)…")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in jpgs:
            z.write(p, p.name)
    print(f"  {zip_path.stat().st_size / 1e9:.2f} GB en {time.time() - t0:.0f} s")
    exists = "ready" in kaggle_cli("datasets", "status", dataset_id, check=False, quiet=True)
    if exists:
        kaggle_cli(
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            f"actualización {datetime.now():%F}",
            "--dir-mode",
            "skip",
        )
    else:
        kaggle_cli("datasets", "create", "-p", str(staging), "--dir-mode", "skip")
    print(
        f"https://www.kaggle.com/datasets/{dataset_id}  (Kaggle extrae el zip; revisar con "
        f"`kaggle datasets files {dataset_id}`)"
    )


def cmd_batch(args: argparse.Namespace) -> None:
    """Empuja todas las corridas (Kaggle ejecuta en paralelo las que su cuota permita y encola el
    resto) y las sondea en un solo bucle: una llamada de estado por corrida y ciclo, con pausa
    entre llamadas para no provocar 429. Al completar una baja su salida; si falla, guarda el
    log."""
    c = cfg()
    pending: dict[str, str] = {}
    for name in args.names:
        folder = stage_kernel(name, c, args.sha, {})
        out = kaggle_cli("kernels", "push", "-p", str(folder), quiet=True)
        m = re.search(r"Kernel version (\d+) successfully pushed", out)
        if not m:
            raise SystemExit(f"push de {name} no confirmó la versión:\n{out}")
        pending[name] = "pushed"
        print(
            f"[{datetime.now():%H:%M:%S}] {ref_of(name, c)}: versión {m.group(1)} empujada",
            flush=True,
        )
        time.sleep(5)
    results: dict[str, str] = {}
    start = time.time()
    while pending:
        for name in list(pending):
            status = status_of(name, c)
            elapsed = (time.time() - start) / 60
            print(f"[{datetime.now():%H:%M:%S}] {name}: {status}  ({elapsed:.0f} min)", flush=True)
            if any(t in status.upper() for t in TERMINAL):
                results[name] = status
                del pending[name]
                if "COMPLETE" in status.upper():
                    save_log(name, c)
                    try:
                        download_output(name, c)
                    except SystemExit as exc:
                        print(f"  salida de {name} no bajada: {exc}")
                else:
                    save_log(name, c)
            time.sleep(5)
        if pending:
            time.sleep(int(c.poll_seconds))
    print("\n===== RESULTADO DEL LOTE =====")
    for name, status in results.items():
        print(f"  {name:16s} {status}")
    if any("COMPLETE" not in s.upper() for s in results.values()):
        raise SystemExit("alguna corrida del lote no completó")


def offline_wandb_dirs(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.rglob("offline-run-*") if p.is_dir())


def cmd_sync_wandb(args: argparse.Namespace) -> None:
    """Sin secretos en el kernel, W&B corre offline en Kaggle; `output` baja esos directorios y
    aquí se sincronizan con la cuenta local (`wandb login` una vez)."""
    c = cfg()
    run_dir = ROOT / c.output_dir / args.name
    dirs = offline_wandb_dirs(run_dir)
    if not dirs:
        raise SystemExit(f"no hay directorios offline-run-* bajo {run_dir}")
    project = str(OmegaConf.load(ROOT / "configs/train/default.yaml").wandb.project)
    for d in dirs:
        print(f"wandb sync {d.relative_to(ROOT)}")
        subprocess.run(
            [str(Path(sys.executable).parent / "wandb"), "sync", "--project", project, str(d)],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="empujar y ejecutar un notebook como kernel")
    p.add_argument("name")
    p.add_argument("--sha", required=True, help="commit del repositorio que ejecuta el kernel")
    p.add_argument(
        "--set", action="append", metavar="PARAM=VALOR", help="parámetro extra (JSON o texto)"
    )
    p.add_argument("--wait", action="store_true")
    p.set_defaults(fn=cmd_push)
    for name, fn in (
        ("status", cmd_status),
        ("wait", cmd_wait),
        ("output", cmd_output),
        ("logs", cmd_logs),
    ):
        q = sub.add_parser(name)
        q.add_argument("name")
        q.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("dataset-images").set_defaults(fn=cmd_dataset_images)
    q = sub.add_parser("sync-wandb")
    q.add_argument("name")
    q.set_defaults(fn=cmd_sync_wandb)
    b = sub.add_parser("batch", help="empujar varias corridas y vigilarlas en un solo bucle")
    b.add_argument("names", nargs="+")
    b.add_argument("--sha", required=True)
    b.set_defaults(fn=cmd_batch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
