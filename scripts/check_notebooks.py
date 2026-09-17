"""Ejecuta los notebooks de Kaggle de principio a fin en un sandbox local que imita a Kaggle.

Por qué existe: la prueba de humo no cubre los notebooks, y dos errores de F1/F2 solo aparecieron
al correr en Kaggle (rutas de /kaggle/input y un `import melanoma` que resolvía al directorio del
clon). Este script reproduce las condiciones que los provocan:

- un intérprete **ya corriendo** (como el kernel) en el que el notebook hace `pip install -e .`;
- `<sandbox>/working` como cwd y en `sys.path`, y el clon dentro de él;
- `<sandbox>/input/...` con la misma estructura que `/kaggle/input/` y datos sintéticos pequeños;
- un venv **limpio** creado con uv (el grupo `train` se preinstala porque en Kaggle torch ya viene).

El código que se clona es el árbol de trabajo actual (con cambios sin commitear), para verificar
ANTES de hacer push. Los parámetros de la primera celda de código se sobreescriben para apuntar al
sandbox; el resto del notebook corre tal cual. Uso: ``make check-notebooks`` (2-4 minutos).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma.data.resize import resize_many  # noqa: E402
from melanoma.data.synthetic import write_synthetic_dataset  # noqa: E402
from melanoma.utils.notebook import (  # noqa: E402
    load_notebook,
    save_notebook,
    substitute_parameters,
)

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

RUNNER = r"""
import json, sys, traceback
nb_path = sys.argv[1]  # notebook ya con los parámetros sustituidos (melanoma.utils.notebook)
cells = [c for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    print(f"\n===== celda {i} =====\n{src[:300]}...\n", flush=True)
    try:
        exec(compile(src, f"{nb_path}:celda{i}", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"\nFALLÓ la celda {i} de {nb_path}", flush=True)
        sys.exit(1)
print(f"\nNOTEBOOK OK: {nb_path}")
"""


def sh(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.run(list(args), cwd=cwd, env=env, check=True)


def make_remote(sandbox: Path) -> Path:
    """Repositorio git con el árbol de trabajo actual (tracked + untracked no ignorados)."""
    remote = sandbox / "remote"
    remote.mkdir(parents=True)
    files = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    ).split(b"\0")
    for f in files:
        if not f:
            continue
        src = ROOT / f.decode()
        if src.is_file():
            dst = remote / f.decode()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "sandbox",
        "GIT_AUTHOR_EMAIL": "sandbox@local",
        "GIT_COMMITTER_NAME": "sandbox",
        "GIT_COMMITTER_EMAIL": "sandbox@local",
    }
    sh("git", "init", "-q", cwd=remote, env=env)
    sh("git", "add", "-A", cwd=remote, env=env)
    sh("git", "commit", "-q", "-m", "sandbox", cwd=remote, env=env)
    return remote


def make_input(sandbox: Path, n_images: int) -> dict[str, Path]:
    """Imita /kaggle/input: competencia (originales), dataset de splits y dataset de 512 px."""
    inp = sandbox / "input"
    comp = inp / "competitions/siim-isic-melanoma-classification/jpeg/train"
    splits = inp / "datasets/edgaronti26/melanoma-isic2020-splits"
    resized = inp / "datasets/edgaronti26/melanoma-isic2020-512"
    for d in (comp, splits, resized):
        d.mkdir(parents=True)
    tmp = sandbox / "synthetic"
    paths = write_synthetic_dataset(
        tmp, n_images=n_images, positive_rate=0.25, seed=0, size=(900, 600)
    )
    import pandas as pd

    manifest = pd.read_csv(paths["manifest_path"], dtype={"image_id": str, "patient_id": str})
    for image_id in manifest["image_id"]:
        shutil.copy2(paths["images_dir"] / f"{image_id}.jpg", comp / f"{image_id}.jpg")
    # "Copia local" redimensionada, como data/processed/isic2020_512 en F1
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs/data/isic2020.yaml")
    pairs = [(comp / f"{i}.jpg", resized / f"{i}.jpg") for i in manifest["image_id"]]
    results = resize_many(pairs, int(cfg.resize.long_side), int(cfg.resize.jpeg_quality), workers=2)
    manifest["sha256_resized"] = [r["sha256"] for r in results]
    manifest["width_resized"] = [r["width"] for r in results]
    manifest["height_resized"] = [r["height"] for r in results]
    manifest.to_csv(splits / "isic2020.csv", index=False)
    for f in ("train.txt", "val.txt", "SHA256SUMS"):
        shutil.copy2(paths["splits_dir"] / f, splits / f)
    return {"competition": comp, "splits": splits, "resized": resized}


def make_venv(sandbox: Path) -> Path:
    venv = sandbox / "venv"
    sh("uv", "venv", "-q", "--seed", "--python", "3.12", str(venv))
    py = venv / "bin" / "python"
    # Como en Kaggle, torch ya está instalado antes de que el notebook haga su pip install.
    sh(
        "uv",
        "pip",
        "install",
        "-q",
        "--python",
        str(py),
        "--index-url",
        TORCH_CPU_INDEX,
        "torch",
        "torchvision",
    )
    sh("uv", "pip", "install", "-q", "--python", str(py), "-e", f"{ROOT}[train]")
    # El editable anterior apunta al repo real; lo quitamos para que solo cuente el del notebook.
    sh("uv", "pip", "uninstall", "-q", "--python", str(py), "melanoma")
    return py


def run_notebook(py: Path, sandbox: Path, notebook: Path, overrides: dict) -> None:
    runner = sandbox / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    working = sandbox / "working"
    working.mkdir(exist_ok=True)
    for stale in working.iterdir():  # cada notebook clona de cero, como un kernel nuevo
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    env = {
        **os.environ,
        "KAGGLE_KERNEL_RUN_TYPE": "Sandbox",
        "PYTHONPATH": "",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "offline",
    }
    env.pop("VIRTUAL_ENV", None)
    staged = sandbox / notebook.name
    save_notebook(substitute_parameters(load_notebook(notebook), overrides), staged)
    sh(str(py), str(runner), str(staged), cwd=working, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sandbox", default=str(ROOT / "outputs" / "notebook_sandbox"))
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--only", choices=["prepare", "train"], default=None)
    args = ap.parse_args()
    sandbox = Path(args.sandbox).resolve()
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    print(f"sandbox: {sandbox}")
    remote = make_remote(sandbox)
    inp = make_input(sandbox, args.images)
    py = make_venv(sandbox)
    work = str(sandbox / "working")
    common = {"REPO_URL": str(remote), "REPO_SHA": "HEAD", "WORK": work}
    if args.only in (None, "prepare"):
        run_notebook(
            py,
            sandbox,
            ROOT / "notebooks/kaggle_prepare_512.ipynb",
            {
                **common,
                "COMPETITION_DIR": str(inp["competition"]),
                "SPLITS_DIR": str(inp["splits"]),
                "OUT_DIR": f"{work}/isic2020_512",
                "WORKERS": 2,
            },
        )
        out = sandbox / "working/isic2020_512"
        report = json.loads((out / "resize_verification.json").read_text())
        assert report["sha256_match"] == report["images"], report
        meta = json.loads((out / "dataset-metadata.json").read_text())
        assert meta["id"].endswith("/melanoma-isic2020-512"), meta
        assert len(list(out.glob("*.jpg"))) == report["images"]
        print(
            f"prepare: {report['sha256_match']}/{report['images']} huellas coinciden en el sandbox"
        )
    if args.only in (None, "train"):
        run_notebook(
            py,
            sandbox,
            ROOT / "notebooks/kaggle_train.ipynb",
            {
                **common,
                "OVERRIDES": [
                    "model=resnet18",
                    "model.pretrained=false",
                    "data.image_size=64",
                    "data.batch_size=8",
                    "train.max_epochs=1",
                    "train.accelerator=cpu",
                    "train.precision=32",
                    "train.eval.bootstrap_resamples=20",
                    "train.wandb.enabled=false",
                    "train.progress_bar=false",
                ],
                "RUN_TAG": "sandbox",
                "SPLITS_DIR": str(inp["splits"]),
                "IMAGES_DIR": str(inp["resized"]),
                "NUM_WORKERS": 0,
                "COPY_IMAGES_LOCAL": True,
            },
        )
        runs = list((sandbox / "working/runs").glob("*/metrics.json"))
        assert runs, "el notebook de entrenamiento no dejó metrics.json"
        print(f"train: {runs[0]}")
    print("\nCHECK NOTEBOOKS OK")


if __name__ == "__main__":
    main()
