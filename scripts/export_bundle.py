# ruff: noqa: E501
"""F6.1 — Exporta el paquete de modelo a models/bundle/ desde el checkpoint final de F3, con la
calibración y los umbrales de F4 y las métricas del test. Se detiene si el SHA256 del
checkpoint no coincide con reports/f3_final_model.json.

Uso: uv run python scripts/export_bundle.py
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from melanoma.eval.predict import load_checkpoint, sha256_file
from melanoma.export.bundle import (
    calibration_spec,
    metrics_spec,
    thresholds_spec,
    write_bundle,
)
from melanoma.train.run import git_sha

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="f6", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt = ROOT / cfg.f6.checkpoint
    sha = sha256_file(ckpt)
    expected = json.loads((ROOT / cfg.f6.checkpoint_manifest).read_text())["sha256"]
    if sha != expected:
        raise SystemExit(f"SHA256 del checkpoint {sha} ≠ {expected} ({cfg.f6.checkpoint_manifest})")
    lit, data_config = load_checkpoint(
        ckpt, cfg.model.backbone, cfg.model.num_classes, cfg.model.dropout
    )
    f4 = json.loads((ROOT / cfg.f6.calibration_json).read_text())
    f4_test = json.loads((ROOT / cfg.f6.test_results_json).read_text())
    if f4["checkpoint"]["sha256"] != sha or f4_test["checkpoint_sha256"] != sha:
        raise SystemExit("la calibración o las métricas de F4 no corresponden a este checkpoint")
    out = ROOT / cfg.f6.bundle_dir
    manifest = write_bundle(
        lit.model,
        data_config,
        out,
        image_size=int(cfg.data.image_size),
        val_resize=str(cfg.data.val_resize),
        opset=int(cfg.f6.opset),
        calibration=calibration_spec(f4),
        thresholds=thresholds_spec(f4),
        metrics=metrics_spec(f4_test),
        checkpoint_sha256=sha,
        checkpoint_file=ckpt.name,
        git_sha=git_sha(ROOT),
        extra={"kind": "release", "backbone": str(cfg.model.backbone)},
    )
    print(json.dumps({k: manifest[k] for k in ("model_version", "onnx", "files")}, indent=2))
    print(
        f"→ {out.relative_to(ROOT)} ({sum((out / f).stat().st_size for f in manifest['files']) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
