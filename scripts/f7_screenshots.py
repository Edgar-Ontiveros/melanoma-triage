"""F7 — Capturas de la interfaz con Playwright: Análisis (vacío y con el resultado del ejemplo
ISIC_5037784) y /modelo, a 1280 y 390 px, en reports/figures/f7_ui_*.png.

Requiere una API levantada con el build de la web (make api) en BASE_URL (por defecto
http://127.0.0.1:8000). Chromium necesita bibliotecas del sistema; si no están (WSL sin sudo)
se puede correr dentro de la imagen oficial de Playwright con la red del anfitrión:

  docker run --rm --network host -v "$PWD:/work" -w /work \\
      mcr.microsoft.com/playwright/python:v1.49.0-jammy \\
      python scripts/f7_screenshots.py --base-url http://127.0.0.1:8020
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
EXAMPLE = "ISIC_5037784"
VIEWPORTS = {"1280": (1280, 900), "390": (390, 844)}
ACCEPT_MODAL = "window.sessionStorage.setItem('melanoma-triage:aviso-aceptado', '1')"


def shoot(page, path: Path) -> None:
    page.wait_for_timeout(300)
    page.screenshot(path=str(path), full_page=True)
    print(f"→ {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for tag, (w, h) in VIEWPORTS.items():
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
            ctx.add_init_script(ACCEPT_MODAL)
            page = ctx.new_page()
            # Análisis vacío (con los ejemplos cargados)
            page.goto(f"{a.base_url}/", wait_until="networkidle")
            page.wait_for_selector(".example")
            shoot(page, OUT / f"f7_ui_analisis_vacio_{tag}.png")
            # Resultado del ejemplo
            page.click(f".example:has(code:text-is('{EXAMPLE}'))")
            page.wait_for_selector(".prob-big", timeout=60_000)
            page.wait_for_timeout(500)
            shoot(page, OUT / f"f7_ui_analisis_resultado_{tag}.png")
            # Ficha del modelo
            page.goto(f"{a.base_url}/modelo", wait_until="networkidle")
            page.wait_for_selector("table")
            shoot(page, OUT / f"f7_ui_modelo_{tag}.png")
            # El modal, tal como lo ve una sesión nueva
            fresh = browser.new_context(viewport={"width": w, "height": h})
            mp = fresh.new_page()
            mp.goto(f"{a.base_url}/", wait_until="networkidle")
            mp.wait_for_selector(".modal")
            mp.screenshot(path=str(OUT / f"f7_ui_aviso_{tag}.png"))
            print(f"→ reports/figures/f7_ui_aviso_{tag}.png")
            fresh.close()
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
