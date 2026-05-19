# -*- coding: utf-8 -*-
"""Capture screenshots of the current Canopy UI for the thesis.

The previous screenshot script targeted an older UI. The 2026-05 redesign
moved branding to "Canopy", added a Map / Image view switcher in the
topbar, replaced the previous fixed sidebar with a context-aware left
panel on the Map view + a progressive-disclosure sidebar on the Image
view, and introduced a centred per-action ScanModelModal that pops up
before drawing modes start.

Run order:
  1. Start backend separately: `venv/Scripts/python.exe -m uvicorn backend.main:app`
  2. Run this script: `venv/Scripts/python.exe thesis/take_screenshots_v2.py`

Each shot's filename matches the figure referenced from chapter 2.
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
BASE_URL = "http://127.0.0.1:8000"

# Astana centre — used to pan the Leaflet map deterministically.
ASTANA = (51.1605, 71.4704)


def shot(page, name: str, *, full: bool = False, delay_ms: int = 1500):
    page.wait_for_timeout(delay_ms)
    out = FIGURES / name
    page.screenshot(path=str(out), full_page=full)
    print(f"  wrote {out.name}")


def main() -> int:
    if not FIGURES.exists():
        FIGURES.mkdir(parents=True)

    with sync_playwright() as p:
        # 1440 × 900 — wide enough to show topbar + left panel + map in one
        # frame on the Map view, and topbar + sidebar + image in one frame
        # on the Image view. Use a deviceScaleFactor of 2 so the resulting
        # PNGs render sharply when embedded in the LaTeX PDF at half size.
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
        )
        page = ctx.new_page()

        # ── Map view (default landing) ────────────────────────────────────
        page.goto(BASE_URL, wait_until="networkidle", timeout=20000)
        page.wait_for_selector(".topbar-brand", timeout=10000)
        # Force-set the Leaflet view to Astana so the basemap is consistent
        # between runs regardless of saved viewport state.
        page.evaluate(
            f"""() => {{
                const divs = document.querySelectorAll('.leaflet-container');
                divs.forEach((d) => {{
                    if (d._leaflet_map) {{
                        d._leaflet_map.setView([{ASTANA[0]}, {ASTANA[1]}], 13);
                    }}
                }});
            }}"""
        )
        shot(page, "ui_canopy_map_view.png", delay_ms=2500)

        # Light mode variant of the same view
        page.click("[title='Switch to light']", timeout=3000)
        shot(page, "ui_canopy_map_view_light.png", delay_ms=1200)
        # Back to dark for the rest
        page.click("[title='Switch to dark']", timeout=3000)
        page.wait_for_timeout(500)

        # ── Scan model modal (centred picker before drawing) ──────────────
        page.click("button:has-text('Scan area')", timeout=3000)
        page.wait_for_selector(".modal", timeout=3000)
        shot(page, "ui_canopy_scan_model_modal.png", delay_ms=600)
        page.click("button:has-text('Cancel')", timeout=2000)
        page.wait_for_timeout(400)

        # ── Library / Manager modal (scans tab) ───────────────────────────
        page.click("button:has-text('Manage')", timeout=3000)
        page.wait_for_selector(".modal-title:has-text('Library')", timeout=3000)
        shot(page, "ui_canopy_library_modal.png", delay_ms=600)
        # Close modal
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        # ── Settings popover ──────────────────────────────────────────────
        page.click("[title='Settings']", timeout=3000)
        page.wait_for_timeout(500)
        shot(page, "ui_canopy_settings_popover.png", delay_ms=400)
        page.click("[title='Settings']", timeout=2000)  # close
        page.wait_for_timeout(400)

        # ── Image view (right sidebar, progressive-disclosure) ────────────
        page.click("button:has-text('Image')", timeout=3000)
        page.wait_for_selector(".upload-zone", timeout=5000)
        shot(page, "ui_canopy_image_view_empty.png", delay_ms=1000)

        browser.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
