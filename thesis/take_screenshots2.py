# -*- coding: utf-8 -*-
"""Take screenshots of the web app showing actual detection results."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
from playwright.sync_api import sync_playwright

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
BASE_URL = "http://127.0.0.1:8000"

def shot(page, path, wait=2000):
    page.wait_for_timeout(wait)
    page.screenshot(path=path)
    print(f"Saved: {path}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    # ── 1. Load result with detections via JS state ───────────────────────
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle", timeout=15000)
    page.wait_for_timeout(3000)

    # Click on history to load existing result
    try:
        history = page.locator("text=deepforest").first
        if history.count() > 0:
            history.click()
            page.wait_for_timeout(3000)
    except:
        pass

    shot(page, os.path.join(FIGURES, "ui_with_detections.png"), wait=2000)

    # ── 2. City map with zoom on Astana trees ────────────────────────────
    try:
        page.get_by_text("City map", exact=False).first.click()
        page.wait_for_timeout(4000)
        # Zoom in more
        page.evaluate("""
            () => {
                const divs = document.querySelectorAll('.leaflet-container');
                divs.forEach(div => {
                    if (div._leaflet_map) div._leaflet_map.setView([51.166, 71.446], 15);
                });
            }
        """)
        page.wait_for_timeout(3000)
        shot(page, os.path.join(FIGURES, "ui_city_map_zoomed.png"), wait=1000)
    except Exception as e:
        print(f"city map zoom: {e}")

    # ── 3. Single image + polygon display mode ────────────────────────────
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle", timeout=15000)
    page.wait_for_timeout(3000)
    shot(page, os.path.join(FIGURES, "ui_single_detection_result.png"), wait=1000)

    browser.close()

print("Done.")
