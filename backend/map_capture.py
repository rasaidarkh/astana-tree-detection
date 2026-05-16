"""Capture a satellite image from a slippy-tile basemap (Esri World Imagery)
for an arbitrary lat/lng bbox + zoom. Stitches tiles, crops to exact bbox,
returns a saved PIL image with embedded bounds.

Why we need this: пользователь рисует rectangle на Leaflet-карте, бэкенд
без вмешательства Google/SAS.Planet получает геопривязанный кадр и сразу
прогоняет через модель. Эквивалент скриншот+вручную NW/SE, но точно.
"""

from __future__ import annotations

import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

TILE_SIZE = 256
MAX_TILES = 144  # 12×12 ≈ 3072×3072 px — потолок чтобы не вешать сервер
ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
USER_AGENT = "AstanaTreeDetection/1.0 (academic; AITU diploma project)"


@dataclass
class CaptureResult:
    image: Image.Image       # точно обрезанная картинка
    nw_lat: float
    nw_lng: float
    se_lat: float
    se_lng: float


def latlng_to_tile_xy(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    """Continuous tile coordinates (float). Целая часть — индекс тайла,
    дробная — смещение внутри тайла в долях."""
    n = 2 ** zoom
    x = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(z: int, x: int, y: int) -> tuple[Image.Image, bool]:
    """Returns (image, ok). After 3 failed attempts returns a gray placeholder
    with ok=False so the caller can count failures and decide whether the
    stitched capture is still worth keeping (instead of silently feeding a
    mostly-gray image into the detector)."""
    url = ESRI_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            return Image.open(BytesIO(data)).convert("RGB"), True
        except Exception:
            continue
    placeholder = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (60, 60, 60))
    return placeholder, False


def capture_bbox(
    nw_lat: float,
    nw_lng: float,
    se_lat: float,
    se_lng: float,
    zoom: int,
    max_workers: int = 8,
) -> CaptureResult:
    """Скачать тайлы, склеить, обрезать точно по bbox."""
    if zoom < 1 or zoom > 19:
        raise ValueError(f"Esri zoom must be 1..19, got {zoom}")
    if nw_lat <= se_lat or nw_lng >= se_lng:
        raise ValueError(
            f"NW must be top-left of SE; got NW=({nw_lat},{nw_lng}) SE=({se_lat},{se_lng})"
        )

    nx_f, ny_f = latlng_to_tile_xy(nw_lat, nw_lng, zoom)
    sx_f, sy_f = latlng_to_tile_xy(se_lat, se_lng, zoom)
    x_min, x_max = int(math.floor(nx_f)), int(math.floor(sx_f))
    y_min, y_max = int(math.floor(ny_f)), int(math.floor(sy_f))
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    n_tiles = cols * rows
    if n_tiles > MAX_TILES:
        raise ValueError(
            f"Bbox too large at zoom {zoom}: {n_tiles} tiles (max {MAX_TILES}). "
            f"Уменьши прямоугольник или zoom."
        )

    # параллельно качаем все тайлы
    jobs = [(zoom, x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda j: _fetch_tile(*j), jobs))
    tiles = [r[0] for r in results]
    failed_count = sum(1 for r in results if not r[1])
    if failed_count and failed_count / len(jobs) > 0.5:
        # More than half of the tiles fell back to gray placeholders — the
        # stitched image would be useless input for the detector, so surface
        # the error instead of returning a mostly-empty grid.
        raise IOError(
            f"Tile capture failed: {failed_count}/{len(jobs)} tiles unreachable "
            f"(more than half — likely network outage or ESRI rate-limit). "
            f"Retry later or use a smaller bbox."
        )
    if failed_count:
        import logging
        logging.getLogger("astana-tree").warning(
            "capture_bbox: %d/%d tiles fell back to gray placeholder",
            failed_count, len(jobs),
        )

    canvas = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for idx, tile in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        canvas.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))

    # пиксельный crop: дробная часть × TILE_SIZE даёт смещение внутри
    # первого/последнего тайла
    left = int(round((nx_f - x_min) * TILE_SIZE))
    top = int(round((ny_f - y_min) * TILE_SIZE))
    right = int(round((sx_f - x_min) * TILE_SIZE))
    bottom = int(round((sy_f - y_min) * TILE_SIZE))
    cropped = canvas.crop((left, top, right, bottom))

    return CaptureResult(
        image=cropped,
        nw_lat=nw_lat,
        nw_lng=nw_lng,
        se_lat=se_lat,
        se_lng=se_lng,
    )


def save_capture(result: CaptureResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out_path, format="PNG", optimize=False)
