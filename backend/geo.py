"""Pixel ↔ GPS преобразования. Поддерживает 4 режима:

1. GeoTIFF (rasterio) — самый правильный, использует affine transform
2. 4 corners — bilinear interpolation, handles rotation
3. 2 corners — axis-aligned WGS84 (как в DeepForest tree_mapper)
4. None — без геопривязки

Координаты везде: lat = latitude (Y), lng = longitude (X).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import Corners2, Corners4, Detection, GeoMode, GeoParams, ImageMeta


@dataclass
class GeoContext:
    """Подготовленный контекст для повторного применения к множеству детекций."""
    mode: GeoMode
    width: int
    height: int
    pixel_size_m: Optional[float] = None
    rasterio_transform: Optional[object] = None
    corners_2: Optional[Corners2] = None
    corners_4: Optional[Corners4] = None


def load_geotiff_meta(path: str | Path) -> ImageMeta:
    """Прочитать метаданные GeoTIFF (без полной загрузки пикселей)."""
    import rasterio

    p = Path(path)
    with rasterio.open(p) as src:
        transform = src.transform
        crs = str(src.crs) if src.crs else None
        pixel_size_m = abs(float(transform[0])) if crs and "32" in str(crs) else None

        bounds = None
        try:
            from rasterio.warp import transform_bounds
            l, b, r, t = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bounds = Corners2(
                nw={"lat": t, "lng": l},
                se={"lat": b, "lng": r},
            )
        except Exception:
            pass

        return ImageMeta(
            image_id=p.stem,
            filename=p.name,
            width=src.width,
            height=src.height,
            size_bytes=p.stat().st_size,
            is_geotiff=True,
            crs=crs,
            pixel_size_m=pixel_size_m,
            bounds=bounds,
        )


def build_context(
    width: int,
    height: int,
    geo: GeoParams,
    geotiff_path: Optional[str] = None,
) -> GeoContext:
    """Построить GeoContext один раз, потом применять к каждой детекции."""
    if geo.mode == GeoMode.GEOTIFF:
        if not geotiff_path:
            raise ValueError("GeoTIFF mode requires the actual file path")
        import rasterio
        with rasterio.open(geotiff_path) as src:
            return GeoContext(
                mode=GeoMode.GEOTIFF,
                width=width,
                height=height,
                pixel_size_m=abs(float(src.transform[0])),
                rasterio_transform=src.transform,
            )

    if geo.mode == GeoMode.CORNERS_4:
        if not geo.corners_4:
            raise ValueError("corners_4 missing")
        return GeoContext(
            mode=GeoMode.CORNERS_4,
            width=width,
            height=height,
            corners_4=geo.corners_4,
            pixel_size_m=_estimate_pixel_size_from_corners4(geo.corners_4, width, height),
        )

    if geo.mode == GeoMode.CORNERS_2:
        if not geo.corners_2:
            raise ValueError("corners_2 missing")
        return GeoContext(
            mode=GeoMode.CORNERS_2,
            width=width,
            height=height,
            corners_2=geo.corners_2,
            pixel_size_m=_estimate_pixel_size_from_corners2(geo.corners_2, width, height),
        )

    return GeoContext(mode=GeoMode.NONE, width=width, height=height)


def pixel_to_gps(px: float, py: float, ctx: GeoContext) -> tuple[Optional[float], Optional[float]]:
    """Возвращает (lat, lng) или (None, None) если режим = NONE."""
    if ctx.mode == GeoMode.NONE:
        return None, None

    if ctx.mode == GeoMode.GEOTIFF:
        from rasterio.transform import xy
        lng, lat = xy(ctx.rasterio_transform, py, px)
        return float(lat), float(lng)

    if ctx.mode == GeoMode.CORNERS_2:
        c = ctx.corners_2
        u = px / ctx.width
        v = py / ctx.height
        lng = c.nw.lng + u * (c.se.lng - c.nw.lng)
        lat = c.nw.lat + v * (c.se.lat - c.nw.lat)  # nw.lat > se.lat для северного полушария
        return lat, lng

    if ctx.mode == GeoMode.CORNERS_4:
        c = ctx.corners_4
        u = px / ctx.width
        v = py / ctx.height
        # Bilinear interpolation between 4 corners
        top_lat = c.nw.lat + u * (c.ne.lat - c.nw.lat)
        top_lng = c.nw.lng + u * (c.ne.lng - c.nw.lng)
        bot_lat = c.sw.lat + u * (c.se.lat - c.sw.lat)
        bot_lng = c.sw.lng + u * (c.se.lng - c.sw.lng)
        lat = top_lat + v * (bot_lat - top_lat)
        lng = top_lng + v * (bot_lng - top_lng)
        return lat, lng

    return None, None


def annotate_detections(detections: list[Detection], ctx: GeoContext) -> list[Detection]:
    """Заполнить lat/lng/crown_diameter_m в каждой детекции (in-place + return)."""
    for det in detections:
        lat, lng = pixel_to_gps(det.box.cx, det.box.cy, ctx)
        det.lat = lat
        det.lng = lng
        if ctx.pixel_size_m and det.box.area_px > 0:
            # diameter ≈ sqrt(area * 4 / pi)
            diameter_px = float(np.sqrt(det.box.area_px * 4 / np.pi))
            det.crown_diameter_m = round(diameter_px * ctx.pixel_size_m, 2)
    return detections


# ---------- helpers ----------

_EARTH_R_M = 6_378_137.0


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = rlat2 - rlat1
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlng / 2) ** 2
    return float(2 * _EARTH_R_M * np.arcsin(np.sqrt(a)))


def _estimate_pixel_size_from_corners2(c: Corners2, w: int, h: int) -> float:
    # Среднее по двум сторонам — грубо, но достаточно для статистики
    horiz_m = _haversine_m(c.nw.lat, c.nw.lng, c.nw.lat, c.se.lng)
    vert_m = _haversine_m(c.nw.lat, c.nw.lng, c.se.lat, c.nw.lng)
    return float(np.mean([horiz_m / w, vert_m / h]))


def _estimate_pixel_size_from_corners4(c: Corners4, w: int, h: int) -> float:
    horiz = _haversine_m(c.nw.lat, c.nw.lng, c.ne.lat, c.ne.lng)
    vert = _haversine_m(c.nw.lat, c.nw.lng, c.sw.lat, c.sw.lng)
    return float(np.mean([horiz / w, vert / h]))
