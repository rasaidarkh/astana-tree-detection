"""Pydantic schemas — единый контракт между моделями, API и frontend."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelKind(str, Enum):
    YOLO = "yolo"
    DEEPFOREST = "deepforest"
    ENSEMBLE = "ensemble"


class GeoMode(str, Enum):
    GEOTIFF = "geotiff"          # GPS из affine transform файла
    CORNERS_4 = "corners_4"      # 4 угла (NW/NE/SW/SE)
    CORNERS_2 = "corners_2"      # 2 угла, axis-aligned
    NONE = "none"                # без геопривязки, только пиксели


class LatLng(BaseModel):
    lat: float
    lng: float


class Corners4(BaseModel):
    nw: LatLng
    ne: LatLng
    sw: LatLng
    se: LatLng


class Corners2(BaseModel):
    nw: LatLng           # top-left
    se: LatLng           # bottom-right


class GeoParams(BaseModel):
    mode: GeoMode
    corners_2: Optional[Corners2] = None
    corners_4: Optional[Corners4] = None


class BBox(BaseModel):
    """Pixel-space bbox: [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area_px(self) -> float:
        return max(0.0, (self.x2 - self.x1)) * max(0.0, (self.y2 - self.y1))


class Detection(BaseModel):
    """Один найденный объект (дерево). Унифицированный для всех моделей."""
    id: int
    box: BBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = "tree"

    # Опциональная маска (instance segmentation): RLE-кодированная или polygon
    mask_polygon: Optional[list[list[float]]] = None  # [[x,y], [x,y], ...]
    crown_area_px: Optional[float] = None

    # Опциональная геопривязка (заполняется после geo-конвертации)
    lat: Optional[float] = None
    lng: Optional[float] = None
    crown_diameter_m: Optional[float] = None
    # Та же маска, но в lat/lng — для рендера на Leaflet
    mask_polygon_geo: Optional[list[list[float]]] = None  # [[lat,lng], ...]
    # 4 угла bbox в lat/lng — NW, NE, SE, SW. Рисуется как Leaflet polygon (точный четырёхугольник).
    box_geo: Optional[list[list[float]]] = None  # [[lat,lng] × 4]


class ImageMeta(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    size_bytes: int
    is_geotiff: bool = False
    crs: Optional[str] = None
    pixel_size_m: Optional[float] = None
    bounds: Optional[Corners2] = None  # auto-extracted из GeoTIFF


class PredictRequest(BaseModel):
    image_id: str
    model: ModelKind = ModelKind.YOLO
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    geo: GeoParams = GeoParams(mode=GeoMode.NONE)


class PredictResult(BaseModel):
    job_id: str
    image_id: str
    model: ModelKind
    detections: list[Detection]
    duration_ms: int
    stats: dict


class HistoryEntry(BaseModel):
    image_id: str
    filename: str
    date: str                     # ISO timestamp
    model: ModelKind
    tree_count: int
    coverage_pct: Optional[float] = None
