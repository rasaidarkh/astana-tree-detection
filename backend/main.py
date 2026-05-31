"""FastAPI app: roots, model registry, статика frontend.

Запуск:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Открыть http://localhost:8000 — встроенный UI.
API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from . import db
from .export import to_csv, to_geojson, to_standalone_html
from .geo import GeoContext, annotate_detections, build_context, load_geotiff_meta
from .map_capture import DEFAULT_PROVIDER, TILE_PROVIDERS, capture_bbox, save_capture
from .models import ModelRegistry
from .region_scan import DEFAULT_MAX_SUBREGIONS, plan_scan
from .models.base import ModelAdapter
from .models.deepforest_adapter import DeepForestAdapter
from .models.deepforest_sam2_adapter import DeepForestSAM2Adapter
from .models.ensemble_adapter import EnsembleAdapter
from .models.yolo_ensemble_adapter import MultiYOLOEnsembleAdapter
from .models.maskrcnn_adapter import MaskRCNNAdapter
from .models.yolo_adapter import YOLOAdapter
from .schemas import (
    Corners2,
    GeoMode,
    GeoParams,
    HistoryEntry,
    ImageMeta,
    LatLng,
    ModelKind,
    PredictRequest,
    PredictResult,
)


class CaptureFromMapRequest(BaseModel):
    nw: LatLng
    se: LatLng
    zoom: int = Field(18, ge=1, le=20)
    provider: str = Field(DEFAULT_PROVIDER, description="Tile provider key: esri | google")


class ScanRegionRequest(BaseModel):
    """Auto-Zoom Region Scan request — см. region_scan.py.

    Пользователь рисует прямоугольник любого размера, сервер сам выбирает
    `zoom` (по умолчанию 19 — максимальный для Esri, ~0.3 м/px) и при
    необходимости дробит bbox в сетку под-регионов. На каждом запускается
    обычный capture+predict; результаты сохраняются как отдельные snapshots
    в БД и автоматически появляются в city-aggregate view.

    `provider` выбирает источник тайлов: `esri` (default, World Imagery)
    или `google` (Maps satellite — та же image base что у Google Earth Pro,
    ближе к тренировочному распределению).
    """
    nw: LatLng
    se: LatLng
    zoom: int = Field(19, ge=14, le=20)
    model: ModelKind = ModelKind.YOLO
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    max_subregions: int = Field(DEFAULT_MAX_SUBREGIONS, ge=1, le=25)
    provider: str = Field(DEFAULT_PROVIDER, description="Tile provider key: esri | google")
    # Опциональный пользовательский полигон. Если задан — bbox (nw/se) должен
    # быть его axis-aligned bounding rectangle (фронт вычисляет и шлёт оба).
    # Бэк делит bbox в Layer 0 как обычно, а потом фильтрует детекции:
    # оставляет только те, центр которых внутри полигона. Это даёт UX
    # "обведи парк / квартал / линию реки" без переписывания region_scan.
    polygon: Optional[list[LatLng]] = Field(
        default=None,
        description="≥3 точки; если задан — детекции фильтруются point-in-polygon",
    )

# ============ Setup ============

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("astana-tree")

ROOT = Path(__file__).parent.parent
STORAGE = ROOT / "storage"
UPLOADS = STORAGE / "uploads"
RESULTS = STORAGE / "results"
WEIGHTS = ROOT / "weights"
FRONTEND = ROOT / "frontend"
# Canonical resolved frontend root, used as the boundary for path-traversal
# checks in serve_static (see below).
FRONTEND_ROOT = FRONTEND.resolve()
DB_PATH = STORAGE / "app.db"

UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
db.init_db(DB_PATH)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# ============ App ============

app = FastAPI(
    title="Canopy — Astana Tree Detection",
    description="Canopy: end-to-end система автоматической инвентаризации городских деревьев Астаны",
    version="0.1.0",
)

# CORS: restrict to local dev origins by default — `allow_origins=["*"]` plus
# `allow_methods=["*"]` is an obvious CSRF vector the moment authentication is
# added. Override via the ASTANA_CORS_ORIGINS env var (comma-separated) when
# deploying behind a different domain.
_DEFAULT_CORS = "http://localhost:8000,http://127.0.0.1:8000"
_cors_origins = [
    o.strip()
    for o in os.environ.get("ASTANA_CORS_ORIGINS", _DEFAULT_CORS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
log.info("CORS allowed origins: %s", _cors_origins)

# ============ Model registry (lazy) ============

registry = ModelRegistry()


def _register_yolo_variant(weights_path: Path, kind: ModelKind, display_name: str) -> None:
    """Регистрация YOLO variant под конкретным ModelKind. Используется для
    debug-выбора между v2 / v3-run1 / v3-run2 / production. Override class
    attributes `kind` + `name` на instance, чтобы один и тот же adapter-класс
    мог появиться под разными ключами в registry."""
    if not weights_path.exists():
        log.info("YOLO variant %s weights missing at %s — skipping", kind.value, weights_path)
        return
    adapter = YOLOAdapter(weights_path=str(weights_path))
    adapter.kind = kind
    adapter.name = display_name
    registry.register(adapter)
    log.info("YOLO variant registered: %s -> %s", kind.value, weights_path.name)


def _load_models() -> None:
    """Регистрируем все доступные адаптеры. Веса грузятся лениво при первом predict."""
    # Production YOLO (whatever is at yolo_satellite.pt сейчас)
    yolo_path = WEIGHTS / "yolo_satellite.pt"
    if yolo_path.exists():
        yolo = YOLOAdapter(weights_path=str(yolo_path))
        registry.register(yolo)
        log.info("YOLO adapter registered: %s", yolo_path)
    else:
        log.warning("YOLO weights missing at %s — endpoint вернёт 503", yolo_path)

    # Debug variants — позволяют пользователю переключаться между моделями для
    # сравнения качества на конкретных сценах (например v3 даёт false positives
    # на стадионных крышах а v2 не давала — можно сравнить side-by-side через
    # выбор модели в UI). Все архивные веса остаются на диске.
    _register_yolo_variant(
        WEIGHTS / "archive" / "yolo" / "yolo_satellite_v2_finetune.pt",
        ModelKind.YOLO_V2,
        "YOLOv8x · v2-finetune (mAP@50 0.187)",
    )
    # v3 run1 / run2 / exp1 — все архивы в weights/v3_runs/.
    for pattern, kind, label in [
        ("v3_finetune_run1_*.pt", ModelKind.YOLO_V3_RUN1, "YOLOv8x · v3 run 1 (mAP@50 0.268)"),
        ("v3_finetune_run2_*.pt", ModelKind.YOLO_V3_RUN2, "YOLOv8x · v3 run 2 (mAP@50 0.246)"),
        ("exp1_m_cocostart_*.pt", ModelKind.YOLO_V3_EXP1, "YOLOv8m · v3 exp1 tuned (mAP@50 0.308)"),
    ]:
        matches = sorted((WEIGHTS / "v3_runs").glob(pattern))
        if matches:
            # Ignore PROD_BACKUP suffix-ed файлы — это duplicate of run1.
            real = [m for m in matches if "PROD_BACKUP" not in m.name]
            if real:
                _register_yolo_variant(real[0], kind, label)

    # v4 clean sweep — Ultralytics defaults, без manual tuning. v4_x_clean это
    # фактический CHAMPION (mAP@50 0.315) — best single-model на merged val.
    for pattern, kind, label in [
        ("v4_x_clean_*.pt", ModelKind.YOLO_V4_X, "YOLOv8x · v4 champion (mAP@50 0.315)"),
        ("v4_m_clean_*.pt", ModelKind.YOLO_V4_M, "YOLOv8m · v4 (mAP@50 0.291)"),
        ("v4_s_clean_*.pt", ModelKind.YOLO_V4_S, "YOLOv8s · v4 fast (mAP@50 0.281)"),
    ]:
        matches = sorted((WEIGHTS / "v4_clean").glob(pattern))
        if matches:
            _register_yolo_variant(matches[0], kind, label)

    df_ckpt = WEIGHTS / "deepforest_astana.pl"
    df = DeepForestAdapter(checkpoint_path=str(df_ckpt) if df_ckpt.exists() else None)
    registry.register(df)
    log.info("DeepForest adapter registered (checkpoint=%s)", df_ckpt if df_ckpt.exists() else "pretrained")

    if ModelKind.YOLO in registry and ModelKind.DEEPFOREST in registry:
        ensemble = EnsembleAdapter(
            yolo_adapter=registry.get(ModelKind.YOLO),
            deepforest_adapter=registry.get(ModelKind.DEEPFOREST),
        )
        registry.register(ensemble)
        log.info("Ensemble adapter registered")

    # Cross-YOLO ensemble — vote_2 over 4 user-selected variants.
    # Uses YOLOAdapter instances already in registry. Skips silently if any
    # member is missing.
    # Members span FOUR distinct generations (v4-x, v3-m, v3-x, v2) on purpose:
    # the stadium-roof false positive is a v4-generation regression, so a vote
    # that is mostly v4 cannot suppress it. Mixing in v3_run1 and the
    # pre-regression v2-finetune lets the K=2 vote outvote the v4 roof artifact.
    yolo_ens_members = []
    for kind in [ModelKind.YOLO_V4_X, ModelKind.YOLO_V3_EXP1,
                 ModelKind.YOLO_V3_RUN1, ModelKind.YOLO_V2]:
        if kind in registry:
            yolo_ens_members.append((kind.value, registry.get(kind)))
    if len(yolo_ens_members) >= 2:
        yolo_ensemble = MultiYOLOEnsembleAdapter(
            members=yolo_ens_members,
            min_models=2,
            iou_threshold=0.5,
        )
        registry.register(yolo_ensemble)
        log.info(
            "Cross-YOLO ensemble registered (%d members, vote_2)",
            len(yolo_ens_members),
        )

    sam2_ckpt = WEIGHTS / "sam2_hiera_base_plus.pt"
    df_sam2 = DeepForestSAM2Adapter(
        df_checkpoint_path=str(df_ckpt) if df_ckpt.exists() else None,
        sam2_checkpoint_path=str(sam2_ckpt) if sam2_ckpt.exists() else None,
    )
    registry.register(df_sam2)
    log.info("DeepForest+SAM2 adapter registered (sam2_local=%s)", sam2_ckpt.exists())

    mrcnn_ckpt = WEIGHTS / "maskrcnn_astana.pt"
    mrcnn = MaskRCNNAdapter(
        checkpoint_path=str(mrcnn_ckpt) if mrcnn_ckpt.exists() else None,
    )
    registry.register(mrcnn)
    log.info(
        "Mask R-CNN adapter registered (checkpoint=%s)",
        mrcnn_ckpt if mrcnn_ckpt.exists() else "torchvision pretrained",
    )


_load_models()

# ============ Persistent store via SQLite (see backend/db.py) ============


# ============ Routes: status ============


@app.get("/api/providers")
def list_providers() -> dict:
    """Список поддерживаемых tile-провайдеров (id, label, url-шаблон, max_zoom).

    Frontend использует чтобы построить provider-dropdown и синхронизировать
    Leaflet base layer (тот же URL что и для backend capture — иначе on-screen
    вид расходится с тем что модель видит)."""
    return {
        "default": DEFAULT_PROVIDER,
        "providers": {
            key: {
                "label": cfg["label"],
                "url": cfg["url"],
                "max_zoom": cfg["max_zoom"],
                "subdomains": cfg["subdomains"],
            }
            for key, cfg in TILE_PROVIDERS.items()
        },
    }


@app.get("/api/status")
def status() -> dict:
    agg = db.aggregate_stats()
    return {
        "status": "ok",
        "models": {
            kind.value: {
                "name": registry.get(kind).name if kind in registry else None,
                "available": kind in registry,
                "loaded": registry.get(kind).is_ready if kind in registry else False,
            }
            for kind in ModelKind
        },
        "snapshots": agg["snapshot_count"],
        "runs": agg["run_count"],
        "total_trees": agg["total_trees"],
    }


# ============ Routes: upload ============


@app.post("/api/upload", response_model=ImageMeta)
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Empty filename")

    # Sanitize: strip any path components (Windows / *nix) so a malicious
    # filename like "../../etc/passwd" cannot end up displayed in the UI / DB.
    safe_filename = Path(file.filename).name or "upload"
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported extension {ext}. Allowed: {sorted(ALLOWED_EXTS)}")

    image_id = uuid.uuid4().hex[:12]
    saved_path = UPLOADS / f"{image_id}{ext}"

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(contents)} bytes, max {MAX_UPLOAD_BYTES})")
    saved_path.write_bytes(contents)

    meta = _build_meta(saved_path, image_id, safe_filename, len(contents))
    db.save_snapshot(meta)
    log.info(
        "Uploaded %s → %s (%dx%d, %s)",
        safe_filename, saved_path.name, meta.width, meta.height,
        "GeoTIFF" if meta.is_geotiff else "regular",
    )
    return meta


@app.post("/api/capture_from_map", response_model=ImageMeta)
async def capture_from_map(req: CaptureFromMapRequest):
    """Скачивает Esri-тайлы для bbox, склеивает и сохраняет как обычный upload.
    Возвращает ImageMeta — клиент дальше идёт по обычному /api/predict."""
    try:
        # capture_bbox is sync + network-bound — off-load to the threadpool so
        # the event loop is not blocked while tens of tiles download serially.
        result = await asyncio.to_thread(
            capture_bbox,
            req.nw.lat, req.nw.lng, req.se.lat, req.se.lng, req.zoom,
            provider=req.provider,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (urllib.error.URLError, urllib.error.HTTPError, IOError) as e:
        log.warning("capture_bbox network/tile-fetch error: %s", e)
        raise HTTPException(502, f"Network error fetching tiles: {e}")
    except MemoryError:
        log.exception("capture_bbox out of memory while stitching")
        raise HTTPException(507, "Out of memory while stitching tiles — reduce bbox or zoom level")
    except Exception as e:
        log.exception("capture_bbox failed (unexpected)")
        raise HTTPException(500, f"Capture failed: {e}")

    image_id = uuid.uuid4().hex[:12]
    out_path = UPLOADS / f"{image_id}.png"
    save_capture(result, out_path)
    size_bytes = out_path.stat().st_size
    w, h = result.image.size

    meta = ImageMeta(
        image_id=image_id,
        filename=f"map_capture_{req.provider}_z{req.zoom}.png",
        width=w,
        height=h,
        size_bytes=size_bytes,
        is_geotiff=False,
        bounds=Corners2(
            nw=LatLng(lat=result.nw_lat, lng=result.nw_lng),
            se=LatLng(lat=result.se_lat, lng=result.se_lng),
        ),
    )
    db.save_snapshot(meta)
    log.info(
        "Captured from map: provider=%s bbox=(%.5f,%.5f → %.5f,%.5f) z=%d → %dx%d (%d bytes)",
        req.provider, req.nw.lat, req.nw.lng, req.se.lat, req.se.lng, req.zoom, w, h, size_bytes,
    )
    return meta


def _scan_region_setup(req: ScanRegionRequest):
    """Pre-flight для scan_region: validate + план под-регионов. Raises HTTPException."""
    adapter: Optional[ModelAdapter] = registry.get(req.model)
    if adapter is None:
        available = [k.value for k in registry.available()]
        raise HTTPException(503, f"Model {req.model.value} not available. Available: {available}")
    if req.provider not in TILE_PROVIDERS:
        raise HTTPException(400, f"Unknown tile provider {req.provider!r}. Known: {sorted(TILE_PROVIDERS)}")
    try:
        subs = plan_scan(
            req.nw.lat, req.nw.lng, req.se.lat, req.se.lng,
            zoom=req.zoom,
            max_subregions=req.max_subregions,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return adapter, subs


def _make_polygon_filter(polygon_points):
    """Возвращает predicate `inside(lat, lng) -> bool` или None если полигона нет.

    Используем shapely Polygon: внутри есть готовая point-in-polygon реализация
    (ray-casting + STR-tree оптимизация для batch вызовов). Опасный момент —
    shapely работает в декартовых координатах, поэтому передаём lng,lat как
    x,y. На масштабе одного scan-bbox-а (≤2-3 км) искажения от проекции
    пренебрежимо малы (Астана = ~51° широты).
    """
    if not polygon_points or len(polygon_points) < 3:
        return None
    from shapely.geometry import Point, Polygon
    poly = Polygon([(p.lng, p.lat) for p in polygon_points])
    if not poly.is_valid:
        # Самопересекающийся полигон — пробуем починить через buffer(0),
        # стандартный shapely-трюк для self-intersection-фиксов.
        poly = poly.buffer(0)
        if not poly.is_valid:
            log.warning("Polygon invalid after buffer(0) — пропускаем фильтрацию")
            return None
    return lambda lat, lng: poly.contains(Point(lng, lat))


async def _scan_region_events(req: ScanRegionRequest, adapter: ModelAdapter, subs: list):
    """Async generator yielding scan-progress events.

    Используется обеими endpoint-ами: sync `/api/scan_region` агрегирует
    события в финальный dict, streaming `/api/scan_region/stream` сериализует
    каждое событие в NDJSON и шлёт клиенту по мере появления.

    Event types:
      * `plan`        — список под-регионов которые сервер будет посещать + scan_session_id.
      * `capturing`   — старт скачивания тайлов под-региона.
      * `capture_done`— тайлы склеены + сохранён snapshot (без detections).
      * `predicting`  — старт инференса модели.
      * `sub_complete`— успешный finish одного под-региона (+ detections).
      * `sub_error`   — fail (capture или predict) — scan продолжается.
      * `done`        — итог всего scan'а.
    """
    # Создаём scan-сессию в БД — все sub-region runs привяжутся к ней,
    # пользователь сможет удалить весь скан одной кнопкой.
    scan_session_id = uuid.uuid4().hex[:12]
    polygon_payload = (
        [[p.lat, p.lng] for p in req.polygon] if getattr(req, "polygon", None) else None
    )
    inside_polygon = _make_polygon_filter(getattr(req, "polygon", None))
    db.create_scan_session(
        session_id=scan_session_id,
        nw_lat=req.nw.lat, nw_lng=req.nw.lng,
        se_lat=req.se.lat, se_lng=req.se.lng,
        zoom=req.zoom, provider=req.provider, model=req.model.value,
        sub_count=len(subs), polygon=polygon_payload,
    )

    sub_bbox_payload = [
        {
            "row": s.row, "col": s.col,
            "nw": {"lat": s.nw_lat, "lng": s.nw_lng},
            "se": {"lat": s.se_lat, "lng": s.se_lng},
        }
        for s in subs
    ]
    yield {
        "type": "plan",
        "scan_session_id": scan_session_id,
        "sub_count": len(subs),
        "sub_regions": sub_bbox_payload,
        "zoom": req.zoom,
        "provider": req.provider,
        "model": req.model.value,
        "polygon": polygon_payload,
        "bbox": {
            "nw": {"lat": req.nw.lat, "lng": req.nw.lng},
            "se": {"lat": req.se.lat, "lng": req.se.lng},
        },
    }

    total_trees = 0
    t_total = time.perf_counter()

    for sub in subs:
        sub_label = f"r{sub.row}c{sub.col}"
        sub_bbox = {
            "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
            "se": {"lat": sub.se_lat, "lng": sub.se_lng},
        }

        yield {"type": "capturing", "row": sub.row, "col": sub.col, "sub_bbox": sub_bbox}
        try:
            cap = await asyncio.to_thread(
                capture_bbox,
                sub.nw_lat, sub.nw_lng, sub.se_lat, sub.se_lng, req.zoom,
                provider=req.provider,
            )
        except ValueError as e:
            log.warning("scan_region sub-%s capture rejected: %s", sub_label, e)
            yield {"type": "sub_error", "row": sub.row, "col": sub.col, "stage": "capture", "error": str(e), "sub_bbox": sub_bbox}
            continue
        except (urllib.error.URLError, urllib.error.HTTPError, IOError) as e:
            log.warning("scan_region sub-%s tile fetch failed: %s", sub_label, e)
            yield {"type": "sub_error", "row": sub.row, "col": sub.col, "stage": "capture", "error": f"tile fetch failed: {e}", "sub_bbox": sub_bbox}
            continue
        except Exception as e:
            log.exception("scan_region sub-%s capture failed unexpectedly", sub_label)
            yield {"type": "sub_error", "row": sub.row, "col": sub.col, "stage": "capture", "error": f"capture failed: {e}", "sub_bbox": sub_bbox}
            continue

        image_id = uuid.uuid4().hex[:12]
        out_path = UPLOADS / f"{image_id}.png"
        save_capture(cap, out_path)
        size_bytes = out_path.stat().st_size
        w, h = cap.image.size
        bounds = Corners2(
            nw=LatLng(lat=cap.nw_lat, lng=cap.nw_lng),
            se=LatLng(lat=cap.se_lat, lng=cap.se_lng),
        )
        meta = ImageMeta(
            image_id=image_id,
            filename=f"scan_{req.provider}_z{req.zoom}_{sub_label}.png",
            width=w, height=h, size_bytes=size_bytes,
            is_geotiff=False,
            bounds=bounds,
        )
        db.save_snapshot(meta)
        yield {
            "type": "capture_done",
            "row": sub.row, "col": sub.col,
            "snapshot_id": image_id,
            "width": w, "height": h, "size_bytes": size_bytes,
            "sub_bbox": sub_bbox,
        }

        yield {"type": "predicting", "row": sub.row, "col": sub.col, "snapshot_id": image_id, "sub_bbox": sub_bbox}
        t_sub = time.perf_counter()
        try:
            detections = await asyncio.to_thread(
                adapter.predict, str(out_path), confidence=req.confidence,
            )
        except Exception as e:
            log.exception("scan_region sub-%s predict failed", sub_label)
            yield {
                "type": "sub_error", "row": sub.row, "col": sub.col,
                "stage": "predict", "error": f"predict failed: {e}",
                "snapshot_id": image_id, "sub_bbox": sub_bbox,
            }
            continue

        ctx = build_context(
            width=w, height=h,
            geo=GeoParams(mode=GeoMode.CORNERS_2, corners_2=bounds),
            geotiff_path=None,
        )
        detections = annotate_detections(detections, ctx)
        # Point-in-polygon фильтрация — если scan был по полигону, отрезаем
        # детекции центры которых выпали за реальные границы. Делается ПОСЛЕ
        # annotate_detections потому что нужны lat/lng.
        if inside_polygon is not None:
            before_n = len(detections)
            detections = [
                d for d in detections
                if d.lat is None or d.lng is None or inside_polygon(d.lat, d.lng)
            ]
            log.info(
                "scan_region sub-%s polygon-filter: %d → %d detections",
                sub_label, before_n, len(detections),
            )
        stats = _compute_stats(detections, meta, ctx)
        duration_ms = int((time.perf_counter() - t_sub) * 1000)
        job_id = uuid.uuid4().hex[:12]
        result = PredictResult(
            job_id=job_id, image_id=image_id, model=req.model,
            detections=detections, duration_ms=duration_ms, stats=stats,
        )
        db.save_run(
            result, geo_mode=GeoMode.CORNERS_2.value,
            confidence=req.confidence, scan_session_id=scan_session_id,
        )
        total_trees += len(detections)
        # Сериализуем детекции в plain dict — Pydantic models не serializable
        # из StreamingResponse без model_dump().
        det_payload = [d.model_dump() for d in detections]
        yield {
            "type": "sub_complete",
            "row": sub.row, "col": sub.col,
            "snapshot_id": image_id, "job_id": job_id,
            "tree_count": len(detections),
            "duration_ms": duration_ms,
            "detections": det_payload,
            "sub_bbox": sub_bbox,
        }
        log.info(
            "scan_region sub-%s: %d trees in %d ms (snapshot=%s, job=%s)",
            sub_label, len(detections), duration_ms, image_id, job_id,
        )

    total_duration_ms = int((time.perf_counter() - t_total) * 1000)
    # ok_count = успешные sub_complete; восстанавливаем через прямой подсчёт
    # runs привязанных к этой сессии (они были INSERT-нуты ровно для успехов).
    ok_count = len(db.get_scan_session_image_ids(scan_session_id))
    db.finalize_scan_session(
        scan_session_id, ok_count=ok_count, total_trees=total_trees,
        duration_ms=total_duration_ms, status="completed",
    )
    yield {
        "type": "done",
        "scan_session_id": scan_session_id,
        "sub_count": len(subs),
        "ok_count": ok_count,
        "total_trees": total_trees,
        "duration_ms": total_duration_ms,
    }


@app.post("/api/scan_region")
async def scan_region(req: ScanRegionRequest):
    """Auto-Zoom Region Scan — большой bbox → сетка под-регионов на фикс. зуме,
    каждая под-область прогоняется через тот же capture+predict pipeline.

    Синхронный endpoint: возвращает только когда ВСЕ под-регионы обработаны.
    Для прогрессивного UI используй `/api/scan_region/stream`.
    """
    adapter, subs = _scan_region_setup(req)
    log.info(
        "scan_region (sync): provider=%s z=%d → %d sub-region(s), model=%s",
        req.provider, req.zoom, len(subs), req.model.value,
    )

    sub_results = []
    total_trees = 0
    duration_ms = 0
    async for ev in _scan_region_events(req, adapter, subs):
        if ev["type"] == "sub_complete":
            sub_results.append({
                "row": ev["row"], "col": ev["col"],
                "snapshot_id": ev["snapshot_id"], "job_id": ev["job_id"],
                "tree_count": ev["tree_count"], "duration_ms": ev["duration_ms"],
                "sub_bbox": ev["sub_bbox"],
            })
        elif ev["type"] == "sub_error":
            sub_results.append({
                "row": ev["row"], "col": ev["col"],
                "error": ev["error"], "sub_bbox": ev["sub_bbox"],
                "snapshot_id": ev.get("snapshot_id"),
            })
        elif ev["type"] == "done":
            total_trees = ev["total_trees"]
            duration_ms = ev["duration_ms"]

    ok_count = sum(1 for r in sub_results if "error" not in r)
    log.info(
        "scan_region done: %d/%d sub-regions ok, %d trees, %d ms total",
        ok_count, len(sub_results), total_trees, duration_ms,
    )
    return {
        "sub_count": len(sub_results),
        "ok_count": ok_count,
        "total_trees": total_trees,
        "duration_ms": duration_ms,
        "zoom": req.zoom,
        "provider": req.provider,
        "model": req.model.value,
        "bbox": {
            "nw": {"lat": req.nw.lat, "lng": req.nw.lng},
            "se": {"lat": req.se.lat, "lng": req.se.lng},
        },
        "sub_regions": sub_results,
    }


@app.post("/api/scan_region/stream")
async def scan_region_stream(req: ScanRegionRequest):
    """Streaming variant of `/api/scan_region` — отдаёт NDJSON-поток событий
    по мере обработки каждого под-региона. Клиент читает через fetch +
    ReadableStream и обновляет UI инкрементально (грид под-регионов,
    деревья появляются батчами, ETA из прогресса).

    Pre-flight (validate model/provider/plan) делается синхронно и может
    отдать обычный 4xx/5xx до старта потока. Внутри потока ошибки одного
    под-региона не валят весь scan — приходят как `sub_error` событие.

    Content-Type: application/x-ndjson — одна JSON-строка на event,
    `\\n`-разделитель. Никакого SSE-префикса `data: ` чтобы парсить было
    тупо через `JSON.parse(line)`.
    """
    adapter, subs = _scan_region_setup(req)
    log.info(
        "scan_region (stream): provider=%s z=%d → %d sub-region(s), model=%s",
        req.provider, req.zoom, len(subs), req.model.value,
    )

    async def ndjson_gen():
        try:
            async for ev in _scan_region_events(req, adapter, subs):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            log.exception("scan_region stream crashed mid-flight")
            yield json.dumps({"type": "fatal", "error": str(e)}) + "\n"

    return StreamingResponse(
        ndjson_gen(),
        media_type="application/x-ndjson",
        # Отрубаем proxy-буферизацию (nginx etc.) — иначе первый событие
        # доедет до клиента только после закрытия стрима, и весь прогресс
        # потеряет смысл.
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    if db.load_snapshot(image_id) is None:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    matches = list(UPLOADS.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(404, "File missing on disk")
    return FileResponse(matches[0])


@app.get("/api/image/{image_id}/meta", response_model=ImageMeta)
def get_image_meta(image_id: str):
    meta = db.load_snapshot(image_id)
    if not meta:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    return meta


# ============ Routes: predict ============


@app.post("/api/predict", response_model=PredictResult)
async def predict(req: PredictRequest):
    meta = db.load_snapshot(req.image_id)
    if not meta:
        raise HTTPException(404, f"Unknown image_id {req.image_id}. Сначала загрузи через /api/upload.")

    adapter: Optional[ModelAdapter] = registry.get(req.model)
    if adapter is None:
        available = [k.value for k in registry.available()]
        raise HTTPException(503, f"Model {req.model.value} not available. Available: {available}")

    matches = list(UPLOADS.glob(f"{req.image_id}.*"))
    if not matches:
        raise HTTPException(404, "Image file missing on disk")
    image_path = str(matches[0])

    log.info("Running %s on %s (conf=%.2f)", adapter.kind.value, image_path, req.confidence)
    t0 = time.perf_counter()
    # Off-load the synchronous PyTorch/Ultralytics inference to a worker
    # thread so the event loop stays responsive (status/aggregate calls).
    detections = await asyncio.to_thread(
        adapter.predict, image_path, confidence=req.confidence
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    ctx = build_context(
        width=meta.width,
        height=meta.height,
        geo=req.geo,
        geotiff_path=image_path if meta.is_geotiff else None,
    )
    detections = annotate_detections(detections, ctx)
    stats = _compute_stats(detections, meta, ctx)

    job_id = uuid.uuid4().hex[:12]
    result = PredictResult(
        job_id=job_id,
        image_id=req.image_id,
        model=req.model,
        detections=detections,
        duration_ms=duration_ms,
        stats=stats,
    )
    db.save_run(result, geo_mode=req.geo.mode.value, confidence=req.confidence)
    log.info(
        "Done %s: %d detections in %d ms (job=%s)",
        adapter.kind.value, len(detections), duration_ms, job_id,
    )
    return result


@app.get("/api/result/{job_id}", response_model=PredictResult)
def get_result(job_id: str):
    result = db.load_run(job_id)
    if not result:
        raise HTTPException(404, f"Unknown job_id {job_id}")
    return result


# ============ Routes: export ============


@app.post("/api/export/{job_id}/{fmt}")
def export(job_id: str, fmt: str):
    result = db.load_run(job_id)
    if not result:
        raise HTTPException(404, f"Unknown job_id {job_id}")

    meta = db.load_snapshot(result.image_id)

    if fmt == "geojson":
        body = json.dumps(to_geojson(result.detections, meta), indent=2)
        return Response(
            content=body,
            media_type="application/geo+json",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.geojson"'},
        )

    if fmt == "csv":
        return Response(
            content=to_csv(result.detections),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'},
        )

    if fmt == "html":
        title = f"Astana Trees · {meta.filename if meta else job_id}"
        return Response(
            content=to_standalone_html(result.detections, title=title),
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.html"'},
        )

    raise HTTPException(400, f"Unsupported format {fmt}. Available: geojson, csv, html")


# ============ Routes: history & aggregate ============


@app.get("/api/history", response_model=list[HistoryEntry])
def history(limit: int = 20):
    runs = db.list_recent_runs(limit=limit)
    out: list[HistoryEntry] = []
    for r in runs:
        stats = json.loads(r["stats_json"]) if r.get("stats_json") else {}
        out.append(HistoryEntry(
            image_id=r["image_id"],
            filename=r["filename"],
            date=r["created_at"],
            model=ModelKind(r["model"]),
            tree_count=r["tree_count"],
            coverage_pct=stats.get("coverage_pct"),
        ))
    return out


class RenameRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=120)


class ScanPatchRequest(BaseModel):
    """Generic PATCH body для scan-session: меняем name и/или hidden flag.
    Оба поля optional — если поле не задано, БД-запись остаётся как есть."""
    name: Optional[str] = Field(None, max_length=120)
    hidden: Optional[bool] = None


@app.patch("/api/scans/{session_id}")
def patch_scan(session_id: str, req: ScanPatchRequest):
    touched = False
    if "name" in req.model_fields_set:
        if not db.rename_scan_session(session_id, req.name):
            raise HTTPException(404, f"Unknown scan session {session_id}")
        touched = True
    if req.hidden is not None:
        if not db.set_scan_hidden(session_id, req.hidden):
            raise HTTPException(404, f"Unknown scan session {session_id}")
        touched = True
    if not touched:
        # Touch nothing — but verify the scan exists so frontend gets a useful 404.
        scans = db.list_scan_sessions()
        if not any(s["id"] == session_id for s in scans):
            raise HTTPException(404, f"Unknown scan session {session_id}")
    return {"ok": True, "id": session_id, "name": req.name, "hidden": req.hidden}


@app.patch("/api/snapshots/{image_id}")
def rename_snapshot(image_id: str, req: RenameRequest):
    ok = db.rename_snapshot(image_id, req.name)
    if not ok:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    return {"ok": True, "image_id": image_id, "name": req.name}


@app.get("/api/scans")
def list_scans():
    """Список Auto-Zoom Scan-сессий — каждая = один большой scan_region,
    раскрученный в N sub-region snapshots с tagging через runs.scan_session_id.
    """
    return db.list_scan_sessions()


@app.delete("/api/scans/{session_id}")
def delete_scan_session(session_id: str):
    """Каскадом удаляет всю scan-сессию: сначала её snapshots (а с ними
    runs+detections через FK CASCADE), потом запись scan_sessions, потом
    файлы snapshots с диска."""
    existed, image_ids = db.delete_scan_session(session_id)
    if not existed:
        raise HTTPException(404, f"Unknown scan session {session_id}")
    files_removed = 0
    for img_id in image_ids:
        for p in UPLOADS.glob(f"{img_id}.*"):
            try:
                p.unlink()
                files_removed += 1
            except Exception as e:
                log.warning("Failed to remove %s: %s", p, e)
    log.info(
        "Deleted scan session %s (snapshots=%d, files=%d)",
        session_id, len(image_ids), files_removed,
    )
    return {
        "deleted": True,
        "scan_session_id": session_id,
        "snapshots_deleted": len(image_ids),
        "files_removed": files_removed,
    }


@app.get("/api/snapshots")
def list_snapshots():
    """Список всех снимков с агрегатами (count runs, last_run_at, total trees)."""
    return db.list_snapshots()


@app.delete("/api/snapshots/{image_id}")
def delete_snapshot(image_id: str):
    """Удалить снимок целиком: запись в БД, runs (cascade), детекции (cascade), файл с диска."""
    snap = db.load_snapshot(image_id)
    if not snap:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    ok = db.delete_snapshot(image_id)
    files_removed = 0
    for p in UPLOADS.glob(f"{image_id}.*"):
        try:
            p.unlink()
            files_removed += 1
        except Exception as e:
            log.warning("Failed to remove %s: %s", p, e)
    log.info("Deleted snapshot %s (db=%s, files=%d)", image_id, ok, files_removed)
    return {"deleted": ok, "image_id": image_id, "files_removed": files_removed}


@app.delete("/api/runs/{job_id}")
def delete_run(job_id: str):
    """Удалить один прогон (не трогая снимок)."""
    ok = db.delete_run(job_id)
    if not ok:
        raise HTTPException(404, f"Unknown job_id {job_id}")
    log.info("Deleted run %s", job_id)
    return {"deleted": True, "job_id": job_id}


@app.get("/api/detections")
def aggregate_detections(
    nw_lat: float | None = None,
    nw_lng: float | None = None,
    se_lat: float | None = None,
    se_lng: float | None = None,
    model: str | None = None,
    min_confidence: float = 0.0,
    include_hidden: bool = False,
    limit: int = 50_000,
):
    """Главный aggregate-запрос для городской карты. По умолчанию пропускает
    детекции из hidden=true scan-сессий — пользователь временно скрыл их
    через UI-toggle. `?include_hidden=true` возвращает всё."""
    bbox = None
    if all(v is not None for v in (nw_lat, nw_lng, se_lat, se_lng)):
        bbox = (nw_lat, nw_lng, se_lat, se_lng)
    models = [model] if model else None
    detections = db.query_detections(
        bbox=bbox, models=models, min_confidence=min_confidence,
        include_hidden_scans=include_hidden, limit=limit,
    )
    return {
        "count": len(detections),
        "detections": detections,
    }


@app.get("/api/aggregate/stats")
def aggregate_statistics(
    nw_lat: float | None = None,
    nw_lng: float | None = None,
    se_lat: float | None = None,
    se_lng: float | None = None,
):
    """Сводка по всему городу/области для главного экрана."""
    bbox = None
    if all(v is not None for v in (nw_lat, nw_lng, se_lat, se_lng)):
        bbox = (nw_lat, nw_lng, se_lat, se_lng)
    return db.aggregate_stats(bbox=bbox)


# ============ Static frontend ============

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = FRONTEND / "index.html"
        if not idx.exists():
            return HTMLResponse(
                "<h1>Frontend not found</h1><p>frontend/index.html missing</p>",
                status_code=404,
            )
        return FileResponse(idx)

    @app.get("/{filename:path}")
    def serve_static(filename: str):
        # Защищаем API и docs
        if filename.startswith(("api/", "docs", "openapi.json", "redoc")):
            raise HTTPException(404, "Not found")
        # Path-traversal guard: resolve the requested path and verify it
        # stays inside FRONTEND. Otherwise a request like
        # `GET /../../storage/app.db` would happily stream the SQLite DB.
        try:
            target = (FRONTEND / filename).resolve()
        except (OSError, RuntimeError):
            raise HTTPException(404, "Not found")
        try:
            target.relative_to(FRONTEND_ROOT)
        except ValueError:
            log.warning(
                "Path traversal attempt blocked: requested=%r resolved=%s",
                filename, target,
            )
            raise HTTPException(404, "Not found")
        if target.is_file():
            return FileResponse(target)
        raise HTTPException(404, f"File {filename} not found")


# ============ helpers ============


def _build_meta(path: Path, image_id: str, original_name: str, size_bytes: int) -> ImageMeta:
    """Извлекает width/height (+ GeoTIFF метаданные если возможно)."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            meta = load_geotiff_meta(path)
            meta.image_id = image_id
            meta.filename = original_name
            return meta
        except Exception as e:
            log.warning("GeoTIFF parse failed (%s), falling back to PIL", e)

    # Regular image
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return ImageMeta(
        image_id=image_id,
        filename=original_name,
        width=w,
        height=h,
        size_bytes=size_bytes,
        is_geotiff=False,
    )


# Stable schema for stats: always return the same keys so the frontend can
# safely call .toFixed() / arithmetic on the values without crashing on
# `undefined`. Empty-detections runs now keep the same shape.
_EMPTY_STATS: dict = {
    "tree_count": 0,
    "avg_confidence": None,
    "min_confidence": None,
    "max_confidence": None,
    "avg_crown_area_px": None,
    "coverage_pct": None,
    "avg_crown_area_m2": None,
    "total_crown_area_m2": None,
    "analyzed_area_ha": None,
}


def _compute_stats(detections: list, meta: ImageMeta, ctx: GeoContext) -> dict:
    stats: dict = dict(_EMPTY_STATS)
    if not detections:
        return stats

    confs = [d.confidence for d in detections]
    crowns = [d.crown_area_px for d in detections if d.crown_area_px is not None]

    stats.update({
        "tree_count": len(detections),
        "avg_confidence": round(sum(confs) / len(confs), 3),
        "min_confidence": round(min(confs), 3),
        "max_confidence": round(max(confs), 3),
    })

    if crowns:
        total_crown_px = sum(crowns)
        stats["avg_crown_area_px"] = round(total_crown_px / len(crowns), 1)
        stats["coverage_pct"] = round(100 * total_crown_px / (meta.width * meta.height), 2)

        if ctx.pixel_size_m:
            m2_per_px = ctx.pixel_size_m ** 2
            stats["avg_crown_area_m2"] = round((total_crown_px / len(crowns)) * m2_per_px, 2)
            stats["total_crown_area_m2"] = round(total_crown_px * m2_per_px, 1)
            stats["analyzed_area_ha"] = round((meta.width * meta.height) * m2_per_px / 10_000, 3)

    return stats
