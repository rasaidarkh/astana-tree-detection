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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from . import db
from .export import to_csv, to_geojson, to_standalone_html
from .geo import GeoContext, annotate_detections, build_context, load_geotiff_meta
from .map_capture import capture_bbox, save_capture
from .models import ModelRegistry
from .region_scan import DEFAULT_MAX_SUBREGIONS, plan_scan
from .models.base import ModelAdapter
from .models.deepforest_adapter import DeepForestAdapter
from .models.deepforest_sam2_adapter import DeepForestSAM2Adapter
from .models.ensemble_adapter import EnsembleAdapter
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
    zoom: int = Field(18, ge=1, le=19)


class ScanRegionRequest(BaseModel):
    """Auto-Zoom Region Scan request — см. region_scan.py.

    Пользователь рисует прямоугольник любого размера, сервер сам выбирает
    `zoom` (по умолчанию 19 — максимальный для Esri, ~0.3 м/px) и при
    необходимости дробит bbox в сетку под-регионов. На каждом запускается
    обычный capture+predict; результаты сохраняются как отдельные snapshots
    в БД и автоматически появляются в city-aggregate view.
    """
    nw: LatLng
    se: LatLng
    zoom: int = Field(19, ge=14, le=19)
    model: ModelKind = ModelKind.YOLO
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    max_subregions: int = Field(DEFAULT_MAX_SUBREGIONS, ge=1, le=25)

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
    title="Astana Tree Detection",
    description="End-to-end система автоматической инвентаризации городских деревьев Астаны",
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


def _load_models() -> None:
    """Регистрируем все доступные адаптеры. Веса грузятся лениво при первом predict."""
    yolo_path = WEIGHTS / "yolo_satellite.pt"
    if yolo_path.exists():
        yolo = YOLOAdapter(weights_path=str(yolo_path))
        registry.register(yolo)
        log.info("YOLO adapter registered: %s", yolo_path)
    else:
        log.warning("YOLO weights missing at %s — endpoint вернёт 503", yolo_path)

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
        filename=f"map_capture_z{req.zoom}.png",
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
        "Captured from map: bbox=(%.5f,%.5f → %.5f,%.5f) z=%d → %dx%d (%d bytes)",
        req.nw.lat, req.nw.lng, req.se.lat, req.se.lng, req.zoom, w, h, size_bytes,
    )
    return meta


@app.post("/api/scan_region")
async def scan_region(req: ScanRegionRequest):
    """Auto-Zoom Region Scan — большой bbox → сетка под-регионов на фикс. зуме,
    каждая под-область прогоняется через тот же capture+predict pipeline.

    Синхронный endpoint: возвращает только когда ВСЕ под-регионы обработаны.
    UI показывает spinner; для 1.5×1.5 км @ z19 типичное время ~30–60 сек.

    Под-регионы сохраняются в БД как отдельные snapshots — после завершения
    они подтягиваются в city-aggregate view (`GET /api/snapshots`,
    `GET /api/detections`) на общих основаниях.
    """
    adapter: Optional[ModelAdapter] = registry.get(req.model)
    if adapter is None:
        available = [k.value for k in registry.available()]
        raise HTTPException(503, f"Model {req.model.value} not available. Available: {available}")

    try:
        subs = plan_scan(
            req.nw.lat, req.nw.lng, req.se.lat, req.se.lng,
            zoom=req.zoom,
            max_subregions=req.max_subregions,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    log.info(
        "scan_region: bbox=(%.5f,%.5f → %.5f,%.5f) z=%d → %d sub-region(s), model=%s",
        req.nw.lat, req.nw.lng, req.se.lat, req.se.lng, req.zoom,
        len(subs), req.model.value,
    )

    sub_results = []
    total_trees = 0
    t_total = time.perf_counter()

    for sub in subs:
        sub_label = f"r{sub.row}c{sub.col}"
        try:
            cap = await asyncio.to_thread(
                capture_bbox,
                sub.nw_lat, sub.nw_lng, sub.se_lat, sub.se_lng, req.zoom,
            )
        except ValueError as e:
            log.warning("scan_region sub-%s capture rejected: %s", sub_label, e)
            sub_results.append({
                "row": sub.row, "col": sub.col,
                "error": str(e),
                "sub_bbox": {
                    "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
                    "se": {"lat": sub.se_lat, "lng": sub.se_lng},
                },
            })
            continue
        except (urllib.error.URLError, urllib.error.HTTPError, IOError) as e:
            log.warning("scan_region sub-%s tile fetch failed: %s", sub_label, e)
            sub_results.append({
                "row": sub.row, "col": sub.col,
                "error": f"tile fetch failed: {e}",
                "sub_bbox": {
                    "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
                    "se": {"lat": sub.se_lat, "lng": sub.se_lng},
                },
            })
            continue
        except Exception as e:
            log.exception("scan_region sub-%s capture failed unexpectedly", sub_label)
            sub_results.append({
                "row": sub.row, "col": sub.col,
                "error": f"capture failed: {e}",
                "sub_bbox": {
                    "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
                    "se": {"lat": sub.se_lat, "lng": sub.se_lng},
                },
            })
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
            filename=f"scan_z{req.zoom}_{sub_label}.png",
            width=w, height=h, size_bytes=size_bytes,
            is_geotiff=False,
            bounds=bounds,
        )
        db.save_snapshot(meta)

        t_sub = time.perf_counter()
        try:
            detections = await asyncio.to_thread(
                adapter.predict, str(out_path), confidence=req.confidence,
            )
        except Exception as e:
            log.exception("scan_region sub-%s predict failed", sub_label)
            sub_results.append({
                "row": sub.row, "col": sub.col,
                "snapshot_id": image_id,
                "error": f"predict failed: {e}",
                "sub_bbox": {
                    "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
                    "se": {"lat": sub.se_lat, "lng": sub.se_lng},
                },
            })
            continue

        ctx = build_context(
            width=w, height=h,
            geo=GeoParams(mode=GeoMode.CORNERS_2, corners_2=bounds),
            geotiff_path=None,
        )
        detections = annotate_detections(detections, ctx)
        stats = _compute_stats(detections, meta, ctx)
        duration_ms = int((time.perf_counter() - t_sub) * 1000)

        job_id = uuid.uuid4().hex[:12]
        result = PredictResult(
            job_id=job_id, image_id=image_id, model=req.model,
            detections=detections, duration_ms=duration_ms, stats=stats,
        )
        db.save_run(result, geo_mode=GeoMode.CORNERS_2.value, confidence=req.confidence)
        total_trees += len(detections)
        sub_results.append({
            "row": sub.row, "col": sub.col,
            "snapshot_id": image_id,
            "job_id": job_id,
            "tree_count": len(detections),
            "duration_ms": duration_ms,
            "sub_bbox": {
                "nw": {"lat": sub.nw_lat, "lng": sub.nw_lng},
                "se": {"lat": sub.se_lat, "lng": sub.se_lng},
            },
        })
        log.info(
            "scan_region sub-%s: %d trees in %d ms (snapshot=%s, job=%s)",
            sub_label, len(detections), duration_ms, image_id, job_id,
        )

    total_duration_ms = int((time.perf_counter() - t_total) * 1000)
    ok_count = sum(1 for r in sub_results if "error" not in r)
    log.info(
        "scan_region done: %d/%d sub-regions ok, %d trees, %d ms total",
        ok_count, len(sub_results), total_trees, total_duration_ms,
    )
    return {
        "sub_count": len(sub_results),
        "ok_count": ok_count,
        "total_trees": total_trees,
        "duration_ms": total_duration_ms,
        "zoom": req.zoom,
        "model": req.model.value,
        "bbox": {
            "nw": {"lat": req.nw.lat, "lng": req.nw.lng},
            "se": {"lat": req.se.lat, "lng": req.se.lng},
        },
        "sub_regions": sub_results,
    }


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
    limit: int = 50_000,
):
    """Главный aggregate-запрос для городской карты."""
    bbox = None
    if all(v is not None for v in (nw_lat, nw_lng, se_lat, se_lng)):
        bbox = (nw_lat, nw_lng, se_lat, se_lng)
    models = [model] if model else None
    detections = db.query_detections(
        bbox=bbox, models=models, min_confidence=min_confidence, limit=limit,
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
