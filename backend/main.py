"""FastAPI app: roots, model registry, статика frontend.

Запуск:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Открыть http://localhost:8000 — встроенный UI.
API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .export import to_csv, to_geojson, to_standalone_html
from .geo import GeoContext, annotate_detections, build_context, load_geotiff_meta
from .models import ModelRegistry
from .models.base import ModelAdapter
from .models.deepforest_adapter import DeepForestAdapter
from .models.ensemble_adapter import EnsembleAdapter
from .models.yolo_adapter import YOLOAdapter
from .schemas import (
    GeoMode,
    GeoParams,
    HistoryEntry,
    ImageMeta,
    ModelKind,
    PredictRequest,
    PredictResult,
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

UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# ============ App ============

app = FastAPI(
    title="Astana Tree Detection",
    description="End-to-end система автоматической инвентаризации городских деревьев Астаны",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


_load_models()

# ============ In-memory job store ============
# В продакшне — заменить на SQLite / Redis. Для прототипа достаточно.
_jobs: dict[str, PredictResult] = {}
_meta: dict[str, ImageMeta] = {}
_history: list[HistoryEntry] = []


# ============ Routes: status ============


@app.get("/api/status")
def status() -> dict:
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
        "uploads": len(list(UPLOADS.glob("*"))),
        "history_size": len(_history),
    }


# ============ Routes: upload ============


@app.post("/api/upload", response_model=ImageMeta)
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Empty filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported extension {ext}. Allowed: {sorted(ALLOWED_EXTS)}")

    image_id = uuid.uuid4().hex[:12]
    saved_path = UPLOADS / f"{image_id}{ext}"

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(contents)} bytes, max {MAX_UPLOAD_BYTES})")
    saved_path.write_bytes(contents)

    meta = _build_meta(saved_path, image_id, file.filename, len(contents))
    _meta[image_id] = meta
    log.info("Uploaded %s → %s (%dx%d, %s)", file.filename, saved_path.name, meta.width, meta.height,
             "GeoTIFF" if meta.is_geotiff else "regular")
    return meta


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    meta = _meta.get(image_id)
    if not meta:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    p = UPLOADS / Path(meta.filename).with_stem(image_id).name  # combine id with ext
    # safer: glob for image_id.*
    matches = list(UPLOADS.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(404, "File missing on disk")
    return FileResponse(matches[0])


@app.get("/api/image/{image_id}/meta", response_model=ImageMeta)
def get_image_meta(image_id: str):
    meta = _meta.get(image_id)
    if not meta:
        raise HTTPException(404, f"Unknown image_id {image_id}")
    return meta


# ============ Routes: predict ============


@app.post("/api/predict", response_model=PredictResult)
def predict(req: PredictRequest):
    meta = _meta.get(req.image_id)
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
    detections = adapter.predict(image_path, confidence=req.confidence)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Geo annotation
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
    _jobs[job_id] = result

    # History entry
    from datetime import datetime

    _history.append(
        HistoryEntry(
            image_id=req.image_id,
            filename=meta.filename,
            date=datetime.utcnow().isoformat() + "Z",
            model=req.model,
            tree_count=len(detections),
            coverage_pct=stats.get("coverage_pct"),
        )
    )
    log.info("Done %s: %d detections in %d ms", adapter.kind.value, len(detections), duration_ms)
    return result


@app.get("/api/result/{job_id}", response_model=PredictResult)
def get_result(job_id: str):
    result = _jobs.get(job_id)
    if not result:
        raise HTTPException(404, f"Unknown job_id {job_id}")
    return result


# ============ Routes: export ============


@app.post("/api/export/{job_id}/{fmt}")
def export(job_id: str, fmt: str):
    result = _jobs.get(job_id)
    if not result:
        raise HTTPException(404, f"Unknown job_id {job_id}")

    meta = _meta.get(result.image_id)

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


# ============ Routes: history ============


@app.get("/api/history", response_model=list[HistoryEntry])
def history(limit: int = 20):
    return list(reversed(_history[-limit:]))


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
        target = FRONTEND / filename
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


def _compute_stats(detections: list, meta: ImageMeta, ctx: GeoContext) -> dict:
    if not detections:
        return {"tree_count": 0}

    confs = [d.confidence for d in detections]
    crowns = [d.crown_area_px for d in detections if d.crown_area_px is not None]

    stats: dict = {
        "tree_count": len(detections),
        "avg_confidence": round(sum(confs) / len(confs), 3),
        "min_confidence": round(min(confs), 3),
        "max_confidence": round(max(confs), 3),
    }

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
