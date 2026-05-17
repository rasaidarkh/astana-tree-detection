"""SQLite persistence: snapshots / runs / detections.

Зачем: in-memory store теряет всё при рестарте бэка, и нет агрегата
по всем загруженным снимкам — главная функция urban-mapping приложения.

Файл БД: storage/app.db. ON DELETE CASCADE — снимок удалили, его прогоны
и детекции уходят за ним.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from .schemas import (
    BBox,
    Corners2,
    Detection,
    ImageMeta,
    LatLng,
    ModelKind,
    PredictResult,
)


_DB_PATH: Optional[Path] = None
_lock = Lock()


def init_db(path: Path) -> None:
    global _DB_PATH
    _DB_PATH = path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
          image_id      TEXT PRIMARY KEY,
          filename      TEXT NOT NULL,
          width         INTEGER NOT NULL,
          height        INTEGER NOT NULL,
          size_bytes    INTEGER NOT NULL,
          is_geotiff    INTEGER NOT NULL DEFAULT 0,
          crs           TEXT,
          pixel_size_m  REAL,
          nw_lat REAL, nw_lng REAL, se_lat REAL, se_lng REAL,
          created_at    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          job_id        TEXT PRIMARY KEY,
          image_id      TEXT NOT NULL,
          model         TEXT NOT NULL,
          confidence    REAL NOT NULL,
          geo_mode      TEXT NOT NULL,
          duration_ms   INTEGER NOT NULL,
          tree_count    INTEGER NOT NULL,
          stats_json    TEXT NOT NULL,
          created_at    TEXT NOT NULL,
          FOREIGN KEY (image_id) REFERENCES snapshots(image_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS detections (
          id                INTEGER PRIMARY KEY AUTOINCREMENT,
          job_id            TEXT NOT NULL,
          local_id          INTEGER NOT NULL,
          lat REAL, lng REAL,
          box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
          confidence        REAL NOT NULL,
          crown_diameter_m  REAL,
          crown_area_px     REAL,
          mask_polygon_geo  TEXT,
          box_geo           TEXT,
          FOREIGN KEY (job_id) REFERENCES runs(job_id) ON DELETE CASCADE
        );
        -- Auto-Zoom Scan-сессии: одно "большое" сканирование = N sub-region snapshots.
        -- runs ссылается через nullable scan_session_id (старые runs остаются NULL).
        -- polygon_json содержит вершины пользовательского полигона если scan был
        -- по полигону (а не по простому axis-aligned bbox).
        CREATE TABLE IF NOT EXISTS scan_sessions (
          id            TEXT PRIMARY KEY,
          nw_lat        REAL NOT NULL,
          nw_lng        REAL NOT NULL,
          se_lat        REAL NOT NULL,
          se_lng        REAL NOT NULL,
          zoom          INTEGER NOT NULL,
          provider      TEXT NOT NULL,
          model         TEXT NOT NULL,
          polygon_json  TEXT,
          status        TEXT NOT NULL DEFAULT 'running',
          sub_count     INTEGER NOT NULL DEFAULT 0,
          ok_count      INTEGER NOT NULL DEFAULT 0,
          total_trees   INTEGER NOT NULL DEFAULT 0,
          duration_ms   INTEGER NOT NULL DEFAULT 0,
          created_at    TEXT NOT NULL,
          completed_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_image_id    ON runs(image_id);
        CREATE INDEX IF NOT EXISTS idx_runs_created_at  ON runs(created_at);
        CREATE INDEX IF NOT EXISTS idx_det_job          ON detections(job_id);
        CREATE INDEX IF NOT EXISTS idx_det_latlng       ON detections(lat, lng);
        CREATE INDEX IF NOT EXISTS idx_scans_created    ON scan_sessions(created_at);
        """)
        # Идемпотентная миграция: добавляем scan_session_id к существующим
        # runs без потери данных. SQLite не имеет IF NOT EXISTS для ADD COLUMN,
        # поэтому проверяем через pragma_table_info.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
        if "scan_session_id" not in cols:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN scan_session_id TEXT REFERENCES scan_sessions(id) ON DELETE SET NULL"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_scan_session ON runs(scan_session_id)")
        conn.commit()


def _connect() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("DB not initialized — call init_db first")
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============ Snapshots ============

def save_snapshot(meta: ImageMeta) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO snapshots
              (image_id, filename, width, height, size_bytes, is_geotiff,
               crs, pixel_size_m,
               nw_lat, nw_lng, se_lat, se_lng, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
              filename     = excluded.filename,
              width        = excluded.width,
              height       = excluded.height,
              size_bytes   = excluded.size_bytes,
              is_geotiff   = excluded.is_geotiff,
              crs          = excluded.crs,
              pixel_size_m = excluded.pixel_size_m,
              nw_lat       = excluded.nw_lat,
              nw_lng       = excluded.nw_lng,
              se_lat       = excluded.se_lat,
              se_lng       = excluded.se_lng
            """,
            (
                meta.image_id, meta.filename, meta.width, meta.height, meta.size_bytes,
                1 if meta.is_geotiff else 0,
                meta.crs, meta.pixel_size_m,
                meta.bounds.nw.lat if meta.bounds else None,
                meta.bounds.nw.lng if meta.bounds else None,
                meta.bounds.se.lat if meta.bounds else None,
                meta.bounds.se.lng if meta.bounds else None,
                _now(),
            ),
        )
        conn.commit()


def load_snapshot(image_id: str) -> Optional[ImageMeta]:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM snapshots WHERE image_id = ?", (image_id,)
        ).fetchone()
    if row is None:
        return None
    bounds = None
    if row["nw_lat"] is not None:
        bounds = Corners2(
            nw=LatLng(lat=row["nw_lat"], lng=row["nw_lng"]),
            se=LatLng(lat=row["se_lat"], lng=row["se_lng"]),
        )
    return ImageMeta(
        image_id=row["image_id"],
        filename=row["filename"],
        width=row["width"],
        height=row["height"],
        size_bytes=row["size_bytes"],
        is_geotiff=bool(row["is_geotiff"]),
        crs=row["crs"],
        pixel_size_m=row["pixel_size_m"],
        bounds=bounds,
    )


def list_snapshots() -> list[dict]:
    """С агрегатами: количество прогонов, общее число деревьев, дата последнего ран'а."""
    sql = """
        SELECT
          s.*,
          COUNT(DISTINCT r.job_id)        AS run_count,
          COALESCE(SUM(r.tree_count), 0)  AS total_trees,
          MAX(r.created_at)               AS last_run_at,
          (SELECT model FROM runs r2 WHERE r2.image_id = s.image_id
            ORDER BY r2.created_at DESC LIMIT 1) AS last_model
        FROM snapshots s
        LEFT JOIN runs r ON r.image_id = s.image_id
        GROUP BY s.image_id
        ORDER BY s.created_at DESC
    """
    with _lock, _connect() as conn:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    return rows


def delete_snapshot(image_id: str) -> bool:
    """CASCADE удалит runs и detections."""
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM snapshots WHERE image_id = ?", (image_id,))
        conn.commit()
        return cur.rowcount > 0


# ============ Runs / Detections ============

def save_run(result: PredictResult, geo_mode: str, confidence: float, scan_session_id: Optional[str] = None) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs
              (job_id, image_id, model, confidence, geo_mode, duration_ms,
               tree_count, stats_json, created_at, scan_session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.job_id, result.image_id, result.model.value, confidence,
                geo_mode, result.duration_ms, len(result.detections),
                json.dumps(result.stats, ensure_ascii=False),
                _now(),
                scan_session_id,
            ),
        )
        for det in result.detections:
            conn.execute(
                """
                INSERT INTO detections
                  (job_id, local_id, lat, lng, box_x1, box_y1, box_x2, box_y2,
                   confidence, crown_diameter_m, crown_area_px,
                   mask_polygon_geo, box_geo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.job_id, det.id, det.lat, det.lng,
                    det.box.x1, det.box.y1, det.box.x2, det.box.y2,
                    det.confidence, det.crown_diameter_m, det.crown_area_px,
                    json.dumps(det.mask_polygon_geo) if det.mask_polygon_geo else None,
                    json.dumps(det.box_geo) if det.box_geo else None,
                ),
            )
        conn.commit()


def load_run(job_id: str) -> Optional[PredictResult]:
    with _lock, _connect() as conn:
        run = conn.execute(
            "SELECT * FROM runs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if run is None:
            return None
        det_rows = conn.execute(
            "SELECT * FROM detections WHERE job_id = ? ORDER BY local_id", (job_id,)
        ).fetchall()
    detections = [_row_to_detection(r) for r in det_rows]
    return PredictResult(
        job_id=run["job_id"],
        image_id=run["image_id"],
        model=ModelKind(run["model"]),
        detections=detections,
        duration_ms=run["duration_ms"],
        stats=json.loads(run["stats_json"]),
    )


def delete_run(job_id: str) -> bool:
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0


def list_recent_runs(limit: int = 50) -> list[dict]:
    sql = """
        SELECT r.*, s.filename
        FROM runs r
        JOIN snapshots s ON s.image_id = r.image_id
        ORDER BY r.created_at DESC
        LIMIT ?
    """
    with _lock, _connect() as conn:
        return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]


def _row_to_detection(r: sqlite3.Row) -> Detection:
    return Detection(
        id=r["local_id"],
        box=BBox(x1=r["box_x1"], y1=r["box_y1"], x2=r["box_x2"], y2=r["box_y2"]),
        confidence=r["confidence"],
        label="tree",
        lat=r["lat"],
        lng=r["lng"],
        crown_diameter_m=r["crown_diameter_m"],
        crown_area_px=r["crown_area_px"],
        mask_polygon_geo=json.loads(r["mask_polygon_geo"]) if r["mask_polygon_geo"] else None,
        box_geo=json.loads(r["box_geo"]) if r["box_geo"] else None,
    )


# ============ Scan sessions ============

def create_scan_session(
    session_id: str,
    nw_lat: float, nw_lng: float, se_lat: float, se_lng: float,
    zoom: int, provider: str, model: str,
    sub_count: int,
    polygon: Optional[list] = None,
) -> None:
    """Создаёт запись scan-сессии при старте /api/scan_region(_stream).
    `polygon` — список [lat,lng] вершин (если scan по полигону, иначе None)."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO scan_sessions
              (id, nw_lat, nw_lng, se_lat, se_lng, zoom, provider, model,
               polygon_json, status, sub_count, ok_count, total_trees,
               duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, 0, 0, 0, ?)
            """,
            (
                session_id, nw_lat, nw_lng, se_lat, se_lng, zoom, provider, model,
                json.dumps(polygon) if polygon else None,
                sub_count,
                _now(),
            ),
        )
        conn.commit()


def finalize_scan_session(
    session_id: str, ok_count: int, total_trees: int, duration_ms: int,
    status: str = "completed",
) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            UPDATE scan_sessions
            SET status = ?, ok_count = ?, total_trees = ?, duration_ms = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (status, ok_count, total_trees, duration_ms, _now(), session_id),
        )
        conn.commit()


def list_scan_sessions() -> list[dict]:
    """Все scan-сессии новые-вперёд + агрегаты из runs (на случай если что-то
    проскочило мимо finalize)."""
    sql = """
        SELECT s.*,
               COUNT(DISTINCT r.job_id)        AS actual_run_count,
               COALESCE(SUM(r.tree_count), 0)  AS actual_trees
        FROM scan_sessions s
        LEFT JOIN runs r ON r.scan_session_id = s.id
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """
    with _lock, _connect() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_scan_session_image_ids(session_id: str) -> list[str]:
    """image_id-ы всех snapshots внутри сессии — нужно чтобы удалить файлы
    с диска при cascade-delete."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT image_id FROM runs WHERE scan_session_id = ?",
            (session_id,),
        ).fetchall()
    return [r["image_id"] for r in rows]


def delete_scan_session(session_id: str) -> tuple[bool, list[str]]:
    """Удаляет сессию + ВСЕ её sub-snapshots (cascade: runs → detections).
    Возвращает (existed, [image_id, ...]) — image_id-ы для удаления файлов
    с диска (БД-каскад сам файлы не трогает)."""
    image_ids = get_scan_session_image_ids(session_id)
    with _lock, _connect() as conn:
        # Сначала прибиваем snapshots — runs/detections уйдут каскадом через
        # snapshot FK. После — пустая scan_session.
        for img_id in image_ids:
            conn.execute("DELETE FROM snapshots WHERE image_id = ?", (img_id,))
        cur = conn.execute("DELETE FROM scan_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0, image_ids


# ============ Aggregate view ============

def query_detections(
    bbox: tuple[float, float, float, float] | None = None,
    models: list[str] | None = None,
    image_ids: list[str] | None = None,
    min_confidence: float = 0.0,
    only_latest_run_per_image: bool = True,
    limit: int = 200_000,
) -> list[dict]:
    """Главный запрос для aggregate map. Возвращает дет-ции по фильтрам.

    bbox = (nw_lat, nw_lng, se_lat, se_lng) — географическая рамка.
    only_latest_run_per_image — при множественных прогонах одной фотки берём только
       последний (иначе на карте увидим дубли деревьев из разных прогонов).
    """
    conditions = ["d.lat IS NOT NULL", "d.confidence >= ?"]
    params: list = [min_confidence]

    if bbox is not None:
        nw_lat, nw_lng, se_lat, se_lng = bbox
        # NW.lat > SE.lat, NW.lng < SE.lng
        conditions.append("d.lat BETWEEN ? AND ?")
        params.extend([se_lat, nw_lat])
        conditions.append("d.lng BETWEEN ? AND ?")
        params.extend([nw_lng, se_lng])

    if models:
        placeholders = ",".join("?" * len(models))
        conditions.append(f"r.model IN ({placeholders})")
        params.extend(models)

    if image_ids:
        placeholders = ",".join("?" * len(image_ids))
        conditions.append(f"r.image_id IN ({placeholders})")
        params.extend(image_ids)

    if only_latest_run_per_image:
        # подзапрос: последний job_id для каждой image_id (по created_at).
        # При желании можно фильтровать с учётом modeling — оставляем простую логику.
        conditions.append("""r.job_id IN (
            SELECT job_id FROM runs r2 WHERE r2.created_at = (
                SELECT MAX(created_at) FROM runs r3 WHERE r3.image_id = r2.image_id
            )
        )""")

    sql = f"""
        SELECT
          d.id, d.local_id, d.lat, d.lng, d.confidence, d.crown_diameter_m,
          d.mask_polygon_geo, d.box_geo,
          r.model, r.job_id, r.image_id, r.created_at AS run_created_at
        FROM detections d
        JOIN runs r ON r.job_id = d.job_id
        WHERE {' AND '.join(conditions)}
        ORDER BY d.id
        LIMIT ?
    """
    params.append(limit)
    with _lock, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "local_id": r["local_id"],
            "lat": r["lat"],
            "lng": r["lng"],
            "confidence": r["confidence"],
            "crown_diameter_m": r["crown_diameter_m"],
            "mask_polygon_geo": json.loads(r["mask_polygon_geo"]) if r["mask_polygon_geo"] else None,
            "box_geo": json.loads(r["box_geo"]) if r["box_geo"] else None,
            "model": r["model"],
            "job_id": r["job_id"],
            "image_id": r["image_id"],
        })
    return out


def aggregate_stats(bbox: tuple[float, float, float, float] | None = None) -> dict:
    """Суммарка по всему городу/области: snapshots, runs, tree count, средняя conf."""
    sql_counts = """
        SELECT COUNT(DISTINCT s.image_id) AS snapshot_count,
               COUNT(DISTINCT r.job_id)   AS run_count
        FROM snapshots s
        LEFT JOIN runs r ON r.image_id = s.image_id
    """
    with _lock, _connect() as conn:
        counts = conn.execute(sql_counts).fetchone()
        det_filter = ["d.lat IS NOT NULL"]
        params: list = []
        if bbox is not None:
            nw_lat, nw_lng, se_lat, se_lng = bbox
            det_filter.append("d.lat BETWEEN ? AND ?")
            params.extend([se_lat, nw_lat])
            det_filter.append("d.lng BETWEEN ? AND ?")
            params.extend([nw_lng, se_lng])
        det_sql = f"""
            SELECT COUNT(*) AS total_trees,
                   AVG(d.confidence) AS avg_conf,
                   AVG(d.crown_diameter_m) AS avg_crown_m
            FROM detections d
            WHERE {' AND '.join(det_filter)}
        """
        det = conn.execute(det_sql, params).fetchone()
    return {
        "snapshot_count": counts["snapshot_count"] or 0,
        "run_count": counts["run_count"] or 0,
        "total_trees": det["total_trees"] or 0,
        "avg_confidence": round(det["avg_conf"], 3) if det["avg_conf"] is not None else None,
        "avg_crown_m": round(det["avg_crown_m"], 2) if det["avg_crown_m"] is not None else None,
    }
