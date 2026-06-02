/* API client for Astana Tree Detection backend.
 * Все вызовы идут через window.api. */

(function () {
  const BASE = window.API_BASE || "";

  async function _json(res) {
    if (!res.ok) {
      const text = await res.text();
      let detail;
      try { detail = JSON.parse(text).detail; } catch { detail = text; }
      throw new Error(`API ${res.status}: ${detail || res.statusText}`);
    }
    return res.json();
  }

  const api = {
    async status() {
      return _json(await fetch(`${BASE}/api/status`));
    },

    async upload(file) {
      const fd = new FormData();
      fd.append("file", file);
      return _json(await fetch(`${BASE}/api/upload`, { method: "POST", body: fd }));
    },

    async captureFromMap({ nw, se, zoom = 18, provider = "esri" }) {
      return _json(await fetch(`${BASE}/api/capture_from_map`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nw, se, zoom, provider }),
      }));
    },

    // Auto-Zoom Region Scan — большой bbox любого размера, сервер дробит
    // на сетку под-регионов на фикс. zoom (по умолчанию 19) и прогоняет каждый.
    // Под-снимки сохраняются как обычные snapshots — после возврата
    // листинг /api/snapshots и /api/detections содержит свежие записи.
    async scanRegion({ nw, se, zoom = 19, model = "yolo", confidence = 0.25, maxSubregions = 9, provider = "esri" }) {
      return _json(await fetch(`${BASE}/api/scan_region`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nw, se, zoom, model, confidence,
          max_subregions: maxSubregions,
          provider,
        }),
      }));
    },

    // Streaming variant — отдаёт прогресс по мере обработки каждого под-региона.
    // Callback вызывается на каждое событие из NDJSON-потока:
    //   {type:"plan", sub_count, sub_regions:[{row,col,nw,se},...]}
    //   {type:"capturing"|"capture_done"|"predicting", row, col, ...}
    //   {type:"sub_complete", row, col, detections:[...], tree_count, snapshot_id, job_id, ...}
    //   {type:"sub_error", row, col, stage, error, ...}
    //   {type:"done", total_trees, duration_ms}
    //   {type:"fatal", error}
    async scanRegionStream({ nw, se, zoom = 19, model = "yolo", confidence = 0.25, maxSubregions = 9, provider = "esri", polygon = null }, onEvent, { signal } = {}) {
      const body = {
        nw, se, zoom, model, confidence,
        max_subregions: maxSubregions, provider,
      };
      // polygon = [{lat, lng}, ...] ≥ 3 точек. Бэк делает point-in-polygon
      // фильтр после annotate_detections.
      if (polygon && polygon.length >= 3) body.polygon = polygon;
      const resp = await fetch(`${BASE}/api/scan_region/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      if (!resp.ok) {
        // pre-flight 4xx/5xx (model недоступна, bbox перевёрнут, scan слишком большой)
        const txt = await resp.text();
        let detail;
        try { detail = JSON.parse(txt).detail; } catch { detail = txt; }
        throw new Error(`API ${resp.status}: ${detail || resp.statusText}`);
      }
      if (!resp.body) throw new Error("ReadableStream not supported");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let lines = buf.split("\n");
          buf = lines.pop();  // последняя строка может быть неполной
          for (const line of lines) {
            const t = line.trim();
            if (!t) continue;
            try { onEvent(JSON.parse(t)); }
            catch (e) { console.warn("Skipping malformed NDJSON line:", t, e); }
          }
        }
        // flush последнего буфера если он не пустой
        if (buf.trim()) {
          try { onEvent(JSON.parse(buf)); } catch (e) {/* ignore */}
        }
      } finally {
        try { reader.releaseLock(); } catch {/* ignore */}
      }
    },

    // Список tile-провайдеров — фронт подтягивает один раз и строит
    // dropdown + Leaflet base layer URL из того же источника что backend.
    async providers() {
      return _json(await fetch(`${BASE}/api/providers`));
    },

    imageUrl(imageId) {
      return `${BASE}/api/image/${imageId}`;
    },

    async predict({ image_id, model = "yolo", confidence = 0.25, geo = null }) {
      const body = {
        image_id,
        model,
        confidence,
        geo: geo || { mode: "none" },
      };
      return _json(await fetch(`${BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }));
    },

    async result(jobId) {
      return _json(await fetch(`${BASE}/api/result/${jobId}`));
    },

    async history(limit = 20) {
      return _json(await fetch(`${BASE}/api/history?limit=${limit}`));
    },

    exportUrl(jobId, fmt) {
      return `${BASE}/api/export/${jobId}/${fmt}`;
    },

    async exportFile(jobId, fmt) {
      const res = await fetch(`${BASE}/api/export/${jobId}/${fmt}`, { method: "POST" });
      if (!res.ok) throw new Error(`Export failed: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${jobId}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    },
  };

  // Преобразование Detection из API в формат, ожидаемый существующим UI (lat/lng/confidence/crown)
  api.adaptDetectionsForUI = function (apiDetections) {
    return apiDetections.map((d) => ({
      id: d.id,
      lat: d.lat,
      lng: d.lng,
      confidence: d.confidence,
      crown: d.crown_diameter_m != null ? d.crown_diameter_m : 4.0,
      box: d.box,
      mask_polygon: d.mask_polygon,
      mask_polygon_geo: d.mask_polygon_geo,  // [[lat, lng], ...] для рендера на Leaflet
      box_geo: d.box_geo,                    // [[lat, lng] × 4 corners]
    }));
  };

  // ============ City-aggregate API ============
  api.listSnapshots = async function () {
    const r = await fetch(`${BASE}/api/snapshots`);
    if (!r.ok) throw new Error(`/api/snapshots HTTP ${r.status}`);
    return await r.json();
  };
  api.getDetections = async function ({ bbox, model, minConfidence, limit } = {}) {
    const params = new URLSearchParams();
    if (bbox) {
      params.set("nw_lat", bbox.nw.lat); params.set("nw_lng", bbox.nw.lng);
      params.set("se_lat", bbox.se.lat); params.set("se_lng", bbox.se.lng);
    }
    if (model) params.set("model", model);
    if (minConfidence != null) params.set("min_confidence", minConfidence);
    if (limit != null) params.set("limit", limit);
    const r = await fetch(`${BASE}/api/detections?${params}`);
    if (!r.ok) throw new Error(`/api/detections HTTP ${r.status}`);
    return await r.json();
  };
  api.aggregateStats = async function (bbox) {
    const params = new URLSearchParams();
    if (bbox) {
      params.set("nw_lat", bbox.nw.lat); params.set("nw_lng", bbox.nw.lng);
      params.set("se_lat", bbox.se.lat); params.set("se_lng", bbox.se.lng);
    }
    const r = await fetch(`${BASE}/api/aggregate/stats?${params}`);
    if (!r.ok) throw new Error(`/api/aggregate/stats HTTP ${r.status}`);
    return await r.json();
  };
  api.deleteSnapshot = async function (imageId) {
    const r = await fetch(`${BASE}/api/snapshots/${imageId}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`DELETE snapshot HTTP ${r.status}`);
    return await r.json();
  };
  api.renameSnapshot = async function (imageId, name) {
    const r = await fetch(`${BASE}/api/snapshots/${imageId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error(`PATCH snapshot HTTP ${r.status}`);
    return await r.json();
  };

  // ============ Scan sessions ============
  api.listScans = async function () {
    const r = await fetch(`${BASE}/api/scans`);
    if (!r.ok) throw new Error(`/api/scans HTTP ${r.status}`);
    return await r.json();
  };
  api.deleteScan = async function (scanId) {
    const r = await fetch(`${BASE}/api/scans/${scanId}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`DELETE scan HTTP ${r.status}`);
    return await r.json();
  };
  api.renameScan = async function (scanId, name) {
    const r = await fetch(`${BASE}/api/scans/${scanId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error(`PATCH scan HTTP ${r.status}`);
    return await r.json();
  };
  api.setScanHidden = async function (scanId, hidden) {
    const r = await fetch(`${BASE}/api/scans/${scanId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hidden }),
    });
    if (!r.ok) throw new Error(`PATCH scan HTTP ${r.status}`);
    return await r.json();
  };

  // Aggregate API возвращает {lat, lng, confidence, crown_diameter_m, mask_polygon_geo, box_geo,
  //   model, job_id, image_id, local_id}. Адаптируем под UI-формат `tree`.
  api.adaptAggregateDetections = function (apiList) {
    return apiList.map((d) => ({
      id: d.id,                              // глобальный db id
      local_id: d.local_id,                  // # внутри прогона
      lat: d.lat, lng: d.lng,
      confidence: d.confidence,
      crown: d.crown_diameter_m != null ? d.crown_diameter_m : 4.0,
      mask_polygon_geo: d.mask_polygon_geo,
      box_geo: d.box_geo,
      model: d.model,
      image_id: d.image_id,
      job_id: d.job_id,
      scan_session_id: d.scan_session_id,   // needed to group crowns per scan/park
    }));
  };

  window.api = api;
})();
