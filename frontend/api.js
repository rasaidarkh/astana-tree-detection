/* API client for Astana Tree Detection backend.
 * Все вызовы идут через window.api. */

(function () {
  const BASE = window.API_BASE || "";

  async function _json(res) {
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch { detail = await res.text(); }
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
    return apiDetections
      .filter((d) => d.lat != null && d.lng != null)
      .map((d) => ({
        id: d.id,
        lat: d.lat,
        lng: d.lng,
        confidence: d.confidence,
        crown: d.crown_diameter_m != null ? d.crown_diameter_m : 4.0,
        box: d.box,
        mask_polygon: d.mask_polygon,
      }));
  };

  window.api = api;
})();
