"""Экспорт результатов: GeoJSON, CSV, standalone HTML map."""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime
from typing import Optional

from .schemas import Detection, ImageMeta


def to_geojson(detections: list[Detection], image: Optional[ImageMeta] = None) -> dict:
    """FeatureCollection из всех детекций с lat/lng. Без GPS — пустая коллекция."""
    features = []
    for det in detections:
        if det.lat is None or det.lng is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [det.lng, det.lat]},
                "properties": {
                    "id": det.id,
                    "confidence": round(det.confidence, 3),
                    "crown_diameter_m": det.crown_diameter_m,
                    "crown_area_px": det.crown_area_px,
                    "label": det.label,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_image": image.filename if image else None,
            "feature_count": len(features),
        },
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }


def to_csv(detections: list[Detection]) -> str:
    """Plain CSV — координаты, confidence, размер кроны."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "lat", "lng", "confidence", "crown_diameter_m", "px_x", "px_y"])
    for det in detections:
        writer.writerow(
            [
                det.id,
                round(det.lat, 7) if det.lat is not None else "",
                round(det.lng, 7) if det.lng is not None else "",
                round(det.confidence, 3),
                det.crown_diameter_m if det.crown_diameter_m is not None else "",
                round(det.box.cx, 1),
                round(det.box.cy, 1),
            ]
        )
    return buf.getvalue()


def to_standalone_html(detections: list[Detection], title: str = "Astana Trees") -> str:
    """Self-contained HTML с Leaflet, без внешних зависимостей кроме CDN."""
    # HTML-escape the title — it is interpolated directly into <title> below,
    # so an uploaded filename containing `<script>…</script>` used to be
    # executable in the exported file.
    safe_title = html.escape(title)
    geo_features = []
    for det in detections:
        if det.lat is None or det.lng is None:
            continue
        geo_features.append(
            {
                "id": det.id,
                "lat": det.lat,
                "lng": det.lng,
                "conf": round(det.confidence, 3),
                "crown_m": det.crown_diameter_m,
            }
        )

    if geo_features:
        center_lat = sum(t["lat"] for t in geo_features) / len(geo_features)
        center_lng = sum(t["lng"] for t in geo_features) / len(geo_features)
    else:
        center_lat, center_lng = 51.1605, 71.4704

    trees_json = json.dumps(geo_features)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>{safe_title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  body,html{{margin:0;padding:0;height:100%;font-family:system-ui}}
  #map{{position:absolute;inset:0}}
  .panel{{position:absolute;top:10px;right:10px;background:#fff;padding:10px 14px;
         border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:1000;font-size:13px}}
  .panel h2{{margin:0 0 6px;font-size:15px;color:#0F6E56}}
  .legend{{position:absolute;bottom:20px;left:10px;background:#fff;padding:8px 12px;
          border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:1000;font-size:12px}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
</style>
</head><body>
<div id="map"></div>
<div class="panel">
  <h2>Astana Tree Detection</h2>
  <div>Total: <b>{len(geo_features)}</b></div>
  <div style="font-size:11px;color:#666;margin-top:4px">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
</div>
<div class="legend">
  <div><span class="dot" style="background:#0F6E56"></span>High &gt; 70%</div>
  <div><span class="dot" style="background:#5DCAA5"></span>Medium 50-70%</div>
  <div><span class="dot" style="background:#EF9F27"></span>Low &lt; 50%</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const trees = {trees_json};
const map = L.map('map').setView([{center_lat}, {center_lng}], 17);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{maxZoom:19, attribution:'Esri'}}).addTo(map);
const color = c => c > .7 ? '#0F6E56' : c > .5 ? '#5DCAA5' : '#EF9F27';
trees.forEach(t => {{
  L.circleMarker([t.lat, t.lng], {{
    radius: 6, color:'#fff', weight:1.5, fillColor:color(t.conf), fillOpacity:.92
  }}).bindPopup(
    `<b>Tree #${{t.id}}</b><br>Confidence: ${{(t.conf*100).toFixed(1)}}%<br>` +
    `Crown: ${{t.crown_m ? t.crown_m+' m' : '—'}}<br>` +
    `${{t.lat.toFixed(6)}}°, ${{t.lng.toFixed(6)}}°`
  ).addTo(map);
}});
if (trees.length) map.fitBounds(trees.map(t => [t.lat, t.lng]), {{padding:[40,40]}});
</script></body></html>"""
