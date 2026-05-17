/* global React, ReactDOM, L */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const ASTANA_CENTER = [51.1605, 71.4704];

// ============ ICONS ============
const Icon = ({ name, size = 16, stroke = 1.75 }) => {
  const paths = {
    upload: <><path d="M12 3v12M7 8l5-5 5 5M5 17v3a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3" /></>,
    play: <path d="M6 4l14 8-14 8z" fill="currentColor" stroke="none" />,
    tree: <><path d="M12 22V14" /><path d="M12 14c-3 0-5-2.5-5-5 0-1 .5-2 1-2.5C7.5 5 8 3 12 3s4.5 2 4 3.5c.5.5 1 1.5 1 2.5 0 2.5-2 5-5 5z" /></>,
    leaf: <><path d="M4 20c0-8 6-14 16-15-1 10-7 16-15 15z" /><path d="M4 20l9-9" /></>,
    check: <path d="M4 12l5 5L20 6" fill="none" />,
    area: <><rect x="4" y="4" width="16" height="16" rx="1" /><path d="M4 9h16M9 4v16" /></>,
    download: <><path d="M12 3v13M7 11l5 5 5-5M5 21h14" /></>,
    layers: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5M3 18l9 5 9-5" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    minus: <path d="M5 12h14" />,
    home: <><path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1z" /></>,
    expand: <><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6" /></>,
    chevron: <path d="M6 9l6 6 6-6" />,
    chevronUp: <path d="M6 15l6-6 6 6" />,
    image: <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="M21 16l-5-5-9 9" /></>,
    x: <><path d="M6 6l12 12M18 6L6 18" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></>,
    history: <><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></>,
    file: <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /></>,
    map: <><path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z" /><path d="M9 3v15M15 6v15" /></>,
    target: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="1" fill="currentColor" /></>,
    zap: <path d="M13 2L4 14h7l-1 8 9-12h-7z" />,
    globe: <><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></>,
    alert: <><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.7L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0z" /></>,
    circle: <circle cx="12" cy="12" r="5" />,
    square: <rect x="6" y="6" width="12" height="12" rx="1.5" />,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      {paths[name]}
    </svg>
  );
};

const LogoMark = ({ size = 28 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
    <rect width="32" height="32" rx="7" fill="#0F6E56" />
    <circle cx="16" cy="13" r="6.5" fill="#5DCAA5" />
    <circle cx="11.5" cy="11" r="4" fill="#1D9E75" />
    <circle cx="20" cy="11.5" r="4.5" fill="#1D9E75" />
    <rect x="15" y="17" width="2" height="8" rx="0.5" fill="#0A3F30" />
    <circle cx="24" cy="22" r="3" fill="#EF9F27" stroke="#0F6E56" strokeWidth="1.5" />
  </svg>
);

// ============ VIEW MODE SWITCH ============
function ViewModeSwitch({ mode, setMode, agg }) {
  return (
    <div className="view-mode">
      <button
        type="button"
        className={`vm-opt ${mode === "single" ? "active" : ""}`}
        onClick={() => setMode("single")}
        title="Работать с одним снимком: загрузить → инференс → результат"
      >
        <Icon name="image" size={13} />
        <span>Single image</span>
      </button>
      <button
        type="button"
        className={`vm-opt ${mode === "city" ? "active" : ""}`}
        onClick={() => setMode("city")}
        title="Aggregate city map: все деревья из всех сохранённых прогонов"
      >
        <Icon name="map" size={13} />
        <span>City map</span>
        {agg && agg.total_trees > 0 && (
          <span className="vm-badge">{agg.total_trees}</span>
        )}
      </button>
    </div>
  );
}

// ============ SNAPSHOTS LIST (city view) ============
// ============ SCANS LIST (city view) ============
function ScansList({ scans, onDelete, loading }) {
  if (loading || !scans || scans.length === 0) return null;
  return (
    <div className="section">
      <div className="section-label">
        <Icon name="grid" size={13} />
        <span>Auto-Zoom Scans</span>
        <span className="badge-count">{scans.length}</span>
      </div>
      <ul className="snap-list">
        {scans.map((s) => {
          // status: 'running' (если упало посередине без finalize) / 'completed'
          // — даём пользователю визуально различать "висит" от "ok".
          const created = s.created_at ? new Date(s.created_at) : null;
          const dateStr = created ? created.toLocaleString() : "";
          const isRunning = s.status === "running";
          return (
            <li key={s.id} className="snap-item">
              <div className="snap-row">
                <div className="snap-name" title={s.id}>
                  Scan <span style={{ fontFamily: "ui-monospace, Menlo, monospace", opacity: 0.7 }}>{s.id.slice(0, 8)}</span>
                  {isRunning && <span style={{ color: "#f59e0b", marginLeft: 6, fontSize: 10 }}>● running</span>}
                  {s.polygon_json && <span title="Polygon scan" style={{ marginLeft: 6 }}>🔷</span>}
                </div>
                <button
                  className="snap-del"
                  onClick={() => onDelete(s.id, s.sub_count)}
                  title="Удалить scan-сессию + ВСЕ её sub-snapshots + детекции (cascade)"
                >
                  <Icon name="x" size={11} />
                </button>
              </div>
              <div className="snap-meta">
                <span>{s.provider}</span>
                <span>·</span>
                <span>z{s.zoom}</span>
                <span>·</span>
                <span className="snap-model">{s.model}</span>
                <span>·</span>
                <span>{s.ok_count}/{s.sub_count} ok</span>
                <span>·</span>
                <span className="snap-trees">{s.total_trees.toLocaleString()} trees</span>
              </div>
              <div className="snap-coords">
                N {s.nw_lat.toFixed(5)}° → S {s.se_lat.toFixed(5)}°
                {s.duration_ms ? ` · ${(s.duration_ms / 1000).toFixed(0)}s` : ""}
              </div>
              {dateStr && <div className="snap-coords" style={{ opacity: 0.6 }}>{dateStr}</div>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function SnapshotsList({ snapshots, onDelete, loading }) {
  return (
    <div className="section">
      <div className="section-label">
        <Icon name="layers" size={13} />
        <span>Snapshots in DB</span>
        <span className="badge-count">{snapshots.length}</span>
      </div>
      {loading && <div className="snap-empty">Loading…</div>}
      {!loading && snapshots.length === 0 && (
        <div className="snap-empty">No snapshots yet. Switch to <b>Single image</b>, capture a region and run detection — it’ll appear here.</div>
      )}
      <ul className="snap-list">
        {snapshots.map((s) => (
          <li key={s.image_id} className="snap-item">
            <div className="snap-row">
              <div className="snap-name" title={s.filename}>{s.filename}</div>
              <button
                className="snap-del"
                onClick={() => onDelete(s.image_id)}
                title="Удалить снимок + все его прогоны"
              >
                <Icon name="x" size={11} />
              </button>
            </div>
            <div className="snap-meta">
              <span>{s.width}×{s.height}</span>
              <span>·</span>
              <span>{s.run_count} run{s.run_count === 1 ? "" : "s"}</span>
              <span>·</span>
              <span className="snap-trees">{s.total_trees} trees</span>
              {s.last_model && <><span>·</span><span className="snap-model">{s.last_model}</span></>}
            </div>
            {s.nw_lat != null && (
              <div className="snap-coords">
                N {s.nw_lat.toFixed(5)}° → S {s.se_lat.toFixed(5)}°
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ============ AGGREGATE STATS PANEL (city view) ============
function AggregateStatsPanel({ stats }) {
  if (!stats || !stats.snapshot_count) return null;
  return (
    <div className="section">
      <div className="section-label">
        <span>City Aggregate</span>
        <span className="badge-live">● LIVE</span>
      </div>
      <div className="agg-grid">
        <div className="agg-card"><div className="agg-v">{stats.total_trees.toLocaleString()}</div><div className="agg-k">trees</div></div>
        <div className="agg-card"><div className="agg-v">{stats.snapshot_count}</div><div className="agg-k">snapshots</div></div>
        <div className="agg-card"><div className="agg-v">{stats.run_count}</div><div className="agg-k">runs</div></div>
        {stats.avg_confidence != null && (
          <div className="agg-card"><div className="agg-v">{Math.round(stats.avg_confidence * 100)}%</div><div className="agg-k">avg conf</div></div>
        )}
        {stats.avg_crown_m != null && (
          <div className="agg-card"><div className="agg-v">{stats.avg_crown_m.toFixed(1)}<span className="agg-suf"> m</span></div><div className="agg-k">avg crown</div></div>
        )}
      </div>
    </div>
  );
}

// ============ HEADER ============
function Header({ dark, onToggleDark, modelStatus }) {
  return (
    <div className="sb-header">
      <div className="sb-logo">
        <LogoMark size={36} />
        <div className="sb-logo-text">
          <div className="sb-title">Astana Tree Detection</div>
          <div className="sb-tagline">Automated urban forest inventory</div>
        </div>
        <button className="theme-toggle" onClick={onToggleDark} title={dark ? "Light" : "Dark"} aria-label="Toggle dark mode">
          <span className={`theme-toggle-track ${dark ? "is-dark" : ""}`}>
            <span className="theme-toggle-thumb">
              {dark ? (
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
              ) : (
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              )}
            </span>
          </span>
        </button>
      </div>
      <div className="sb-univ">
        <div className="sb-univ-mark">AITU</div>
        <div className="sb-univ-text">
          <div>Astana IT University</div>
          <div className="sb-univ-sub">Diploma Project · 2026</div>
        </div>
      </div>
    </div>
  );
}

// ============ UPLOAD ZONE ============
function UploadZone({ image, onUpload, onClear, scanning, uploading, error, captureMode, onStartCapture, onCancelCapture, captureZoom, setCaptureZoom, scanMode, onStartScan, onCancelScan, scanRunning, scanStatus, model, tileProvider, setTileProvider, providersMap, scanProgress, polygonMode, onStartPolygon, onCancelPolygon, pendingPolygon, onStartPolygonScan, onClearPolygon }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    onUpload(file);
  };

  return (
    <div className="section">
      <div className="section-label">
        <Icon name="upload" size={13} />
        <span>Satellite Image</span>
      </div>
      {!image ? (
        <div
          className={`upload-zone ${drag ? "dragging" : ""} ${uploading ? "is-scanning" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,.webp"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className="upload-icon"><Icon name={uploading ? "globe" : "image"} size={22} /></div>
          <div className="upload-primary">{uploading ? "Uploading…" : "Drop image or click to browse"}</div>
          <div className="upload-secondary">PNG · JPG · TIFF · GeoTIFF · Max 100MB</div>
          <div className="upload-hint">Source: Google Earth / SAS.Planet · Zoom 17–20</div>
        </div>
      ) : null}
      {!image && !captureMode && !scanMode && !scanRunning && (
        <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6, fontSize: 11, color: "#666" }}>
          <span>Imagery:</span>
          <select
            value={tileProvider}
            onChange={(e) => setTileProvider(e.target.value)}
            className="select"
            style={{ flex: 1, height: 28, fontSize: 11 }}
            title="Источник тайлов для capture + Leaflet base layer. Google = ближе к training distribution YOLO/Mask R-CNN."
          >
            {providersMap
              ? Object.entries(providersMap).map(([key, cfg]) => (
                  <option key={key} value={key}>{cfg.label}</option>
                ))
              : (<>
                  <option value="google">Google Satellite</option>
                  <option value="esri">Esri World Imagery</option>
                </>)
            }
          </select>
        </div>
      )}
      {!image && !captureMode && !scanMode && !scanRunning && (
        <div className="capture-row">
          <button
            type="button"
            className="btn-capture"
            onClick={onStartCapture}
            disabled={uploading}
            title={`Нарисовать прямоугольник на карте, скачать тайлы ${tileProvider}`}
          >
            <Icon name="target" size={14} />
            <span>Capture from map</span>
          </button>
          <div className="capture-zoom">
            <label>Zoom</label>
            <input
              type="number" min="14" max="20" step="1"
              value={captureZoom}
              onChange={(e) => setCaptureZoom(Math.max(14, Math.min(20, +e.target.value || 18)))}
            />
          </div>
        </div>
      )}
      {!image && !captureMode && !scanMode && !polygonMode && !scanRunning && (
        <div className="capture-row" style={{ marginTop: 6 }}>
          <button
            type="button"
            className="btn-capture"
            onClick={onStartScan}
            disabled={uploading}
            style={{ background: "linear-gradient(135deg,#0F6E56,#1a9170)" }}
            title="Большой bbox → авто-сетка под-регионов на zoom 19 → predict каждого"
          >
            <Icon name="grid" size={14} />
            <span>Auto-Zoom Scan</span>
          </button>
          <div className="capture-zoom">
            <label>z</label>
            <input type="number" value={19} disabled readOnly title="Фиксированный max-zoom для максимальной детализации" />
          </div>
        </div>
      )}
      {!image && !captureMode && !scanMode && !polygonMode && !scanRunning && (
        <div className="capture-row" style={{ marginTop: 6 }}>
          <button
            type="button"
            className="btn-capture"
            onClick={onStartPolygon}
            disabled={uploading}
            style={{ background: "linear-gradient(135deg,#5DCAA5,#0F6E56)", flex: 1 }}
            title="Кликай по карте чтобы рисовать произвольный полигон. Двойной клик = замкнуть и запустить scan."
          >
            <Icon name="leaf" size={14} />
            <span>Polygon Scan</span>
          </button>
        </div>
      )}
      {!image && captureMode && (
        <div className="capture-active">
          <div className="capture-active-row">
            <span className="capture-pulse"></span>
            <span>Нарисуй прямоугольник на карте справа</span>
          </div>
          <button type="button" className="btn-capture-cancel" onClick={onCancelCapture}>
            Отмена
          </button>
        </div>
      )}
      {!image && scanMode && !scanRunning && (
        <div className="capture-active" style={{ borderColor: "#0F6E56" }}>
          <div className="capture-active-row">
            <span className="capture-pulse" style={{ background: "#0F6E56" }}></span>
            <span>Auto-Zoom Scan: нарисуй большой прямоугольник</span>
          </div>
          <div style={{ fontSize: 11, color: "#666", margin: "4px 0 8px" }}>
            Сервер сам дробит на сетку под-регионов на zoom 19 ({model || "yolo"}).
            Лимит: 9 под-регионов за запрос.
          </div>
          <button type="button" className="btn-capture-cancel" onClick={onCancelScan}>
            Отмена
          </button>
        </div>
      )}
      {!image && polygonMode && !scanRunning && !pendingPolygon && (
        <div className="capture-active" style={{ borderColor: "#0F6E56" }}>
          <div className="capture-active-row">
            <span className="capture-pulse" style={{ background: "#0F6E56" }}></span>
            <span>Polygon Scan: клик = вершина, double-click = замкнуть</span>
          </div>
          <div style={{ fontSize: 11, color: "#666", margin: "4px 0 8px" }}>
            <b>Right-click</b> в любой момент — очистить и начать заново.
            Bbox-сетка строится по axis-aligned обёртке, детекции фильтруются
            point-in-polygon ({model || "yolo"} · z19 · ≤9 sub-regions).
          </div>
          <button type="button" className="btn-capture-cancel" onClick={onCancelPolygon}>
            Отмена
          </button>
        </div>
      )}
      {!image && polygonMode && !scanRunning && pendingPolygon && pendingPolygon.length >= 3 && (
        <div className="capture-active" style={{ borderColor: "#0F6E56" }}>
          <div className="capture-active-row">
            <span style={{
              display: "inline-block", width: 10, height: 10,
              background: "#0F6E56", borderRadius: 2, marginRight: 4,
            }}></span>
            <span>Polygon ready · {pendingPolygon.length} vertices</span>
          </div>
          <div style={{ fontSize: 11, color: "#666", margin: "4px 0 8px" }}>
            Готов к запуску. Right-click на карте или Clear ниже = перерисовать.
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              onClick={onStartPolygonScan}
              className="btn-capture"
              style={{ flex: 1, background: "linear-gradient(135deg,#0F6E56,#1a9170)" }}
              title="Запустить scan по нарисованному полигону"
            >
              <Icon name="play" size={12} />
              <span>Start Polygon Scan</span>
            </button>
            <button type="button" className="btn-capture-cancel" onClick={onClearPolygon}>
              Clear
            </button>
          </div>
        </div>
      )}
      {scanRunning && (
        <div className="capture-active" style={{ borderColor: "#0F6E56" }}>
          <div className="capture-active-row">
            <span className="capture-pulse" style={{ background: "#0F6E56" }}></span>
            <span>{scanStatus || "Сканирую под-регионы…"}</span>
          </div>
          {scanProgress && scanProgress.regions && scanProgress.regions.length > 0 && (() => {
            const total = scanProgress.regions.length;
            const done = scanProgress.regions.filter((r) => r.status === "done").length;
            const errs = scanProgress.regions.filter((r) => r.status === "error").length;
            const totalTrees = scanProgress.trees ? scanProgress.trees.length : 0;
            // Авто-определяем число столбцов для grid: квадрат от N regions.
            const cols = Math.ceil(Math.sqrt(total));
            const STATUS_BG = {
              pending: "#dadce0",
              capturing: "#3b82f6",
              captured: "#1d4ed8",
              predicting: "#f59e0b",
              done: "#0F6E56",
              error: "#dc3545",
            };
            return (
              <>
                <div style={{ display: "flex", gap: 8, fontSize: 11, marginTop: 6, color: "#444" }}>
                  <span><b>{done}/{total}</b> done</span>
                  {errs > 0 && <span style={{ color: "#dc3545" }}>· {errs} err</span>}
                  <span style={{ marginLeft: "auto" }}>🌳 {totalTrees}</span>
                </div>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: `repeat(${cols}, 1fr)`,
                  gap: 3,
                  marginTop: 6,
                }}>
                  {scanProgress.regions.map((r) => (
                    <div
                      key={`${r.row}-${r.col}`}
                      title={`r${r.row}c${r.col}: ${r.status}${r.tree_count ? ` · ${r.tree_count} trees` : ""}${r.error ? ` · ${r.error}` : ""}`}
                      className={r.status === "capturing" || r.status === "predicting" ? "scan-region-pulse" : ""}
                      style={{
                        aspectRatio: "1 / 1",
                        background: STATUS_BG[r.status] || "#ccc",
                        borderRadius: 3,
                        opacity: r.status === "pending" ? 0.5 : 1,
                      }}
                    />
                  ))}
                </div>
              </>
            );
          })()}
          <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
            Каждый под-регион качает ~100 тайлов и прогоняет модель — не закрывай вкладку.
          </div>
        </div>
      )}
      {image && (
        <div className="upload-preview">
          <div className={`upload-preview-img ${scanning ? "is-scanning" : ""}`}>
            {image.url ? (<img src={image.url} alt={image.name} />) : (
              <div className="upload-mock-img">
                <div className="mock-grid"></div>
                <div className="mock-label">SATELLITE PATCH</div>
              </div>
            )}
            {scanning && (
              <>
                <div className="scan-line"></div>
                <div className="scan-grid-overlay"></div>
                <div className="scan-corners">
                  <span className="sc tl"></span><span className="sc tr"></span>
                  <span className="sc bl"></span><span className="sc br"></span>
                </div>
              </>
            )}
            <button className="upload-clear" onClick={onClear} title="Remove">
              <Icon name="x" size={12} />
            </button>
          </div>
          <div className="upload-meta">
            <div className="upload-name">{image.name}</div>
            <div className="upload-stats">
              <span>{image.width} × {image.height} px</span>
              {image.is_geotiff && <><span>·</span><span>GeoTIFF</span></>}
              {image.crs && <><span>·</span><span>{image.crs}</span></>}
            </div>
            {image.bounds && (
              <div className="upload-bounds">
                <div className="ub-row"><span className="ub-k">N</span><span className="ub-v">{image.bounds.nw.lat.toFixed(4)}°</span><span className="ub-k">S</span><span className="ub-v">{image.bounds.se.lat.toFixed(4)}°</span></div>
                <div className="ub-row"><span className="ub-k">W</span><span className="ub-v">{image.bounds.nw.lng.toFixed(4)}°</span><span className="ub-k">E</span><span className="ub-v">{image.bounds.se.lng.toFixed(4)}°</span></div>
              </div>
            )}
          </div>
        </div>
      )}
      {error && (
        <div style={{ marginTop: 8, padding: "8px 12px", background: "rgba(239,159,39,0.15)", border: "1px solid #EF9F27", borderRadius: 6, fontSize: 12, color: "#a8651a" }}>
          <Icon name="alert" size={12} /> {error}
        </div>
      )}
    </div>
  );
}

// ============ DETECTION CONTROLS ============
function DetectionControls({ canRun, status, progress, eta, onRun, model, setModel, modelStatus, error }) {
  const isModelAvailable = (kind) => modelStatus?.models?.[kind]?.available;
  return (
    <div className="section">
      <div className="section-label">
        <Icon name="zap" size={13} />
        <span>Detection</span>
      </div>
      <div className="model-row">
        <label className="model-label">Model</label>
        <div className="select-wrap">
          <select value={model} onChange={(e) => setModel(e.target.value)} className="select">
            <option value="yolo" disabled={!isModelAvailable("yolo")}>
              YOLOv8-seg{isModelAvailable("yolo") ? "" : " (weights missing)"}
            </option>
            <option value="deepforest" disabled={!isModelAvailable("deepforest")}>
              DeepForest (Astana fine-tuned)
            </option>
            <option value="ensemble" disabled={!isModelAvailable("ensemble")}>
              Ensemble · YOLO + DeepForest{isModelAvailable("ensemble") ? "" : " (need both)"}
            </option>
            <option value="deepforest_sam2" disabled={!isModelAvailable("deepforest_sam2")}>
              DeepForest + SAM2 (crown masks)
            </option>
          </select>
          <div className="select-chev"><Icon name="chevron" size={14} /></div>
        </div>
      </div>

      <button
        className={`btn-primary ${!canRun || status === "running" ? "disabled" : ""}`}
        onClick={onRun}
        disabled={!canRun || status === "running"}
      >
        {status === "running" ? (
          <>
            <span className="spinner"></span>
            <span>Detecting trees…</span>
          </>
        ) : status === "done" ? (
          <>
            <Icon name="check" size={15} />
            <span>Re-run detection</span>
          </>
        ) : (
          <>
            <Icon name="play" size={13} />
            <span>Run detection</span>
          </>
        )}
      </button>

      {status === "running" && (
        <div className="progress">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            <div className="progress-shimmer"></div>
          </div>
          <div className="progress-meta">
            <span className="progress-stage">
              <span className="progress-pulse"></span>
              {progress < 30 ? "Loading model weights…" : progress < 65 ? "Sliding window inference…" : progress < 90 ? "NMS post-processing…" : "Geo-referencing…"}
            </span>
            <span className="progress-pct">{progress}%</span>
          </div>
          <div className="progress-eta">
            <span>ETA</span>
            <span className="progress-eta-val">{eta != null ? (eta < 1 ? "< 1s" : `${Math.ceil(eta)}s remaining`) : "—"}</span>
          </div>
        </div>
      )}

      {error && (
        <div style={{ marginTop: 8, padding: "8px 12px", background: "rgba(220,53,69,0.12)", border: "1px solid #dc3545", borderRadius: 6, fontSize: 12, color: "#a02334" }}>
          <Icon name="alert" size={12} /> {error}
        </div>
      )}
    </div>
  );
}

// ============ GEO PANEL ============
function GeoPanel({ geo, setGeo, image }) {
  const setMode = (mode) => setGeo({ ...geo, mode });
  const setCorner = (which, key, value) => {
    const parsed = parseFloat(value);
    // Keep the previous value if the user typed something non-numeric — the
    // old `parseFloat(value) || 0` silently coerced invalid input to 0 and
    // teleported the corner marker to (0, 0).
    if (!Number.isFinite(parsed)) return;
    const corners = { ...(geo.corners_2 || { nw: { lat: 51.17, lng: 71.46 }, se: { lat: 51.15, lng: 71.49 } }) };
    corners[which] = { ...corners[which], [key]: parsed };
    setGeo({ ...geo, corners_2: corners });
  };

  const isGeotiff = image?.is_geotiff;

  return (
    <div className="section">
      <div className="section-label">
        <Icon name="globe" size={13} />
        <span>Geo-referencing</span>
      </div>
      <div className="model-row">
        <label className="model-label">Source</label>
        <div className="select-wrap">
          <select value={geo.mode} onChange={(e) => setMode(e.target.value)} className="select">
            <option value="none">None (pixel coords only)</option>
            <option value="geotiff" disabled={!isGeotiff}>
              {isGeotiff ? "GeoTIFF (auto from file)" : "GeoTIFF (upload .tif first)"}
            </option>
            <option value="corners_2">2 corners (axis-aligned)</option>
            <option value="corners_4">4 corners (handles rotation)</option>
          </select>
          <div className="select-chev"><Icon name="chevron" size={14} /></div>
        </div>
      </div>

      {geo.mode === "corners_2" && (
        <div className="corners-grid">
          <div className="corners-hint">Перетащи маркеры NW/SE на карте, либо вбей вручную:</div>
          <div className="corner-row">
            <label>NW lat</label>
            <input type="number" step="0.0001" value={geo.corners_2?.nw?.lat ?? 51.17}
              onChange={(e) => setCorner("nw", "lat", e.target.value)} />
            <label>NW lng</label>
            <input type="number" step="0.0001" value={geo.corners_2?.nw?.lng ?? 71.46}
              onChange={(e) => setCorner("nw", "lng", e.target.value)} />
          </div>
          <div className="corner-row">
            <label>SE lat</label>
            <input type="number" step="0.0001" value={geo.corners_2?.se?.lat ?? 51.15}
              onChange={(e) => setCorner("se", "lat", e.target.value)} />
            <label>SE lng</label>
            <input type="number" step="0.0001" value={geo.corners_2?.se?.lng ?? 71.49}
              onChange={(e) => setCorner("se", "lng", e.target.value)} />
          </div>
        </div>
      )}
    </div>
  );
}

// ============ STATS HELPERS ============
function useCountUp(target, duration = 900, decimals = 0) {
  const [val, setVal] = useState(0);
  const startRef = useRef(null);
  const fromRef = useRef(0);
  useEffect(() => {
    fromRef.current = val;
    startRef.current = null;
    let raf;
    const tick = (t) => {
      if (!startRef.current) startRef.current = t;
      const p = Math.min(1, (t - startRef.current) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const next = fromRef.current + (target - fromRef.current) * eased;
      setVal(next);
      if (p < 1) raf = requestAnimationFrame(tick); else setVal(target);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line
  }, [target]);
  return decimals > 0 ? val.toFixed(decimals) : Math.round(val);
}

function Sparkline({ data, color = "#1D9E75", height = 22, width = 64 }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const points = data.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return [x, y];
  });
  const d = points.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(" ");
  const dArea = `${d} L${width},${height} L0,${height} Z`;
  const last = points[points.length - 1];
  const id = `spk-${color.replace("#", "")}`;
  return (
    <svg width={width} height={height} className="sparkline" viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={dArea} fill={`url(#${id})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r="2" fill={color} stroke="var(--surface)" strokeWidth="1" />
    </svg>
  );
}

function TrendArrow({ delta, suffix = "%" }) {
  if (delta == null) return null;
  const dir = delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const sign = delta > 0 ? "+" : "";
  return (
    <span className={`trend trend-${dir}`}>
      <svg width="9" height="9" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        {dir === "up" && <><path d="M3 8l3-3 3 3" /><path d="M6 5v3" /></>}
        {dir === "down" && <><path d="M3 4l3 3 3-3" /><path d="M6 7V4" /></>}
        {dir === "flat" && <path d="M3 6h6" />}
      </svg>
      <span>{sign}{Math.abs(delta).toFixed(1)}{suffix}</span>
    </span>
  );
}

const SPARK_COLORS = {
  "green-dark": "#0F6E56",
  "green": "#1D9E75",
  "green-light": "#5DCAA5",
  "neutral": "#7C8682",
};

function StatCard({ icon, label, value, suffix, accent, decimals, history, prev }) {
  const animated = useCountUp(value || 0, 900, decimals || 0);
  const delta = prev != null && prev !== 0 ? ((value - prev) / prev) * 100 : null;
  return (
    <div className="stat-card">
      <div className="stat-top">
        <div className={`stat-icon ${accent || ""}`}><Icon name={icon} size={13} /></div>
        <div className="stat-label">{label}</div>
      </div>
      <div className="stat-mid">
        <div className="stat-value">
          {animated}
          {suffix && <span className="stat-suffix">{suffix}</span>}
        </div>
        <Sparkline data={history} color={SPARK_COLORS[accent] || "#1D9E75"} />
      </div>
      <div className="stat-bot">
        <TrendArrow delta={delta} suffix={suffix === "%" ? "pp" : "%"} />
        <span className="stat-vs">vs. previous scan</span>
      </div>
    </div>
  );
}

const STAT_SERIES = {
  trees:    { history: [38, 42, 31, 47, 39, 44, 47],       prev: 39 },
  coverage: { history: [9.1, 10.4, 8.7, 12.0, 11.3, 11.9], prev: 11.3 },
  conf:     { history: [68, 71, 69, 73, 72, 74, 73],       prev: 72 },
  area:     { history: [8.4, 10.2, 9.6, 11.8, 12.1, 12.4], prev: 12.1 },
};

function StatsPanel({ trees, stats }) {
  const total = stats?.tree_count ?? (trees ? trees.length : 0);
  if (!total) return null;
  const avgConf = total && trees && trees.length
    ? Math.round((trees.reduce((s, t) => s + t.confidence, 0) / trees.length) * 100)
    : 0;
  const coverage = stats?.coverage_pct != null ? stats.coverage_pct : null;
  const area = stats?.analyzed_area_ha;

  return (
    <div className="section">
      <div className="section-label">
        <span>Results</span>
        <span className="badge-live">● LIVE</span>
      </div>
      <div className="stats-grid">
        <StatCard icon="tree"  label="Trees detected"   value={total}    accent="green-dark"  history={STAT_SERIES.trees.history}    prev={STAT_SERIES.trees.prev} />
        {coverage != null && (
          <StatCard icon="leaf"  label="Green coverage"   value={coverage} suffix="%" decimals={1} accent="green" history={STAT_SERIES.coverage.history} prev={STAT_SERIES.coverage.prev} />
        )}
        <StatCard icon="check" label="Avg. confidence"  value={avgConf}  suffix="%" accent="green-light" history={STAT_SERIES.conf.history}     prev={STAT_SERIES.conf.prev} />
        {area != null && (
          <StatCard icon="area"  label="Area analyzed"    value={area}     suffix=" ha" decimals={2} accent="neutral" history={STAT_SERIES.area.history} prev={STAT_SERIES.area.prev} />
        )}
      </div>
    </div>
  );
}

// ============ MAP LAYERS PANEL ============
function MapLayersPanel({ baseLayer, setBaseLayer, showOverlay, setShowOverlay, displayMode, setDisplayMode, overlayOpacity, setOverlayOpacity, hasImage, hasTrees, hasMasks }) {
  return (
    <div className="section">
      <div className="section-label">
        <Icon name="layers" size={13} />
        <span>Map Layers</span>
      </div>
      <div className="layer-stack">
        <div className="layer-group">
          <div className="layer-group-label">Base</div>
          <label className={`layer-toggle ${baseLayer === "satellite" ? "active" : ""}`}>
            <input type="radio" name="base" checked={baseLayer === "satellite"} onChange={() => setBaseLayer("satellite")} />
            <span className="layer-icon sat"><Icon name="map" size={12} /></span>
            <span className="layer-name">Satellite</span>
            <span className="layer-meta">Esri</span>
          </label>
          <label className={`layer-toggle ${baseLayer === "clean" ? "active" : ""}`}>
            <input type="radio" name="base" checked={baseLayer === "clean"} onChange={() => setBaseLayer("clean")} />
            <span className="layer-icon clean"><Icon name="file" size={12} /></span>
            <span className="layer-name">Clean Map</span>
            <span className="layer-meta">CartoDB</span>
          </label>
        </div>
        <div className="layer-group">
          <div className="layer-group-label">Overlays</div>
          <label className={`layer-toggle ${showOverlay ? "active" : ""} ${!hasImage ? "disabled" : ""}`}>
            <input type="checkbox" checked={showOverlay} disabled={!hasImage} onChange={(e) => setShowOverlay(e.target.checked)} />
            <span className="layer-icon overlay"><Icon name="image" size={12} /></span>
            <span className="layer-name">Image Overlay</span>
            <span className={`layer-switch ${showOverlay ? "on" : ""}`}><span className="layer-switch-thumb"></span></span>
          </label>
          {showOverlay && (
            <div className="layer-opacity">
              <span className="layer-opacity-label">Opacity</span>
              <input type="range" min="0" max="100" step="1"
                value={Math.round(overlayOpacity * 100)}
                onChange={(e) => setOverlayOpacity(+e.target.value / 100)} />
              <span className="layer-opacity-val">{Math.round(overlayOpacity * 100)}%</span>
            </div>
          )}
        </div>
        <div className="layer-group">
          <div className="layer-group-label">Detection Display</div>
          <div className={`display-mode-switch ${!hasTrees ? "disabled" : ""}`}>
            <button
              type="button"
              className={`dms-opt ${displayMode === "point" ? "active" : ""}`}
              onClick={() => hasTrees && setDisplayMode("point")}
              disabled={!hasTrees}
              title="Single point at detection center"
            >
              <Icon name="circle" size={12} />
              <span>Point</span>
            </button>
            <button
              type="button"
              className={`dms-opt ${displayMode === "bbox" ? "active" : ""}`}
              onClick={() => hasTrees && setDisplayMode("bbox")}
              disabled={!hasTrees}
              title="Bounding box around detection"
            >
              <Icon name="square" size={12} />
              <span>BBox</span>
            </button>
            <button
              type="button"
              className={`dms-opt ${displayMode === "polygon" ? "active" : ""} ${!hasMasks ? "no-data" : ""}`}
              onClick={() => hasTrees && setDisplayMode("polygon")}
              disabled={!hasTrees}
              title={hasMasks ? "Crown segmentation polygon (YOLO only)" : "No segmentation masks in current result"}
            >
              <Icon name="leaf" size={12} />
              <span>Polygon</span>
            </button>
            <button
              type="button"
              className={`dms-opt ${displayMode === "heat" ? "active" : ""}`}
              onClick={() => hasTrees && setDisplayMode("heat")}
              disabled={!hasTrees}
              title="KDE-style density heatmap (хорошо смотрится в city view на 1000+ деревьях)"
            >
              <Icon name="globe" size={12} />
              <span>Heat</span>
            </button>
          </div>
          {displayMode === "polygon" && hasTrees && !hasMasks && (
            <div className="dms-hint">No masks in current detection — falling back to points. Run YOLO or Ensemble to get polygons.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============ CONFIDENCE FILTER ============
function ConfidenceFilter({ filter, setFilter, trees }) {
  if (!trees) return null;
  const counts = useMemo(() => ({
    high: trees.filter((t) => t.confidence > 0.7).length,
    med: trees.filter((t) => t.confidence > 0.5 && t.confidence <= 0.7).length,
    low: trees.filter((t) => t.confidence <= 0.5).length,
  }), [trees]);

  const tiers = [
    { key: "high", label: "High", range: "> 70%", color: "#0F6E56", count: counts.high },
    { key: "med", label: "Medium", range: "50–70%", color: "#5DCAA5", count: counts.med },
    { key: "low", label: "Low", range: "< 50%", color: "#EF9F27", count: counts.low },
  ];

  return (
    <div className="section">
      <div className="section-label"><span>Confidence Filter</span></div>
      <div className="filter-tiers">
        {tiers.map((t) => (
          <label key={t.key} className={`filter-row ${filter[t.key] ? "active" : ""}`}>
            <input type="checkbox" checked={filter[t.key]} onChange={() => setFilter({ ...filter, [t.key]: !filter[t.key] })} />
            <span className="filter-dot" style={{ background: t.color }}></span>
            <span className="filter-label">{t.label}</span>
            <span className="filter-range">{t.range}</span>
            <span className="filter-count">{t.count}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

// ============ EXPORT PANEL ============
function ExportPanel({ enabled, onExport }) {
  return (
    <div className="section">
      <div className="section-label">
        <Icon name="download" size={13} />
        <span>Export</span>
      </div>
      <div className="export-grid">
        <button className="btn-export" disabled={!enabled} onClick={() => onExport("geojson")}>
          <div className="export-icon"><Icon name="map" size={14} /></div>
          <div className="export-body">
            <div className="export-name">GeoJSON</div>
            <div className="export-sub">QGIS · ArcGIS</div>
          </div>
        </button>
        <button className="btn-export" disabled={!enabled} onClick={() => onExport("csv")}>
          <div className="export-icon"><Icon name="file" size={14} /></div>
          <div className="export-body">
            <div className="export-name">CSV</div>
            <div className="export-sub">Coordinates</div>
          </div>
        </button>
        <button className="btn-export wide" disabled={!enabled} onClick={() => onExport("html")}>
          <div className="export-icon"><Icon name="download" size={14} /></div>
          <div className="export-body">
            <div className="export-name">Standalone HTML map</div>
            <div className="export-sub">Self-contained · shareable</div>
          </div>
        </button>
      </div>
    </div>
  );
}

// ============ HISTORY PANEL ============
function HistoryPanel({ open, setOpen, history, onLoad }) {
  return (
    <div className="section">
      <button className="section-toggle" onClick={() => setOpen(!open)}>
        <Icon name="history" size={13} />
        <span>History</span>
        <span className="history-count">{history.length}</span>
        <span className="toggle-chev"><Icon name={open ? "chevronUp" : "chevron"} size={14} /></span>
      </button>
      {open && history.length > 0 && (
        <div className="history-list">
          {history.map((h) => (
            <button key={h.image_id || h.id} className="history-item" onClick={() => onLoad(h)}>
              <div className="history-thumb"><div className="history-thumb-grid"></div></div>
              <div className="history-info">
                <div className="history-name">{h.filename || h.name}</div>
                <div className="history-meta">
                  <span>{h.date ? new Date(h.date).toLocaleDateString() : "—"}</span>
                  <span>·</span>
                  <span>{h.tree_count ?? h.trees ?? 0} trees</span>
                  <span>·</span>
                  <span>{h.model || "demo"}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
      {open && history.length === 0 && (
        <div style={{ padding: "12px", color: "var(--text-muted)", fontSize: 13, textAlign: "center" }}>
          No predictions yet. Upload an image and run detection.
        </div>
      )}
    </div>
  );
}

// ============ MAP COMPONENT ============
function MapView({ trees, filter, threshold, baseLayer, setBaseLayer, onTreeClick, markerSize, scanning, showOverlay, displayMode, overlayOpacity, image, imageBounds, geo, setGeo, captureMode, onCaptureBbox, scanMode, onScanBbox, tileProvider, providersMap, scanProgress, polygonMode, onPolygonComplete, pendingPolygon, onPolygonReset }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const layerRef = useRef(null);
  const tileLayerRef = useRef(null);
  const overlayRef = useRef(null);
  const cornersLayerRef = useRef(null);
  const nwMarkerRef = useRef(null);
  const seMarkerRef = useRef(null);
  const rectRef = useRef(null);
  const captureRectRef = useRef(null);
  // Auto-Zoom Scan progress overlay: grid под-регионов + прогрессивные деревья.
  const scanGridRef = useRef(null);
  const scanTreesRef = useRef(null);
  // KDE-плотность для displayMode="heat" — не layerGroup а сам L.heatLayer.
  const heatLayerRef = useRef(null);

  useEffect(() => {
    if (mapInstance.current || !mapRef.current) return;
    const map = L.map(mapRef.current, { zoomControl: false, attributionControl: false }).setView(ASTANA_CENTER, 14);
    mapInstance.current = map;

    tileLayerRef.current = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "" }
    ).addTo(map);

    L.control.scale({ position: "bottomright", imperial: false }).addTo(map);
    layerRef.current = L.layerGroup().addTo(map);
    cornersLayerRef.current = L.layerGroup().addTo(map);
    scanGridRef.current = L.layerGroup().addTo(map);
    scanTreesRef.current = L.layerGroup().addTo(map);

    return () => { map.remove(); mapInstance.current = null; };
  }, []);

  useEffect(() => {
    if (!mapInstance.current) return;
    // Satellite URL берём из providersMap (если фронт уже получил список)
    // или из локального fallback — гарантирует Leaflet base layer совпадает
    // с тем, что backend качает для capture/scan.
    const providerCfg = (providersMap && providersMap[tileProvider]) || null;
    const satUrl = providerCfg
      ? providerCfg.url.replace(/\{s\}/g, "{s}")  // Leaflet handles {s} via subdomains option
      : "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    const satOpts = providerCfg
      ? { maxZoom: providerCfg.max_zoom || 19, subdomains: providerCfg.subdomains || "abc", attribution: "" }
      : { maxZoom: 19, attribution: "" };

    const layers = {
      satellite: { url: satUrl, opts: satOpts },
      streets:   { url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", opts: { maxZoom: 19, attribution: "" } },
      clean:     { url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", opts: { maxZoom: 19, attribution: "" } },
    };
    const pick = layers[baseLayer] || layers.satellite;
    if (tileLayerRef.current) mapInstance.current.removeLayer(tileLayerRef.current);
    tileLayerRef.current = L.tileLayer(pick.url, pick.opts).addTo(mapInstance.current);
  }, [baseLayer, tileProvider, providersMap]);

  useEffect(() => {
    if (!layerRef.current || !mapInstance.current) return;
    layerRef.current.clearLayers();
    // Старый heat-layer, если был — снимаем (новый создаётся ниже, если режим heat).
    if (heatLayerRef.current) {
      mapInstance.current.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }
    if (!trees) return;
    const visible = trees.filter((t) => {
      if (t.confidence < threshold) return false;
      if (t.confidence > 0.7) return filter.high;
      if (t.confidence > 0.5) return filter.med;
      return filter.low;
    });

    // Heat-mode: вместо маркеров — KDE-плотность через leaflet.heat.
    // Confidence идёт как intensity weight, чтобы шумные low-conf детекции
    // не пересиливали уверенные high-conf в плотных кластерах.
    if (displayMode === "heat" && typeof L.heatLayer === "function") {
      const points = visible
        .filter((t) => t.lat != null && t.lng != null)
        .map((t) => [t.lat, t.lng, Math.max(0.1, t.confidence || 0.5)]);
      if (points.length) {
        heatLayerRef.current = L.heatLayer(points, {
          radius: 18, blur: 22, maxZoom: 19, minOpacity: 0.35,
          gradient: { 0.2: "#EF9F27", 0.5: "#5DCAA5", 0.8: "#0F6E56", 1.0: "#0A3F30" },
        }).addTo(mapInstance.current);
        try {
          mapInstance.current.fitBounds(points.map(([la, ln]) => [la, ln]), { padding: [40, 40], maxZoom: 18 });
        } catch {}
      }
      return;
    }

    const popupHtml = (t, color) => (
      `<div class="tree-popup">
         <div class="tp-head">
           <span class="tp-id">Tree #${String(t.id).padStart(3, "0")}</span>
           <span class="tp-conf" style="background:${color}">${Math.round(t.confidence * 100)}%</span>
         </div>
         <div class="tp-grid">
           <div class="tp-k">Lat</div><div class="tp-v">${t.lat.toFixed(6)}°</div>
           <div class="tp-k">Lng</div><div class="tp-v">${t.lng.toFixed(6)}°</div>
           <div class="tp-k">Crown</div><div class="tp-v">${t.crown ? t.crown.toFixed(1) + " m" : "—"}</div>
         </div>
       </div>`
    );

    const bounds = [];
    visible.forEach((t) => {
      const color = t.confidence > 0.7 ? "#0F6E56" : t.confidence > 0.5 ? "#5DCAA5" : "#EF9F27";
      let primary = null;

      if (displayMode === "polygon" && t.mask_polygon_geo && t.mask_polygon_geo.length >= 3) {
        primary = L.polygon(t.mask_polygon_geo, {
          color, weight: 1.2, opacity: 0.85, fillColor: color, fillOpacity: 0.28,
        });
      } else if (displayMode === "bbox" && t.box_geo && t.box_geo.length === 4) {
        primary = L.polygon(t.box_geo, {
          color, weight: 1.4, opacity: 0.9, fillColor: color, fillOpacity: 0.12,
        });
      } else if (displayMode === "point") {
        primary = L.circleMarker([t.lat, t.lng], {
          radius: markerSize,
          fillColor: color, color: "#ffffff", weight: 1.5, opacity: 1, fillOpacity: 0.92,
        });
      }

      // Fallback: если выбранный режим не имеет данных (например polygon для DF) —
      // показываем точку, чтобы детекция не пропала с карты.
      if (!primary) {
        primary = L.circleMarker([t.lat, t.lng], {
          radius: Math.max(3, markerSize - 2),
          fillColor: color, color: "#ffffff", weight: 1, opacity: 0.9, fillOpacity: 0.8,
        });
      }

      primary.bindPopup(popupHtml(t, color), {
        className: "tree-popup-wrap", closeButton: false, offset: [0, -markerSize],
      });
      primary.on("click", () => onTreeClick && onTreeClick(t));
      layerRef.current.addLayer(primary);
      bounds.push([t.lat, t.lng]);
    });

    if (bounds.length > 0) {
      try { mapInstance.current.fitBounds(bounds, { padding: [40, 40], maxZoom: 18 }); } catch {}
    }
  }, [trees, filter, threshold, markerSize, displayMode]);

  // Auto-Zoom Scan progress overlay — рисует сетку под-регионов с цветом
  // по статусу + прогрессивные деревья по мере прихода sub_complete событий.
  useEffect(() => {
    if (!scanGridRef.current || !scanTreesRef.current || !mapInstance.current) return;
    scanGridRef.current.clearLayers();
    scanTreesRef.current.clearLayers();
    if (!scanProgress || !scanProgress.regions) return;

    // Цвет/толщина по статусу — pulse-анимация для активных через className.
    const STATUS = {
      pending:    { color: "#9aa0a6", weight: 1.5, fillOpacity: 0.05, dashArray: "4 4", className: "" },
      capturing:  { color: "#3b82f6", weight: 2.5, fillOpacity: 0.10, dashArray: null,  className: "scan-region-pulse" },
      captured:   { color: "#1d4ed8", weight: 2,   fillOpacity: 0.08, dashArray: null,  className: "" },
      predicting: { color: "#f59e0b", weight: 2.5, fillOpacity: 0.10, dashArray: null,  className: "scan-region-pulse" },
      done:       { color: "#0F6E56", weight: 2,   fillOpacity: 0.15, dashArray: null,  className: "" },
      error:      { color: "#dc3545", weight: 2,   fillOpacity: 0.10, dashArray: "2 3", className: "" },
    };

    // Если scan был по полигону — рисуем его поверх grid'а, чтобы было видно
    // какая именно фигура определила фильтрацию детекций.
    if (scanProgress.polygon && scanProgress.polygon.length >= 3) {
      L.polygon(
        scanProgress.polygon.map((p) => [p.lat, p.lng]),
        {
          color: "#0F6E56", weight: 2.5, fillColor: "#0F6E56",
          fillOpacity: 0.03, dashArray: "6 3", interactive: false,
        },
      ).addTo(scanGridRef.current);
    }

    scanProgress.regions.forEach((r) => {
      const s = STATUS[r.status] || STATUS.pending;
      const nw = r.sub_bbox.nw, se = r.sub_bbox.se;
      const rect = L.rectangle(
        [[nw.lat, nw.lng], [se.lat, se.lng]],
        {
          color: s.color, weight: s.weight, fillColor: s.color,
          fillOpacity: s.fillOpacity, dashArray: s.dashArray || undefined,
          className: s.className,
          interactive: false,
        },
      ).addTo(scanGridRef.current);

      // Лейбл в углу: "r0c0 · capturing" / "r0c0 · 12 trees" / "r0c0 · err"
      let label = `r${r.row}c${r.col}`;
      if (r.status === "done") label += ` · ${r.tree_count} trees`;
      else if (r.status === "error") label += ` · err`;
      else label += ` · ${r.status}`;
      const icon = L.divIcon({
        className: "scan-region-label",
        html: `<span>${label}</span>`,
        iconSize: null, iconAnchor: [0, 0],
      });
      L.marker([nw.lat, nw.lng], { icon, interactive: false }).addTo(scanGridRef.current);
    });

    // Прогрессивные деревья — рисуем как circleMarker, цвет по confidence.
    // Простой dot — детальный displayMode (polygon/bbox) можно подключить позже.
    (scanProgress.trees || []).forEach((t) => {
      if (t.lat == null || t.lng == null) return;
      const conf = t.confidence ?? 0;
      const color = conf > 0.7 ? "#0F6E56" : conf > 0.5 ? "#5DCAA5" : "#EF9F27";
      L.circleMarker([t.lat, t.lng], {
        radius: 4, color: "#fff", weight: 1, fillColor: color, fillOpacity: 0.9,
        className: "scan-tree-pop",
        interactive: false,
      }).addTo(scanTreesRef.current);
    });

    // Авто-zoom к первому событию plan-а — чтобы пользователь видел сетку
    // целиком даже если изначально смотрел на другой район. Только один раз
    // (когда regions появились впервые и trees ещё пуст).
    if (scanProgress.regions.length && (scanProgress.trees || []).length === 0) {
      const all = scanProgress.regions.flatMap((r) => [
        [r.sub_bbox.nw.lat, r.sub_bbox.nw.lng],
        [r.sub_bbox.se.lat, r.sub_bbox.se.lng],
      ]);
      try { mapInstance.current.fitBounds(all, { padding: [60, 60], maxZoom: 17 }); } catch {}
    }
  }, [scanProgress]);

  // Image overlay (real image)
  useEffect(() => {
    if (!mapInstance.current) return;
    if (overlayRef.current) {
      mapInstance.current.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }
    if (showOverlay && image?.url && imageBounds) {
      const bounds = [
        [imageBounds.nw.lat, imageBounds.nw.lng],
        [imageBounds.se.lat, imageBounds.se.lng],
      ];
      overlayRef.current = L.imageOverlay(image.url, bounds, {
        opacity: overlayOpacity,
        className: "map-image-overlay",
        interactive: false,
      }).addTo(mapInstance.current);
    }
  }, [showOverlay, overlayOpacity, image, imageBounds]);

  // Draggable corner markers — создаём один раз когда включился режим corners_2 + есть image
  useEffect(() => {
    if (!cornersLayerRef.current || !mapInstance.current) return;
    cornersLayerRef.current.clearLayers();
    nwMarkerRef.current = null;
    seMarkerRef.current = null;
    rectRef.current = null;
    if (!image || geo?.mode !== "corners_2" || !geo.corners_2) return;

    const { nw, se } = geo.corners_2;
    const rect = L.rectangle(
      [[nw.lat, nw.lng], [se.lat, se.lng]],
      { color: "#1D9E75", weight: 2, dashArray: "6 6", fillColor: "#1D9E75", fillOpacity: 0.04, interactive: false },
    );
    cornersLayerRef.current.addLayer(rect);
    rectRef.current = rect;

    const makeHandle = (which, latlng, label) => {
      const icon = L.divIcon({
        className: "corner-handle",
        html: `<div class="corner-handle-inner"><span>${label}</span></div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });
      const m = L.marker(latlng, { icon, draggable: true, autoPan: true });
      m.on("drag", (e) => {
        const ll = e.target.getLatLng();
        setGeo((prev) => ({
          ...prev,
          corners_2: { ...prev.corners_2, [which]: { lat: ll.lat, lng: ll.lng } },
        }));
      });
      cornersLayerRef.current.addLayer(m);
      return m;
    };

    nwMarkerRef.current = makeHandle("nw", [nw.lat, nw.lng], "NW");
    seMarkerRef.current = makeHandle("se", [se.lat, se.lng], "SE");
  }, [image, geo?.mode, setGeo]);

  // Sync позиций маркеров и rectangle когда geo.corners_2 меняется (поля ввода / drag другого маркера)
  useEffect(() => {
    if (geo?.mode !== "corners_2" || !geo.corners_2) return;
    const { nw, se } = geo.corners_2;
    if (nwMarkerRef.current && !nwMarkerRef.current.dragging?._draggable?._moving) {
      nwMarkerRef.current.setLatLng([nw.lat, nw.lng]);
    }
    if (seMarkerRef.current && !seMarkerRef.current.dragging?._draggable?._moving) {
      seMarkerRef.current.setLatLng([se.lat, se.lng]);
    }
    rectRef.current?.setBounds([[nw.lat, nw.lng], [se.lat, se.lng]]);
  }, [geo?.corners_2]);

  // Capture-mode: рисуем прямоугольник мышью, отдаём bbox наружу на mouseup
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    const container = map.getContainer();
    const drawing = captureMode || scanMode;
    if (!drawing) {
      container.classList.remove("capture-cursor");
      if (captureRectRef.current) {
        map.removeLayer(captureRectRef.current);
        captureRectRef.current = null;
      }
      return;
    }

    container.classList.add("capture-cursor");
    // Scan mode рисуем в зелёный (DeepForest-ish, отличает от обычного capture).
    const rectColor = scanMode ? "#0F6E56" : "#EF9F27";
    let startLL = null;

    const onDown = (e) => {
      startLL = e.latlng;
      map.dragging.disable();
      if (captureRectRef.current) {
        map.removeLayer(captureRectRef.current);
        captureRectRef.current = null;
      }
      captureRectRef.current = L.rectangle(
        [startLL, startLL],
        { color: rectColor, weight: 2, fillColor: rectColor, fillOpacity: 0.12, dashArray: "4 4" },
      ).addTo(map);
    };
    const onMove = (e) => {
      if (!startLL || !captureRectRef.current) return;
      captureRectRef.current.setBounds([startLL, e.latlng]);
    };
    const onUp = (e) => {
      if (!startLL) return;
      const endLL = e.latlng;
      map.dragging.enable();
      const a = startLL, b = endLL;
      startLL = null;
      const nw = { lat: Math.max(a.lat, b.lat), lng: Math.min(a.lng, b.lng) };
      const se = { lat: Math.min(a.lat, b.lat), lng: Math.max(a.lng, b.lng) };
      // Минимальный размер защита: ≥ ~30 м на земле
      const dlat = Math.abs(nw.lat - se.lat);
      const dlng = Math.abs(se.lng - nw.lng);
      if (dlat < 0.0003 || dlng < 0.0003) {
        if (captureRectRef.current) {
          map.removeLayer(captureRectRef.current);
          captureRectRef.current = null;
        }
        return;
      }
      if (scanMode && onScanBbox) onScanBbox({ nw, se });
      else if (captureMode && onCaptureBbox) onCaptureBbox({ nw, se });
    };

    map.on("mousedown", onDown);
    map.on("mousemove", onMove);
    map.on("mouseup", onUp);
    return () => {
      map.off("mousedown", onDown);
      map.off("mousemove", onMove);
      map.off("mouseup", onUp);
      map.dragging.enable();
    };
  }, [captureMode, onCaptureBbox, scanMode, onScanBbox]);

  // Polygon-scan draw mode: клик добавляет вершину, double-click "закрывает"
  // полигон (отдаётся родителю в pendingPolygon, но scan НЕ запускается —
  // юзер увидит превью + явные кнопки Start/Clear). Right-click чистит всё.
  // Esc обрабатывается выше через setPolygonMode(false).
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    const container = map.getContainer();
    if (!polygonMode) {
      container.classList.remove("capture-cursor");
      return;
    }
    container.classList.add("capture-cursor");

    let vertices = [];
    let polyPreview = null;
    let lineDots = [];
    const POLY_COLOR = "#0F6E56";
    // Если pendingPolygon уже выставлен в App — закидываем его как
    // initial state, чтобы дорисовка после reset работала корректно.
    if (pendingPolygon && pendingPolygon.length >= 3) {
      vertices = pendingPolygon.slice();
    }

    const redraw = () => {
      if (polyPreview) { map.removeLayer(polyPreview); polyPreview = null; }
      lineDots.forEach((m) => map.removeLayer(m));
      lineDots = [];
      if (vertices.length === 0) return;
      vertices.forEach((v) => {
        const dot = L.circleMarker(v, {
          radius: 5, color: "#fff", weight: 2,
          fillColor: POLY_COLOR, fillOpacity: 1, interactive: false,
        }).addTo(map);
        lineDots.push(dot);
      });
      if (vertices.length >= 2 && vertices.length < 3) {
        polyPreview = L.polyline(vertices, {
          color: POLY_COLOR, weight: 2.5, dashArray: "4 4", interactive: false,
        }).addTo(map);
      } else if (vertices.length >= 3) {
        polyPreview = L.polygon(vertices, {
          color: POLY_COLOR, weight: 2.5, fillColor: POLY_COLOR,
          fillOpacity: 0.12, dashArray: "4 4", interactive: false,
        }).addTo(map);
      }
    };
    redraw();

    const onClick = (e) => {
      // Если уже есть pendingPolygon — игнорируем клики до Clear/Start.
      // Юзер должен явно сбросить (right-click) чтобы продолжить править.
      if (pendingPolygon && pendingPolygon.length >= 3) return;
      vertices.push([e.latlng.lat, e.latlng.lng]);
      redraw();
    };
    const onDblClick = (e) => {
      if (vertices.length < 3) return;
      L.DomEvent.stopPropagation(e);
      // Заморозили вершины — отдаём родителю в pendingPolygon. Превью
      // остаётся на карте (vertices не очищаем, только лочим клики).
      onPolygonComplete && onPolygonComplete(vertices.slice());
    };
    const onContextMenu = (e) => {
      // Right-click — clear current draw + pendingPolygon в родителе.
      L.DomEvent.preventDefault(e);
      L.DomEvent.stopPropagation(e);
      vertices = [];
      redraw();
      onPolygonReset && onPolygonReset();
    };

    map.on("click", onClick);
    map.on("dblclick", onDblClick);
    map.on("contextmenu", onContextMenu);
    map.doubleClickZoom.disable();

    return () => {
      map.off("click", onClick);
      map.off("dblclick", onDblClick);
      map.off("contextmenu", onContextMenu);
      map.doubleClickZoom.enable();
      if (polyPreview) map.removeLayer(polyPreview);
      lineDots.forEach((m) => map.removeLayer(m));
      container.classList.remove("capture-cursor");
    };
  }, [polygonMode, onPolygonComplete, onPolygonReset, pendingPolygon]);

  const zoomIn = () => mapInstance.current?.zoomIn();
  const zoomOut = () => mapInstance.current?.zoomOut();
  const reset = () => mapInstance.current?.setView(ASTANA_CENTER, 14);
  const fullscreen = () => {
    const el = mapRef.current?.parentElement;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen?.(); else document.exitFullscreen?.();
  };

  return (
    <div className="map-wrap">
      <div ref={mapRef} className="map" />
      <div className="map-top">
        <div className="layer-switcher">
          {[{ k: "satellite", label: "Satellite" }, { k: "clean", label: "Clean" }, { k: "streets", label: "Streets" }].map((l) => (
            <button key={l.k} className={baseLayer === l.k ? "active" : ""} onClick={() => setBaseLayer(l.k)}>
              {l.label}
            </button>
          ))}
        </div>
        <div className="map-coord">
          <Icon name="target" size={12} />
          <span>51.1605°N · 71.4704°E</span>
          <span className="map-coord-sep">·</span>
          <span>WGS 84</span>
        </div>
      </div>

      <div className="map-controls">
        <button onClick={zoomIn} title="Zoom in"><Icon name="plus" size={14} /></button>
        <button onClick={zoomOut} title="Zoom out"><Icon name="minus" size={14} /></button>
        <div className="map-controls-sep"></div>
        <button onClick={reset} title="Reset view"><Icon name="home" size={14} /></button>
        <button onClick={fullscreen} title="Fullscreen"><Icon name="expand" size={14} /></button>
      </div>
    </div>
  );
}

// ============ LEGEND ============
function Legend({ trees, filter, threshold }) {
  if (!trees) return null;
  const visible = trees.filter((t) => {
    if (t.confidence < threshold) return false;
    if (t.confidence > 0.7) return filter.high;
    if (t.confidence > 0.5) return filter.med;
    return filter.low;
  });
  return (
    <div className="legend">
      <div className="legend-head"><Icon name="tree" size={13} /><span>Detection Legend</span></div>
      <div className="legend-rows">
        <div className="legend-row"><span className="legend-dot" style={{ background: "#0F6E56" }}></span><span className="legend-label">High confidence</span><span className="legend-meta">&gt; 70%</span></div>
        <div className="legend-row"><span className="legend-dot" style={{ background: "#5DCAA5" }}></span><span className="legend-label">Medium</span><span className="legend-meta">50–70%</span></div>
        <div className="legend-row"><span className="legend-dot" style={{ background: "#EF9F27" }}></span><span className="legend-label">Low</span><span className="legend-meta">&lt; 50%</span></div>
      </div>
      <div className="legend-foot"><span>Visible</span><span className="legend-count">{visible.length} / {trees.length}</span></div>
    </div>
  );
}

// ============ TOAST ============
function Toast({ msg, kind = "info" }) {
  if (!msg) return null;
  return (
    <div className="toast" style={kind === "error" ? { background: "#dc3545" } : undefined}>
      <Icon name={kind === "error" ? "alert" : "check"} size={14} />
      <span>{msg}</span>
    </div>
  );
}

// ============ MAIN APP ============
function App() {
  const [image, setImage] = useState(null);             // {image_id, name, url, width, height, is_geotiff, ...}
  const [imageId, setImageId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [status, setStatus] = useState("idle");          // idle | running | done
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState(null);
  const [trees, setTrees] = useState(null);
  const [stats, setStats] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [duration, setDuration] = useState(null);

  const [model, setModel] = useState("deepforest");
  const [modelStatus, setModelStatus] = useState(null);
  const [geo, setGeo] = useState({ mode: "corners_2", corners_2: { nw: { lat: 51.17, lng: 71.46 }, se: { lat: 51.15, lng: 71.49 } } });

  const [filter, setFilter] = useState({ high: true, med: true, low: true });
  const [threshold, setThreshold] = useState(0.1);
  const [baseLayer, setBaseLayer] = useState("satellite");
  const [historyOpen, setHistoryOpen] = useState(true);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);
  const [toastKind, setToastKind] = useState("info");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dark, setDark] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  // "point" | "bbox" | "polygon" — единый переключатель отображения детекций.
  // По умолчанию polygon — главное визуальное преимущество YOLOv8-seg.
  const [displayMode, setDisplayMode] = useState("polygon");
  // "single" — workflow одного снимка, как было; "city" — aggregate map по всем сохранённым.
  const [viewMode, setViewMode] = useState("single");
  const [snapshots, setSnapshots] = useState([]);
  const [scans, setScans] = useState([]);
  const [aggregateTrees, setAggregateTrees] = useState([]);
  const [aggregateStats, setAggregateStats] = useState({ snapshot_count: 0, run_count: 0, total_trees: 0 });
  const [aggLoading, setAggLoading] = useState(false);

  const refreshAggregate = useCallback(async () => {
    setAggLoading(true);
    try {
      const [snaps, dets, stats, scanList] = await Promise.all([
        window.api.listSnapshots(),
        window.api.getDetections({ limit: 50000 }),
        window.api.aggregateStats(),
        window.api.listScans(),
      ]);
      setSnapshots(snaps);
      setAggregateTrees(window.api.adaptAggregateDetections(dets.detections));
      setAggregateStats(stats);
      setScans(scanList);
    } catch (e) {
      console.error("Aggregate refresh failed:", e);
    } finally {
      setAggLoading(false);
    }
  }, []);

  useEffect(() => {
    if (viewMode === "city") refreshAggregate();
  }, [viewMode, refreshAggregate]);

  const handleDeleteSnapshot = useCallback(async (imageId) => {
    if (!confirm(`Удалить снимок ${imageId}? Все его прогоны и детекции тоже уйдут.`)) return;
    try {
      await window.api.deleteSnapshot(imageId);
      await refreshAggregate();
    } catch (e) {
      console.error("Delete failed:", e);
    }
  }, [refreshAggregate]);

  const handleDeleteScan = useCallback(async (scanId, sub_count) => {
    if (!confirm(`Удалить scan-сессию ${scanId}? Каскадом уйдут ${sub_count || "все"} sub-region snapshots + их детекции.`)) return;
    try {
      const res = await window.api.deleteScan(scanId);
      await refreshAggregate();
      console.info("Scan deleted:", res);
    } catch (e) {
      console.error("Delete scan failed:", e);
    }
  }, [refreshAggregate]);
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [captureMode, setCaptureMode] = useState(false);
  const [captureZoom, setCaptureZoom] = useState(18);
  // Auto-Zoom Region Scan — параллельный режим к captureMode. На "Start" — войти,
  // на рисование bbox — отправить scan, после возврата автоматически в city view.
  const [scanMode, setScanMode] = useState(false);
  // Polygon scan mode — параллельный к scanMode (rectangle). User кликает по
  // карте N раз, добавляя вершины, double-click замыкает полигон, после чего
  // фронт сохраняет в pendingPolygon и показывает кнопку "Start". Так юзер
  // успевает посмотреть полигон до запуска (или удалить через right-click).
  const [polygonMode, setPolygonMode] = useState(false);
  const [pendingPolygon, setPendingPolygon] = useState(null);  // [[lat, lng], ...]
  const [scanRunning, setScanRunning] = useState(false);
  const [scanStatus, setScanStatus] = useState(null);
  // Streaming-прогресс scan: { regions: [{row, col, status: pending|capturing|predicting|done|error, sub_bbox, tree_count, error}], trees: [adapted detections so far], done: bool, total: {ok, total_trees, duration_ms} }
  // null = scan не активен (overlay / progress не рисуем).
  const [scanProgress, setScanProgress] = useState(null);
  // Tile-провайдер для capture / scan / Leaflet base layer.
  // Default = google (та же image base что Google Earth Pro = тренировочные
  // данные YOLO/Mask R-CNN). Список грузим из /api/providers чтобы один
  // источник правды о URL и max_zoom.
  const [tileProvider, setTileProvider] = useState("google");
  const [providersMap, setProvidersMap] = useState(null);

  useEffect(() => {
    window.api.providers()
      .then((r) => setProvidersMap(r.providers))
      .catch((e) => console.warn("Failed to load tile providers:", e));
  }, []);

  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  // Загрузка статуса бэкенда + модели
  useEffect(() => {
    window.api.status().then(setModelStatus).catch((e) => {
      console.error("Backend status failed:", e);
      // Surface the error on the status object so the header / status panel
      // can render a "Backend unreachable" banner instead of silently looking
      // healthy when the API is actually down (which used to bite us during
      // the demo: an empty-models response looked identical to "everything OK").
      setModelStatus({ models: {}, _error: e?.message || String(e) });
    });
    refreshHistory();
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const items = await window.api.history(10);
      setHistory(items);
    } catch (e) {
      console.warn("History fetch failed:", e);
    }
  }, []);

  const showToast = useCallback((m, kind = "info") => {
    setToast(m);
    setToastKind(kind);
    setTimeout(() => setToast(null), 2500);
  }, []);

  // ============ Upload ============
  const handleUpload = useCallback(async (file) => {
    setUploadError(null);
    setUploading(true);
    try {
      const meta = await window.api.upload(file);
      setImage({
        ...meta,
        name: meta.filename,
        url: window.api.imageUrl(meta.image_id),
      });
      setImageId(meta.image_id);
      setTrees(null);     // clear mock
      setStats(null);
      setJobId(null);
      setStatus("idle");

      // Auto-pick GPS mode if GeoTIFF
      if (meta.is_geotiff) {
        setGeo({ ...geo, mode: "geotiff" });
        showToast(`Uploaded ${meta.filename} (GeoTIFF detected)`);
      } else {
        showToast(`Uploaded ${meta.filename}`);
      }
    } catch (e) {
      setUploadError(e.message);
      showToast("Upload failed: " + e.message, "error");
    } finally {
      setUploading(false);
    }
  }, [geo, showToast]);

  const handleClear = useCallback(() => {
    setImage(null);
    setImageId(null);
    setTrees(null);
    setStats(null);
    setJobId(null);
    setStatus("idle");
    setUploadError(null);
    setShowOverlay(false);
  }, []);

  // ============ Capture from map ============
  const handleCaptureBbox = useCallback(async (bbox) => {
    setCaptureMode(false);
    setUploadError(null);
    setUploading(true);
    try {
      const meta = await window.api.captureFromMap({
        nw: bbox.nw,
        se: bbox.se,
        zoom: captureZoom,
        provider: tileProvider,
      });
      setImage({
        ...meta,
        name: meta.filename,
        url: window.api.imageUrl(meta.image_id),
      });
      setImageId(meta.image_id);
      setTrees(null);
      setStats(null);
      setJobId(null);
      setStatus("idle");
      // bounds уже есть в meta — выставим режим corners_2 с этими углами
      if (meta.bounds) {
        setGeo({ mode: "corners_2", corners_2: meta.bounds });
        setShowOverlay(true);
      }
      showToast(`Captured ${meta.width}×${meta.height} from map (zoom ${captureZoom})`);
    } catch (e) {
      setUploadError(e.message);
      showToast("Capture failed: " + e.message, "error");
    } finally {
      setUploading(false);
    }
  }, [captureZoom, showToast, tileProvider]);

  // ============ Auto-Zoom Region Scan (streaming) ============
  // На draw bbox открываем NDJSON-стрим к /api/scan_region/stream и обновляем
  // scanProgress инкрементально:
  //   - "plan" событие → создаёт grid с N пустых regions со status=pending
  //   - "capturing"/"predicting" → меняет status конкретного row/col
  //   - "sub_complete" → status=done + кладёт detections в scanProgress.trees
  //     (progressive рендеринг на карте без ожидания всего скана)
  //   - "sub_error" → status=error + сообщение
  //   - "done" → итог + soft-переход в city view
  const handleScanBbox = useCallback(async (bbox, polygon = null) => {
    setScanMode(false);
    setPolygonMode(false);
    setScanRunning(true);
    setScanStatus(polygon ? "Запрашиваю план (polygon scan)…" : "Запрашиваю план…");
    setScanProgress({ regions: [], trees: [], done: false, total: null, polygon });
    try {
      await window.api.scanRegionStream(
        {
          nw: bbox.nw, se: bbox.se, zoom: 19,
          model, confidence: threshold,
          maxSubregions: 9, provider: tileProvider,
          polygon: polygon || undefined,
        },
        (ev) => {
          if (ev.type === "plan") {
            const regions = ev.sub_regions.map((r) => ({
              row: r.row, col: r.col,
              sub_bbox: { nw: r.nw, se: r.se },
              status: "pending",
              tree_count: 0,
              error: null,
            }));
            setScanProgress((p) => ({ ...p, regions }));
            setScanStatus(`Plan: ${ev.sub_count} sub-regions · ${ev.provider} · z${ev.zoom}`);
          } else if (ev.type === "capturing" || ev.type === "predicting" || ev.type === "capture_done") {
            const newStatus = ev.type === "capturing"
              ? "capturing"
              : (ev.type === "capture_done" ? "captured" : "predicting");
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col ? { ...r, status: newStatus } : r,
              ),
            }));
            setScanStatus(`r${ev.row}c${ev.col}: ${newStatus}…`);
          } else if (ev.type === "sub_complete") {
            // Адаптируем детекции к UI-формату того же что используется в city view.
            const adapted = window.api.adaptAggregateDetections(
              ev.detections.map((d) => ({
                ...d,
                local_id: d.id,
                model: ev.model,
                image_id: ev.snapshot_id,
                job_id: ev.job_id,
              }))
            );
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col
                  ? { ...r, status: "done", tree_count: ev.tree_count }
                  : r,
              ),
              trees: [...p.trees, ...adapted],
            }));
            setScanStatus(`r${ev.row}c${ev.col}: ${ev.tree_count} trees`);
          } else if (ev.type === "sub_error") {
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col
                  ? { ...r, status: "error", error: ev.error }
                  : r,
              ),
            }));
            console.warn(`scan sub-r${ev.row}c${ev.col} ${ev.stage} failed: ${ev.error}`);
          } else if (ev.type === "done") {
            setScanProgress((p) => ({
              ...p,
              done: true,
              total: { total_trees: ev.total_trees, duration_ms: ev.duration_ms, sub_count: ev.sub_count },
            }));
            const okMsg = `Scan done · ${ev.total_trees} trees · ${(ev.duration_ms / 1000).toFixed(1)} s`;
            showToast(okMsg);
            setScanStatus(okMsg);
          } else if (ev.type === "fatal") {
            console.error("Scan fatal:", ev.error);
            showToast("Scan crashed: " + ev.error, "error");
          }
        },
      );
      // Подтягиваем aggregate (city view карточки и snapshots-list)
      await refreshAggregate();
    } catch (e) {
      console.error("Auto-Zoom Scan failed:", e);
      setScanStatus(null);
      showToast("Scan failed: " + e.message, "error");
    } finally {
      setScanRunning(false);
      // Оставляем scanProgress на экране ~5 сек чтобы видеть итог, потом гасим
      // overlay (но детекции остаются в city view / aggregate).
      setTimeout(() => {
        setScanProgress(null);
        setScanStatus(null);
      }, 5000);
    }
  }, [model, threshold, showToast, refreshAggregate, tileProvider]);

  // Polygon-scan: double-click на карте сохраняет polygon в pendingPolygon
  // (НЕ запускает scan сразу). Юзер видит превью, может удалить через
  // right-click или подтвердить кнопкой "Start Polygon Scan".
  const handlePolygonComplete = useCallback((polygon) => {
    if (!polygon || polygon.length < 3) return;
    setPendingPolygon(polygon);  // [[lat, lng], ...] точки
  }, []);

  // Запуск scan'а из pendingPolygon — считаем axis-aligned bbox и шлём.
  const handleStartPolygonScan = useCallback(() => {
    if (!pendingPolygon || pendingPolygon.length < 3) return;
    const lats = pendingPolygon.map(([lat]) => lat);
    const lngs = pendingPolygon.map(([, lng]) => lng);
    const bbox = {
      nw: { lat: Math.max(...lats), lng: Math.min(...lngs) },
      se: { lat: Math.min(...lats), lng: Math.max(...lngs) },
    };
    const polyPoints = pendingPolygon.map(([lat, lng]) => ({ lat, lng }));
    setPendingPolygon(null);
    handleScanBbox(bbox, polyPoints);
  }, [pendingPolygon, handleScanBbox]);

  // Right-click / Clear button — сбрасывает pendingPolygon и любые
  // полу-нарисованные вершины (MapView через ту же ссылку перерисует).
  const handleClearPolygon = useCallback(() => {
    setPendingPolygon(null);
  }, []);

  // ============ Run detection ============
  const runDetection = useCallback(async () => {
    if (!imageId) {
      showToast("Сначала загрузи снимок", "error");
      return;
    }
    setStatus("running");
    setProgress(0);
    setEta(null);
    setTrees(null);
    setStats(null);
    setJobId(null);
    setDuration(null);
    setPredictError(null);

    // Псевдо-прогресс пока ждём бэкенд (FastAPI отвечает синхронно)
    const startTime = Date.now();
    let p = 0;
    let progressInterval = null;
    progressInterval = setInterval(() => {
      // Asymptotic to 90%, real 100% выставится при завершении
      p = Math.min(90, p + 1.5 + Math.random() * 2);
      setProgress(Math.floor(p));
      const elapsed = (Date.now() - startTime) / 1000;
      setEta(Math.max(0, (elapsed / Math.max(p, 1)) * 100 - elapsed));
    }, 180);

    try {
      const result = await window.api.predict({
        image_id: imageId,
        model,
        confidence: threshold,
        geo,
      });
      clearInterval(progressInterval);
      setProgress(100);
      setEta(0);

      const adapted = window.api.adaptDetectionsForUI(result.detections);
      setTrees(adapted);
      setStats(result.stats);
      setJobId(result.job_id);
      setStatus("done");
      setDuration(result.duration_ms);
      showToast(`Detection complete · ${adapted.length} trees in ${result.duration_ms} ms`);
      refreshHistory();
      // Свежий прогон → инвалидируем aggregate-кэш (он подгружается лениво при заходе в city view).
      if (viewMode === "city") refreshAggregate();
    } catch (e) {
      clearInterval(progressInterval);
      setPredictError(e.message);
      setStatus("idle");
      setProgress(0);
      showToast("Detection failed: " + e.message, "error");
    }
  }, [imageId, model, threshold, geo, showToast, refreshHistory, viewMode, refreshAggregate]);

  // ============ Export ============
  const handleExport = useCallback(async (kind) => {
    if (!jobId) {
      showToast("Запусти детекцию сначала", "error");
      return;
    }
    try {
      await window.api.exportFile(jobId, kind);
      showToast(`${kind.toUpperCase()} downloaded`);
    } catch (e) {
      showToast("Export failed: " + e.message, "error");
    }
  }, [jobId, showToast]);

  // ============ Load history entry ============
  const handleLoadHistory = useCallback(async (h) => {
    if (!h.image_id) return;
    showToast("Loading from history…");
    // For simplicity: just mark current image, user can re-run
    setImageId(h.image_id);
    setImage({
      image_id: h.image_id,
      filename: h.filename,
      name: h.filename,
      url: window.api.imageUrl(h.image_id),
      width: 0,
      height: 0,
    });
  }, [showToast]);

  return (
    <div className="app">
      <aside className="sidebar">
        <Header dark={dark} onToggleDark={() => setDark(!dark)} modelStatus={modelStatus} />
        <ViewModeSwitch mode={viewMode} setMode={setViewMode} agg={aggregateStats} />
        <div className="sidebar-scroll">
          {viewMode === "single" ? (
            <>
              <UploadZone
                image={image}
                uploading={uploading}
                scanning={status === "running"}
                onUpload={handleUpload}
                onClear={handleClear}
                error={uploadError}
                captureMode={captureMode}
                onStartCapture={() => { setScanMode(false); setCaptureMode(true); }}
                onCancelCapture={() => setCaptureMode(false)}
                captureZoom={captureZoom}
                setCaptureZoom={setCaptureZoom}
                scanMode={scanMode}
                onStartScan={() => { setCaptureMode(false); setPolygonMode(false); setScanMode(true); }}
                onCancelScan={() => setScanMode(false)}
                scanRunning={scanRunning}
                scanStatus={scanStatus}
                model={model}
                tileProvider={tileProvider}
                setTileProvider={setTileProvider}
                providersMap={providersMap}
                scanProgress={scanProgress}
                polygonMode={polygonMode}
                onStartPolygon={() => { setCaptureMode(false); setScanMode(false); setPolygonMode(true); setPendingPolygon(null); }}
                onCancelPolygon={() => { setPolygonMode(false); setPendingPolygon(null); }}
                pendingPolygon={pendingPolygon}
                onStartPolygonScan={handleStartPolygonScan}
                onClearPolygon={handleClearPolygon}
              />
              <DetectionControls
                canRun={!!imageId}
                status={status}
                progress={progress}
                eta={eta}
                onRun={runDetection}
                model={model}
                setModel={setModel}
                modelStatus={modelStatus}
                error={predictError}
              />
              <GeoPanel geo={geo} setGeo={setGeo} image={image} />
              <StatsPanel trees={trees} stats={stats} />
              <MapLayersPanel
                baseLayer={baseLayer} setBaseLayer={setBaseLayer}
                showOverlay={showOverlay} setShowOverlay={setShowOverlay}
                displayMode={displayMode} setDisplayMode={setDisplayMode}
                overlayOpacity={overlayOpacity} setOverlayOpacity={setOverlayOpacity}
                hasImage={!!image} hasTrees={!!trees && trees.length > 0}
                hasMasks={!!trees && trees.some((t) => t.mask_polygon_geo && t.mask_polygon_geo.length >= 3)}
              />
              <ConfidenceFilter filter={filter} setFilter={setFilter} trees={trees} />
              <ExportPanel enabled={!!jobId} onExport={handleExport} />
              <HistoryPanel open={historyOpen} setOpen={setHistoryOpen} history={history} onLoad={handleLoadHistory} />
            </>
          ) : (
            <>
              <AggregateStatsPanel stats={aggregateStats} />
              <ScansList
                scans={scans}
                onDelete={handleDeleteScan}
                loading={aggLoading}
              />
              <SnapshotsList
                snapshots={snapshots}
                onDelete={handleDeleteSnapshot}
                loading={aggLoading}
              />
              <MapLayersPanel
                baseLayer={baseLayer} setBaseLayer={setBaseLayer}
                showOverlay={false} setShowOverlay={() => {}}
                displayMode={displayMode} setDisplayMode={setDisplayMode}
                overlayOpacity={overlayOpacity} setOverlayOpacity={setOverlayOpacity}
                hasImage={false} hasTrees={aggregateTrees.length > 0}
                hasMasks={aggregateTrees.some((t) => t.mask_polygon_geo && t.mask_polygon_geo.length >= 3)}
              />
              <ConfidenceFilter filter={filter} setFilter={setFilter} trees={aggregateTrees} />
            </>
          )}
        </div>
        <div className="sidebar-footer">
          <button className="footer-btn" onClick={() => setSettingsOpen(!settingsOpen)}>
            <Icon name="settings" size={13} />
            <span>Settings</span>
          </button>
        </div>
        {settingsOpen && (
          <div className="settings-popover">
            <div className="settings-head">
              <span>Detection Settings</span>
              <button onClick={() => setSettingsOpen(false)}><Icon name="x" size={12} /></button>
            </div>
            <div className="settings-row">
              <label>Confidence threshold</label>
              <div className="settings-slider">
                <input type="range" min="0.05" max="0.95" step="0.05" value={threshold}
                  onChange={(e) => setThreshold(+e.target.value)} />
                <span className="settings-val">{Math.round(threshold * 100)}%</span>
              </div>
            </div>
          </div>
        )}
      </aside>

      <main className="main">
        <MapView
          trees={viewMode === "city" ? aggregateTrees : trees}
          filter={filter}
          threshold={threshold}
          baseLayer={baseLayer}
          setBaseLayer={setBaseLayer}
          markerSize={viewMode === "city" ? 5 : 7}
          scanning={status === "running"}
          showOverlay={viewMode === "single" && showOverlay}
          displayMode={displayMode}
          overlayOpacity={overlayOpacity}
          image={viewMode === "city" ? null : image}
          imageBounds={viewMode === "city" ? null : ((geo.mode === "corners_2" ? geo.corners_2 : null) || image?.bounds)}
          geo={geo}
          setGeo={setGeo}
          captureMode={viewMode === "single" && captureMode}
          onCaptureBbox={handleCaptureBbox}
          scanMode={viewMode === "single" && scanMode}
          onScanBbox={handleScanBbox}
          tileProvider={tileProvider}
          providersMap={providersMap}
          scanProgress={scanProgress}
          polygonMode={viewMode === "single" && polygonMode}
          onPolygonComplete={handlePolygonComplete}
          pendingPolygon={pendingPolygon}
          onPolygonReset={handleClearPolygon}
        />
        <Legend
          trees={viewMode === "city" ? aggregateTrees : trees}
          filter={filter}
          threshold={threshold}
        />
        <div className="map-info-chip">
          <div className="chip-row">
            <span className="chip-k">Region</span>
            <span className="chip-v">Astana</span>
          </div>
          <div className="chip-row">
            <span className="chip-k">Model</span>
            <span className="chip-v">{model.toUpperCase()}{duration ? ` · ${duration}ms` : ""}</span>
          </div>
          {stats?.tree_count != null && (
            <div className="chip-row">
              <span className="chip-k">Detected</span>
              <span className="chip-v">{stats.tree_count} trees</span>
            </div>
          )}
        </div>
      </main>

      <Toast msg={toast} kind={toastKind} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
