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
function UploadZone({ image, onUpload, onClear, scanning, uploading, error, captureMode, onStartCapture, onCancelCapture, captureZoom, setCaptureZoom }) {
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
      {!image && !captureMode && (
        <div className="capture-row">
          <button
            type="button"
            className="btn-capture"
            onClick={onStartCapture}
            disabled={uploading}
            title="Нарисовать прямоугольник на карте, скачать тайлы Esri"
          >
            <Icon name="target" size={14} />
            <span>Capture from map</span>
          </button>
          <div className="capture-zoom">
            <label>Zoom</label>
            <input
              type="number" min="14" max="19" step="1"
              value={captureZoom}
              onChange={(e) => setCaptureZoom(Math.max(14, Math.min(19, +e.target.value || 18)))}
            />
          </div>
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
    const corners = { ...(geo.corners_2 || { nw: { lat: 51.17, lng: 71.46 }, se: { lat: 51.15, lng: 71.49 } }) };
    corners[which] = { ...corners[which], [key]: parseFloat(value) || 0 };
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
  const avgConf = total ? Math.round((trees.reduce((s, t) => s + t.confidence, 0) / total) * 100) : 0;
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
function MapLayersPanel({ baseLayer, setBaseLayer, showOverlay, setShowOverlay, showMarkers, setShowMarkers, overlayOpacity, setOverlayOpacity, hasImage, hasTrees }) {
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
          <label className={`layer-toggle ${showMarkers ? "active" : ""} ${!hasTrees ? "disabled" : ""}`}>
            <input type="checkbox" checked={showMarkers} disabled={!hasTrees} onChange={(e) => setShowMarkers(e.target.checked)} />
            <span className="layer-icon markers"><Icon name="tree" size={12} /></span>
            <span className="layer-name">Tree Markers</span>
            <span className={`layer-switch ${showMarkers ? "on" : ""}`}><span className="layer-switch-thumb"></span></span>
          </label>
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
function MapView({ trees, filter, threshold, baseLayer, setBaseLayer, onTreeClick, markerSize, scanning, showOverlay, showMarkers, overlayOpacity, image, imageBounds, geo, setGeo, captureMode, onCaptureBbox }) {
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

    return () => { map.remove(); mapInstance.current = null; };
  }, []);

  useEffect(() => {
    if (!mapInstance.current) return;
    const urls = {
      satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      streets: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      clean: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    };
    if (tileLayerRef.current) mapInstance.current.removeLayer(tileLayerRef.current);
    tileLayerRef.current = L.tileLayer(urls[baseLayer] || urls.satellite, { maxZoom: 19, attribution: "" }).addTo(mapInstance.current);
  }, [baseLayer]);

  useEffect(() => {
    if (!layerRef.current || !mapInstance.current) return;
    layerRef.current.clearLayers();
    if (!trees || !showMarkers) return;
    const visible = trees.filter((t) => {
      if (t.confidence < threshold) return false;
      if (t.confidence > 0.7) return filter.high;
      if (t.confidence > 0.5) return filter.med;
      return filter.low;
    });

    const bounds = [];
    visible.forEach((t) => {
      const color = t.confidence > 0.7 ? "#0F6E56" : t.confidence > 0.5 ? "#5DCAA5" : "#EF9F27";
      const marker = L.circleMarker([t.lat, t.lng], {
        radius: markerSize,
        fillColor: color,
        color: "#ffffff",
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.92,
      });
      marker.bindPopup(
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
         </div>`,
        { className: "tree-popup-wrap", closeButton: false, offset: [0, -markerSize] }
      );
      marker.on("click", () => onTreeClick && onTreeClick(t));
      layerRef.current.addLayer(marker);
      bounds.push([t.lat, t.lng]);
    });

    if (bounds.length > 0) {
      try { mapInstance.current.fitBounds(bounds, { padding: [40, 40], maxZoom: 18 }); } catch {}
    }
  }, [trees, filter, threshold, markerSize, showMarkers]);

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
    if (!captureMode) {
      container.classList.remove("capture-cursor");
      if (captureRectRef.current) {
        map.removeLayer(captureRectRef.current);
        captureRectRef.current = null;
      }
      return;
    }

    container.classList.add("capture-cursor");
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
        { color: "#EF9F27", weight: 2, fillColor: "#EF9F27", fillOpacity: 0.12, dashArray: "4 4" },
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
      onCaptureBbox && onCaptureBbox({ nw, se });
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
  }, [captureMode, onCaptureBbox]);

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
  const [showMarkers, setShowMarkers] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [captureMode, setCaptureMode] = useState(false);
  const [captureZoom, setCaptureZoom] = useState(18);

  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  // Загрузка статуса бэкенда + модели
  useEffect(() => {
    window.api.status().then(setModelStatus).catch((e) => {
      console.warn("Backend status failed:", e);
      setModelStatus({ models: {} });
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
  }, [captureZoom, showToast]);

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
    } catch (e) {
      clearInterval(progressInterval);
      setPredictError(e.message);
      setStatus("idle");
      setProgress(0);
      showToast("Detection failed: " + e.message, "error");
    }
  }, [imageId, model, threshold, geo, showToast, refreshHistory]);

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
        <div className="sidebar-scroll">
          <UploadZone
            image={image}
            uploading={uploading}
            scanning={status === "running"}
            onUpload={handleUpload}
            onClear={handleClear}
            error={uploadError}
            captureMode={captureMode}
            onStartCapture={() => setCaptureMode(true)}
            onCancelCapture={() => setCaptureMode(false)}
            captureZoom={captureZoom}
            setCaptureZoom={setCaptureZoom}
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
            showMarkers={showMarkers} setShowMarkers={setShowMarkers}
            overlayOpacity={overlayOpacity} setOverlayOpacity={setOverlayOpacity}
            hasImage={!!image} hasTrees={!!trees && trees.length > 0}
          />
          <ConfidenceFilter filter={filter} setFilter={setFilter} trees={trees} />
          <ExportPanel enabled={!!jobId} onExport={handleExport} />
          <HistoryPanel open={historyOpen} setOpen={setHistoryOpen} history={history} onLoad={handleLoadHistory} />
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
          trees={trees}
          filter={filter}
          threshold={threshold}
          baseLayer={baseLayer}
          setBaseLayer={setBaseLayer}
          markerSize={7}
          scanning={status === "running"}
          showOverlay={showOverlay}
          showMarkers={showMarkers}
          overlayOpacity={overlayOpacity}
          image={image}
          imageBounds={image?.bounds || (geo.mode === "corners_2" ? geo.corners_2 : null)}
          geo={geo}
          setGeo={setGeo}
          captureMode={captureMode}
          onCaptureBbox={handleCaptureBbox}
        />
        <Legend trees={trees} filter={filter} threshold={threshold} />
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
