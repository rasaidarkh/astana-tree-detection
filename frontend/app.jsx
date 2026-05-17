/* =========================================================================
   Astana Tree Detection — UI v2 (design rethink)
   Map-first, progressive disclosure, MVP-grade.
   ========================================================================= */

const { useState, useEffect, useRef, useCallback, useMemo } = React;

const ASTANA_CENTER = [51.1605, 71.4704];
const ASTANA_ZOOM = 13;

/* ==================================================================
   Icon set (subset — only what's used)
   ================================================================== */
function Icon({ name, size = 16, stroke = 1.8 }) {
  const paths = {
    map:        <><path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z" /><path d="M9 3v15M15 6v15" /></>,
    image:      <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="M21 16l-5-5-9 9" /></>,
    settings:   <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></>,
    grid:       <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    polygon:    <><path d="M12 3l8 5-3 11h-10l-3-11z" /></>,
    upload:     <><path d="M12 3v12M7 8l5-5 5 5M5 17v3a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3" /></>,
    download:   <><path d="M12 3v13M7 11l5 5 5-5M5 21h14" /></>,
    play:       <path d="M6 4l14 8-14 8z" fill="currentColor" stroke="none" />,
    plus:       <><path d="M12 5v14M5 12h14" /></>,
    x:          <><path d="M6 6l12 12M18 6L6 18" /></>,
    chevron:    <path d="M9 6l6 6-6 6" />,
    layers:     <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5M3 18l9 5 9-5" /></>,
    history:    <><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></>,
    sun:        <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    moon:       <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
    alert:      <><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.7L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0z" /></>,
    check:      <path d="M4 12l5 5L20 6" />,
    trash:      <><path d="M4 7h16M9 7V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" /></>,
    refresh:    <><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></>,
    file:       <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /></>,
    target:     <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="1" fill="currentColor" /></>,
    flame:      <path d="M12 3c0 4 4 5 4 9a4 4 0 0 1-8 0c0-2 1-3 1-5 0-1 1-2 3-4z" />,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" style={{flexShrink: 0}}>
      {paths[name]}
    </svg>
  );
}

/* ==================================================================
   TopBar — brand, view switcher, settings
   ================================================================== */
function TopBar({ view, setView, onOpenSettings, settingsOpen, backendStatus, dark, setDark }) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="topbar-mark"><Icon name="polygon" size={16} stroke={2} /></div>
        <div>
          <div className="topbar-title">Astana Tree Detection</div>
          <div className="topbar-sub">Urban canopy inventory · AITU 2026</div>
        </div>
      </div>

      <div className="topbar-spacer" />

      <div className="topbar-views" role="tablist">
        <button
          className={`topbar-view-btn ${view === "map" ? "active" : ""}`}
          onClick={() => setView("map")}
          role="tab"
        >
          <Icon name="map" size={14} />
          <span>Map</span>
        </button>
        <button
          className={`topbar-view-btn ${view === "image" ? "active" : ""}`}
          onClick={() => setView("image")}
          role="tab"
        >
          <Icon name="image" size={14} />
          <span>Image</span>
        </button>
      </div>

      <div className="topbar-spacer" />

      <div className="topbar-actions">
        <button className="icon-btn" onClick={() => setDark(!dark)} title={dark ? "Switch to light" : "Switch to dark"}>
          <Icon name={dark ? "sun" : "moon"} size={16} />
        </button>
        <button
          className={`icon-btn ${settingsOpen ? "active" : ""}`}
          onClick={onOpenSettings}
          title="Settings"
        >
          <Icon name="settings" size={16} />
          <span
            className="status-dot"
            style={{
              position: "absolute", bottom: 4, right: 4,
              background: backendStatus === "ok" ? "var(--success)" : backendStatus === "warn" ? "var(--warning)" : "var(--danger)",
              width: 6, height: 6, border: "1.5px solid var(--surface)",
            }}
          />
        </button>
      </div>
    </header>
  );
}

/* ==================================================================
   StatsCard (top-left in map view)
   ================================================================== */
function StatsCard({ stats, onOpenDrawer }) {
  const t = (stats && stats.total_trees) || 0;
  const s = (stats && stats.snapshot_count) || 0;
  const r = (stats && stats.run_count) || 0;
  return (
    <div className="float float-tl stats-card">
      <div className="stats-eyebrow">Astana · city aggregate</div>
      <div className="stats-headline">
        {t.toLocaleString()}
        <span className="unit">trees</span>
      </div>
      <div className="stats-row">
        <span><b>{s}</b> snapshots</span>
        <span><b>{r}</b> runs</span>
      </div>
      <button className="stats-btn" onClick={onOpenDrawer}>
        <Icon name="history" size={11} />
        <span>View history</span>
      </button>
    </div>
  );
}

/* ==================================================================
   ScanActionStack (bottom-left in map view)
   ================================================================== */
function ScanActionStack({ scanMode, polygonMode, onStartScan, onStartPolygon, onCancel, pendingPolygon, onStartPolygonScan, onClearPolygon }) {
  // 3 visual states: idle / drawing / pending-polygon
  if (pendingPolygon && pendingPolygon.length >= 3) {
    return (
      <div className="float float-bl action-stack drawing">
        <div className="action-hint">
          <span className="pulse"></span>
          <span>Polygon ready · <b>{pendingPolygon.length}</b> vertices</span>
        </div>
        <button className="action-btn primary" onClick={onStartPolygonScan}>
          <span className="icon-wrap"><Icon name="play" size={13} /></span>
          <span>Start scan</span>
        </button>
        <button className="action-btn" onClick={onClearPolygon}>
          <span className="icon-wrap"><Icon name="refresh" size={13} /></span>
          <span>Redraw</span>
        </button>
      </div>
    );
  }
  if (scanMode || polygonMode) {
    return (
      <div className="float float-bl action-stack drawing">
        <div className="action-hint">
          <span className="pulse"></span>
          <span>
            {scanMode
              ? "Drag a rectangle on the map"
              : "Click to add vertices · double-click to finish · right-click to clear"}
          </span>
        </div>
        <button className="action-btn" onClick={onCancel}>
          <span className="icon-wrap"><Icon name="x" size={13} /></span>
          <span>Cancel</span>
        </button>
      </div>
    );
  }
  return (
    <div className="float float-bl action-stack">
      <button className="action-btn primary" onClick={onStartScan}>
        <span className="icon-wrap"><Icon name="grid" size={14} /></span>
        <span>Scan area</span>
      </button>
      <button className="action-btn" onClick={onStartPolygon}>
        <span className="icon-wrap"><Icon name="polygon" size={14} /></span>
        <span>Polygon scan</span>
      </button>
    </div>
  );
}

/* ==================================================================
   Display & Filter strip (bottom-right in map view)
   ================================================================== */
function DisplayStrip({ displayMode, setDisplayMode, hasMasks, hasTrees, threshold, setThreshold, filter, setFilter, treeCount }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const onDocClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  return (
    <div className="float-br" style={{ position: "absolute", bottom: 16, right: 16, zIndex: "var(--z-float)" }} ref={ref}>
      <div className="controls-strip">
        {/* Display mode */}
        <div className="control-group">
          {[
            { k: "point", label: "Point" },
            { k: "bbox", label: "BBox" },
            { k: "polygon", label: "Polygon", disabled: !hasMasks },
            { k: "heat", label: "Heat" },
          ].map((it) => (
            <button
              key={it.k}
              className={`seg-btn ${displayMode === it.k ? "active" : ""}`}
              onClick={() => !it.disabled && hasTrees && setDisplayMode(it.k)}
              disabled={it.disabled || !hasTrees}
              title={it.disabled ? "No segmentation masks available" : it.label}
            >
              {it.label}
            </button>
          ))}
        </div>

        {/* Filters chip */}
        <div className="control-group">
          <button className="chip-btn" onClick={() => setOpen(!open)}>
            <Icon name="layers" size={12} />
            <span>Filters</span>
            <span className="muted mono" style={{ fontSize: 10 }}>·</span>
            <span className="mono" style={{ fontSize: 11 }}>≥{Math.round(threshold * 100)}%</span>
          </button>
        </div>
      </div>

      {open && (
        <div className="popover" style={{ position: "absolute", right: 0, bottom: 48 }}>
          <div className="popover-row">
            <div className="popover-label" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Min confidence</span>
              <span className="mono">{Math.round(threshold * 100)}%</span>
            </div>
            <input
              type="range" min={0} max={1} step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="range"
            />
          </div>
          <div className="popover-row">
            <div className="popover-label">Confidence tiers</div>
            <div className="tier-chips">
              {[
                { k: "high", label: "High", color: "var(--conf-high)" },
                { k: "med",  label: "Med",  color: "var(--conf-mid)"  },
                { k: "low",  label: "Low",  color: "var(--conf-low)"  },
              ].map((t) => (
                <button
                  key={t.k}
                  className={`tier-chip ${filter[t.k] ? "on" : ""}`}
                  onClick={() => setFilter({ ...filter, [t.k]: !filter[t.k] })}
                >
                  <span className="dot" style={{ background: t.color, opacity: filter[t.k] ? 1 : 0.3 }}></span>
                  <span>{t.label}</span>
                </button>
              ))}
            </div>
          </div>
          {treeCount != null && (
            <div className="legend-count">Showing <b>{treeCount.visible.toLocaleString()}</b> / {treeCount.total.toLocaleString()} trees</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ==================================================================
   ScanProgressCard (top-center, only during scan)
   ================================================================== */
function ScanProgressCard({ scanProgress, onCancel }) {
  if (!scanProgress) return null;
  const regions = scanProgress.regions || [];
  const total = regions.length;
  const done = regions.filter((r) => r.status === "done").length;
  const errs = regions.filter((r) => r.status === "error").length;
  const treeCount = scanProgress.trees ? scanProgress.trees.length : 0;
  const cols = Math.max(1, Math.ceil(Math.sqrt(total || 1)));
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div className="float float-tc scan-progress">
      <div className="scan-progress-head">
        <div className="scan-spinner" />
        <div className="scan-progress-title">
          {scanProgress.done ? "Scan complete" : "Scanning area"}
        </div>
        <div className="scan-progress-meta mono">{done}/{total}{errs ? ` · ${errs} err` : ""}</div>
      </div>

      {total > 0 && (
        <div
          className="scan-progress-grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        >
          {regions.map((r) => (
            <div
              key={`${r.row}-${r.col}`}
              className={`scan-cell ${r.status}`}
              title={`r${r.row}c${r.col}: ${r.status}${r.tree_count ? ` · ${r.tree_count} trees` : ""}${r.error ? ` · ${r.error}` : ""}`}
            />
          ))}
        </div>
      )}

      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="scan-progress-counter">
          <b>{treeCount.toLocaleString()}</b>
          <span className="muted">trees found</span>
        </div>
        <div className="scan-progress-actions">
          {onCancel && !scanProgress.done && (
            <button className="ghost-btn danger" onClick={onCancel}>
              <Icon name="x" size={11} />
              <span>Cancel</span>
            </button>
          )}
        </div>
      </div>
      <div className="scan-progress-line">
        <div className="scan-progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/* ==================================================================
   Welcome (shown on first load when no data)
   ================================================================== */
function Welcome({ onStart }) {
  return (
    <div className="welcome">
      <div className="welcome-title">Map a city's trees in minutes</div>
      <div className="welcome-sub">
        Draw a rectangle anywhere in Astana. The system fetches satellite imagery
        at maximum resolution, splits big areas into a grid, and detects every
        individual tree crown automatically.
      </div>
      <button className="btn btn-primary" onClick={onStart}>
        <Icon name="grid" size={14} />
        <span>Scan an area</span>
      </button>
    </div>
  );
}

/* ==================================================================
   Drawer (scans + snapshots history)
   ================================================================== */
function HistoryDrawer({ open, onClose, scans, snapshots, onDeleteScan, onDeleteSnapshot, loading }) {
  if (!open) return null;
  return (
    <>
      <div className="drawer-overlay" onClick={onClose} />
      <div className="drawer" role="dialog">
        <div className="drawer-head">
          <Icon name="history" size={16} />
          <div className="drawer-title">History</div>
          <button className="icon-btn" onClick={onClose}><Icon name="x" size={14} /></button>
        </div>
        <div className="drawer-body">
          <div className="section">
            <div className="section-title">Area scans · {scans.length}</div>
            {loading && <div className="list-empty">Loading…</div>}
            {!loading && scans.length === 0 && (
              <div className="list-empty">No scans yet. Close this panel and click <b>Scan area</b> to start.</div>
            )}
            {scans.map((s) => {
              const date = s.created_at ? new Date(s.created_at) : null;
              return (
                <div key={s.id} className="list-item">
                  <div className="list-item-row">
                    <div className="list-item-name">
                      Scan <span className="list-item-mono">{s.id.slice(0, 8)}</span>
                      {s.polygon_json && <span className="tag gray" style={{ marginLeft: 6 }}>polygon</span>}
                      {s.status === "running" && <span className="tag running" style={{ marginLeft: 6 }}>running</span>}
                    </div>
                  </div>
                  <div className="list-item-meta">
                    <span><b>{(s.total_trees || 0).toLocaleString()}</b> trees</span>
                    <span><b>{s.ok_count}</b>/{s.sub_count} ok</span>
                    <span>{s.provider} · z{s.zoom}</span>
                    <span className="muted">{s.model}</span>
                  </div>
                  <div className="list-item-coords">
                    {s.nw_lat.toFixed(5)}°, {s.nw_lng.toFixed(5)}° → {s.se_lat.toFixed(5)}°, {s.se_lng.toFixed(5)}°
                  </div>
                  {date && <div className="list-item-coords muted">{date.toLocaleString()}</div>}
                  <div className="list-item-actions">
                    <button className="ghost-btn danger" onClick={() => onDeleteScan(s.id, s.sub_count)}>
                      <Icon name="trash" size={11} />
                      <span>Delete scan</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="section">
            <div className="section-title">Snapshots · {snapshots.length}</div>
            {!loading && snapshots.length === 0 && (
              <div className="list-empty">No individual snapshots.</div>
            )}
            {snapshots.map((s) => (
              <div key={s.image_id} className="list-item">
                <div className="list-item-row">
                  <div className="list-item-name" title={s.filename}>{s.filename}</div>
                </div>
                <div className="list-item-meta">
                  <span><b>{s.total_trees || 0}</b> trees</span>
                  <span><b>{s.run_count}</b> run{s.run_count === 1 ? "" : "s"}</span>
                  <span>{s.width}×{s.height}</span>
                  {s.last_model && <span className="muted">{s.last_model}</span>}
                </div>
                {s.nw_lat != null && (
                  <div className="list-item-coords">
                    N {s.nw_lat.toFixed(5)}° → S {s.se_lat.toFixed(5)}°
                  </div>
                )}
                <div className="list-item-actions">
                  <button className="ghost-btn danger" onClick={() => onDeleteSnapshot(s.image_id)}>
                    <Icon name="trash" size={11} />
                    <span>Delete</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

/* ==================================================================
   Settings popover (top-right, accessed via gear)
   ================================================================== */
function SettingsPopover({ open, tileProvider, setTileProvider, providersMap, modelStatus, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    setTimeout(() => document.addEventListener("mousedown", onDocClick), 50);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open, onClose]);
  if (!open) return null;

  const models = modelStatus?.models || {};
  return (
    <div className="settings-popover popover" ref={ref}>
      <div className="popover-row">
        <div className="popover-label">Satellite imagery</div>
        <select
          className="select-full"
          value={tileProvider}
          onChange={(e) => setTileProvider(e.target.value)}
        >
          {providersMap
            ? Object.entries(providersMap).map(([k, cfg]) => <option key={k} value={k}>{cfg.label}</option>)
            : (<>
                <option value="google">Google Satellite</option>
                <option value="esri">Esri World Imagery</option>
              </>)}
        </select>
        <div className="field-help">Source for both map display and scan capture.</div>
      </div>
      <div className="popover-row">
        <div className="popover-label">Models loaded</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {Object.entries(models).map(([k, m]) => (
            <div key={k} className="row" style={{ fontSize: 11 }}>
              <span className="status-dot" style={{ background: m.available ? "var(--success)" : "var(--text-3)" }}></span>
              <span style={{ flex: 1 }}>{m.name || k}</span>
              <span className="muted mono" style={{ fontSize: 10 }}>
                {m.available ? (m.loaded ? "ready" : "lazy") : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ==================================================================
   Image view sidebar (progressive disclosure)
   ================================================================== */
function ImageSidebar({
  image, uploading, onUpload, onClear, uploadError,
  geo, setGeo, status, progress, eta, onRun, onCancelRun,
  trees, stats, model, setModel, modelStatus, threshold, setThreshold,
  filter, setFilter, displayMode, setDisplayMode,
  jobId, onExport, predictError,
  showOverlay, setShowOverlay, overlayOpacity, setOverlayOpacity,
}) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const hasImage = !!image;
  const hasResults = !!trees && trees.length > 0;
  const hasMasks = hasResults && trees.some((t) => t.mask_polygon_geo && t.mask_polygon_geo.length >= 3);

  return (
    <aside className="sidebar">
      <div className="sidebar-scroll">
        {/* SECTION 1: Image */}
        <div className="section">
          <div className="section-title">1 · Image</div>
          {!hasImage ? (
            <div
              className={`upload-zone ${drag ? "dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); onUpload(e.dataTransfer.files[0]); }}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.webp"
                style={{ display: "none" }}
                onChange={(e) => onUpload(e.target.files[0])}
              />
              <div className="upload-icon"><Icon name={uploading ? "refresh" : "upload"} size={22} /></div>
              <div className="upload-headline">{uploading ? "Uploading…" : "Drop image or click"}</div>
              <div className="upload-sub">PNG · JPG · TIFF · GeoTIFF · max 100 MB</div>
            </div>
          ) : (
            <div className="image-preview">
              {image.url && <img className="image-preview-img" src={image.url} alt={image.name} />}
              <div className="image-preview-meta">
                <div className="image-preview-name">{image.name || image.filename}</div>
                <div className="image-preview-stats">
                  {image.width}×{image.height}
                  {image.is_geotiff && <> · GeoTIFF</>}
                  {image.crs && <> · {image.crs}</>}
                </div>
                <button className="image-preview-clear" onClick={onClear}>
                  <Icon name="x" size={11} /><span>Remove</span>
                </button>
              </div>
            </div>
          )}
          {uploadError && (
            <div className="field-help" style={{ color: "var(--danger)", marginTop: 8 }}>
              <Icon name="alert" size={11} /> {uploadError}
            </div>
          )}
        </div>

        {/* SECTION 2: Georeferencing — only if image without auto-geo */}
        {hasImage && !image.is_geotiff && (
          <div className="section">
            <div className="section-title">2 · Georeferencing</div>
            <div className="field">
              <label className="field-label">Mode</label>
              <select
                className="select-full"
                value={geo.mode}
                onChange={(e) => setGeo({ ...geo, mode: e.target.value })}
              >
                <option value="none">None (pixel coords only)</option>
                <option value="corners_2">Two corners (NW + SE)</option>
                <option value="corners_4">Four corners (handles rotation)</option>
              </select>
              <div className="field-help">
                {geo.mode === "none" && "Detections will have pixel coordinates only — no map placement."}
                {geo.mode === "corners_2" && "Drag the NW and SE markers on the map to position the image."}
                {geo.mode === "corners_4" && "Four corners — for tilted screenshots from Google Earth Pro."}
              </div>
            </div>
            {geo.mode === "corners_2" && image.bounds && (
              <div className="conf-preview" style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
                <div className="mono" style={{ fontSize: 10.5 }}>
                  NW {image.bounds.nw.lat.toFixed(5)}°, {image.bounds.nw.lng.toFixed(5)}°
                </div>
                <div className="mono" style={{ fontSize: 10.5 }}>
                  SE {image.bounds.se.lat.toFixed(5)}°, {image.bounds.se.lng.toFixed(5)}°
                </div>
              </div>
            )}
          </div>
        )}
        {hasImage && image.is_geotiff && (
          <div className="section">
            <div className="section-title">2 · Georeferencing</div>
            <div className="conf-preview">
              <span className="status-dot ok"></span>
              <span>GeoTIFF — auto from file metadata ({image.crs || "EPSG:4326"})</span>
            </div>
          </div>
        )}

        {/* SECTION 3: Detection */}
        {hasImage && (
          <div className="section">
            <div className="section-title">3 · Detection</div>
            <div className="field">
              <label className="field-label">Model</label>
              <select
                className="select-full"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {Object.entries(modelStatus?.models || {}).map(([k, m]) => (
                  <option key={k} value={k} disabled={!m.available}>
                    {m.name || k}{m.available ? "" : " (unavailable)"}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label className="field-label" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>Confidence threshold</span>
                <span className="mono">{Math.round(threshold * 100)}%</span>
              </label>
              <input
                type="range" min={0.05} max={0.95} step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="range"
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={onRun}
              disabled={status === "running"}
            >
              {status === "running" ? (
                <>
                  <div className="scan-spinner" style={{ width: 12, height: 12 }} />
                  <span>Detecting… {progress > 0 ? `${progress}%` : ""}</span>
                </>
              ) : (
                <>
                  <Icon name="play" size={12} />
                  <span>Run detection</span>
                </>
              )}
            </button>
            {predictError && (
              <div className="field-help" style={{ color: "var(--danger)", marginTop: 8 }}>
                <Icon name="alert" size={11} /> {predictError}
              </div>
            )}
          </div>
        )}

        {/* SECTION 4: Results */}
        {hasResults && (
          <div className="section">
            <div className="section-title">4 · Results</div>
            <div className="stats-grid">
              <div className="stat-tile">
                <div className="stat-tile-v">{stats?.tree_count?.toLocaleString() ?? trees.length}</div>
                <div className="stat-tile-k">trees</div>
              </div>
              {stats?.avg_confidence != null && (
                <div className="stat-tile">
                  <div className="stat-tile-v">{Math.round(stats.avg_confidence * 100)}%</div>
                  <div className="stat-tile-k">avg conf</div>
                </div>
              )}
              {stats?.coverage_pct != null && (
                <div className="stat-tile">
                  <div className="stat-tile-v">{stats.coverage_pct.toFixed(1)}%</div>
                  <div className="stat-tile-k">canopy</div>
                </div>
              )}
              {stats?.avg_crown_area_m2 != null && (
                <div className="stat-tile">
                  <div className="stat-tile-v">{stats.avg_crown_area_m2.toFixed(1)}<span className="muted" style={{ fontSize: 11 }}> m²</span></div>
                  <div className="stat-tile-k">avg crown</div>
                </div>
              )}
            </div>

            <div style={{ marginTop: 12 }}>
              <div className="field-label">Display</div>
              <div className="control-group" style={{ background: "var(--surface-2)", boxShadow: "none", borderColor: "var(--border)", marginTop: 4 }}>
                {["point", "bbox", hasMasks && "polygon", "heat"].filter(Boolean).map((m) => (
                  <button
                    key={m}
                    className={`seg-btn ${displayMode === m ? "active" : ""}`}
                    onClick={() => setDisplayMode(m)}
                  >
                    {m[0].toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="field" style={{ marginTop: 12 }}>
              <label className="field-label">Image overlay</label>
              <div className="row">
                <input
                  type="checkbox"
                  checked={showOverlay}
                  onChange={(e) => setShowOverlay(e.target.checked)}
                  id="overlay-toggle"
                />
                <label htmlFor="overlay-toggle" style={{ fontSize: 12, cursor: "pointer" }}>
                  Show source image on map
                </label>
              </div>
              {showOverlay && (
                <div style={{ marginTop: 8 }}>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                    className="range"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* SECTION 5: Export — only after results */}
        {hasResults && jobId && (
          <div className="section">
            <div className="section-title">5 · Export</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <button className="btn btn-secondary" onClick={() => onExport("geojson")}>
                <Icon name="download" size={12} />
                <span>GeoJSON</span>
              </button>
              <button className="btn btn-secondary" onClick={() => onExport("csv")}>
                <Icon name="download" size={12} />
                <span>CSV</span>
              </button>
              <button className="btn btn-secondary" onClick={() => onExport("html")}>
                <Icon name="download" size={12} />
                <span>Standalone HTML map</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

/* ==================================================================
   MapHost — Leaflet container + all overlays
   This is the workhorse — handles the actual map rendering,
   tree markers, scan progress overlay, draw modes.
   ================================================================== */
function MapHost({
  view,
  // Map data
  trees, threshold, filter, displayMode, markerSize = 6,
  // Provider / base
  tileProvider, providersMap,
  // Scan flow
  scanMode, onScanBbox, polygonMode, onPolygonComplete, pendingPolygon, onPolygonReset,
  scanProgress,
  // Image flow
  image, imageBounds, geo, setGeo, showOverlay, overlayOpacity,
  // Tree click handler
  onTreeClick,
}) {
  const mapRef = useRef(null);
  const map = useRef(null);
  const tileLayerRef = useRef(null);
  const layerRef = useRef(null);
  const heatLayerRef = useRef(null);
  const overlayRef = useRef(null);
  const cornersLayerRef = useRef(null);
  const nwMarkerRef = useRef(null);
  const seMarkerRef = useRef(null);
  const rectRef = useRef(null);
  const captureRectRef = useRef(null);
  const scanGridRef = useRef(null);
  const scanTreesRef = useRef(null);
  const scanTreesRenderedRef = useRef(0);
  const scanCanvasRef = useRef(null);

  // ----- mount -----
  useEffect(() => {
    if (map.current || !mapRef.current) return;
    const m = L.map(mapRef.current, { zoomControl: false, attributionControl: false }).setView(ASTANA_CENTER, ASTANA_ZOOM);
    map.current = m;
    tileLayerRef.current = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19 }
    ).addTo(m);
    L.control.zoom({ position: "topright" }).addTo(m);
    L.control.scale({ position: "bottomleft", imperial: false }).addTo(m);
    layerRef.current = L.layerGroup().addTo(m);
    cornersLayerRef.current = L.layerGroup().addTo(m);
    scanGridRef.current = L.layerGroup().addTo(m);
    scanTreesRef.current = L.layerGroup().addTo(m);
    scanCanvasRef.current = L.canvas({ padding: 0.5 });
    return () => { m.remove(); map.current = null; };
  }, []);

  // ----- base layer (provider-aware) -----
  useEffect(() => {
    if (!map.current) return;
    const cfg = providersMap && providersMap[tileProvider];
    const url = cfg ? cfg.url : "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    const opts = cfg
      ? { maxZoom: cfg.max_zoom || 19, subdomains: cfg.subdomains || "abc" }
      : { maxZoom: 19 };
    if (tileLayerRef.current) map.current.removeLayer(tileLayerRef.current);
    tileLayerRef.current = L.tileLayer(url, opts).addTo(map.current);
  }, [tileProvider, providersMap]);

  // ----- tree rendering (single image or aggregate, NOT scan progress) -----
  useEffect(() => {
    if (!layerRef.current || !map.current) return;
    layerRef.current.clearLayers();
    if (heatLayerRef.current) {
      map.current.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }
    if (!trees || trees.length === 0) return;

    const visible = trees.filter((t) => {
      if (t.confidence < threshold) return false;
      if (t.confidence > 0.7) return filter.high;
      if (t.confidence > 0.5) return filter.med;
      return filter.low;
    });

    if (displayMode === "heat") {
      if (typeof L.heatLayer !== "function") {
        console.warn("L.heatLayer plugin not loaded; falling back to points");
      } else {
        const pts = visible
          .filter((t) => t.lat != null && t.lng != null)
          .map((t) => [t.lat, t.lng, Math.max(0.1, t.confidence || 0.5)]);
        if (pts.length) {
          heatLayerRef.current = L.heatLayer(pts, {
            radius: 24, blur: 28, maxZoom: 20, minOpacity: 0.4,
            gradient: { 0.2: "#EA9F27", 0.5: "#5DCAA5", 0.8: "#0F6E56", 1.0: "#0A3F30" },
          }).addTo(map.current);
        }
        return;
      }
    }

    const popupHtml = (t, color) => (
      `<div>
         <div class="tree-popup-head">
           <span class="tree-popup-id">#${String(t.id).padStart(3, "0")}</span>
           <span class="tree-popup-conf" style="background:${color}">${Math.round(t.confidence * 100)}%</span>
         </div>
         <div class="tree-popup-grid">
           <span class="k">Lat</span><span class="v">${t.lat.toFixed(6)}°</span>
           <span class="k">Lng</span><span class="v">${t.lng.toFixed(6)}°</span>
           <span class="k">Crown</span><span class="v">${t.crown ? t.crown.toFixed(1) + " m" : "—"}</span>
         </div>
       </div>`
    );

    visible.forEach((t) => {
      const color = t.confidence > 0.7 ? "#0F6E56" : t.confidence > 0.5 ? "#5DCAA5" : "#EA9F27";
      let g = null;
      if (displayMode === "polygon" && t.mask_polygon_geo && t.mask_polygon_geo.length >= 3) {
        g = L.polygon(t.mask_polygon_geo, { color, weight: 1.2, fillColor: color, fillOpacity: 0.28 });
      } else if (displayMode === "bbox" && t.box_geo && t.box_geo.length === 4) {
        g = L.polygon(t.box_geo, { color, weight: 1.4, fillColor: color, fillOpacity: 0.12 });
      } else if (displayMode === "point") {
        g = L.circleMarker([t.lat, t.lng], {
          radius: markerSize, fillColor: color, color: "#fff", weight: 1.5, fillOpacity: 0.95,
        });
      }
      if (!g) {
        g = L.circleMarker([t.lat, t.lng], {
          radius: 4, fillColor: color, color: "#fff", weight: 1, fillOpacity: 0.85,
        });
      }
      g.bindPopup(popupHtml(t, color), { className: "", closeButton: false });
      g.on("click", () => onTreeClick && onTreeClick(t));
      layerRef.current.addLayer(g);
    });
  }, [trees, threshold, filter, displayMode, markerSize]);

  // ----- single-image overlay (only in image view) -----
  useEffect(() => {
    if (!map.current) return;
    if (overlayRef.current) { map.current.removeLayer(overlayRef.current); overlayRef.current = null; }
    if (view === "image" && showOverlay && image && image.url && imageBounds) {
      overlayRef.current = L.imageOverlay(
        image.url,
        [[imageBounds.nw.lat, imageBounds.nw.lng], [imageBounds.se.lat, imageBounds.se.lng]],
        { opacity: overlayOpacity }
      ).addTo(map.current);
    }
  }, [view, showOverlay, image && image.url, imageBounds, overlayOpacity]);

  // ----- corners markers for corners_2 geo mode -----
  useEffect(() => {
    if (!map.current || !cornersLayerRef.current) return;
    cornersLayerRef.current.clearLayers();
    if (rectRef.current) { map.current.removeLayer(rectRef.current); rectRef.current = null; }
    if (view !== "image" || !image || image.is_geotiff || geo.mode !== "corners_2" || !geo.corners_2) return;

    const bounds = geo.corners_2;
    const onDrag = (which) => (e) => {
      const ll = e.target.getLatLng();
      const next = { ...bounds, [which]: { lat: ll.lat, lng: ll.lng } };
      setGeo({ ...geo, corners_2: next });
    };
    nwMarkerRef.current = L.marker([bounds.nw.lat, bounds.nw.lng], { draggable: true })
      .on("drag", onDrag("nw"))
      .addTo(cornersLayerRef.current);
    seMarkerRef.current = L.marker([bounds.se.lat, bounds.se.lng], { draggable: true })
      .on("drag", onDrag("se"))
      .addTo(cornersLayerRef.current);
    rectRef.current = L.rectangle(
      [[bounds.nw.lat, bounds.nw.lng], [bounds.se.lat, bounds.se.lng]],
      { color: "#0F6E56", weight: 1.5, dashArray: "4 4", fillOpacity: 0, interactive: false },
    ).addTo(cornersLayerRef.current);
  }, [view, image && image.image_id, image && image.is_geotiff, geo]);

  // ----- rectangle drawing for scan mode -----
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const c = m.getContainer();
    if (!scanMode) {
      c.style.cursor = "";
      if (captureRectRef.current) { m.removeLayer(captureRectRef.current); captureRectRef.current = null; }
      return;
    }
    c.style.cursor = "crosshair";
    let start = null;
    const onDown = (e) => {
      start = e.latlng;
      m.dragging.disable();
      if (captureRectRef.current) m.removeLayer(captureRectRef.current);
      captureRectRef.current = L.rectangle([start, start], {
        color: "#0F6E56", weight: 2, fillColor: "#0F6E56", fillOpacity: 0.12, dashArray: "4 4",
      }).addTo(m);
    };
    const onMove = (e) => {
      if (!start || !captureRectRef.current) return;
      captureRectRef.current.setBounds([start, e.latlng]);
    };
    const onUp = (e) => {
      if (!start) return;
      m.dragging.enable();
      const a = start; const b = e.latlng; start = null;
      const nw = { lat: Math.max(a.lat, b.lat), lng: Math.min(a.lng, b.lng) };
      const se = { lat: Math.min(a.lat, b.lat), lng: Math.max(a.lng, b.lng) };
      if (Math.abs(nw.lat - se.lat) < 0.0003 || Math.abs(se.lng - nw.lng) < 0.0003) {
        if (captureRectRef.current) m.removeLayer(captureRectRef.current);
        return;
      }
      onScanBbox && onScanBbox({ nw, se });
    };
    m.on("mousedown", onDown);
    m.on("mousemove", onMove);
    m.on("mouseup", onUp);
    return () => {
      m.off("mousedown", onDown);
      m.off("mousemove", onMove);
      m.off("mouseup", onUp);
      m.dragging.enable();
    };
  }, [scanMode, onScanBbox]);

  // ----- polygon drawing -----
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const c = m.getContainer();
    if (!polygonMode) { c.style.cursor = ""; return; }
    c.style.cursor = "crosshair";

    let vertices = (pendingPolygon && pendingPolygon.length >= 3) ? pendingPolygon.slice() : [];
    let preview = null;
    let dots = [];
    const COLOR = "#0F6E56";

    const draw = () => {
      if (preview) { m.removeLayer(preview); preview = null; }
      dots.forEach((d) => m.removeLayer(d));
      dots = [];
      vertices.forEach((v) => {
        const dot = L.circleMarker(v, {
          radius: 5, color: "#fff", weight: 2, fillColor: COLOR, fillOpacity: 1, interactive: false,
        }).addTo(m);
        dots.push(dot);
      });
      if (vertices.length === 2) {
        preview = L.polyline(vertices, { color: COLOR, weight: 2.5, dashArray: "4 4", interactive: false }).addTo(m);
      } else if (vertices.length >= 3) {
        preview = L.polygon(vertices, {
          color: COLOR, weight: 2.5, fillColor: COLOR, fillOpacity: 0.12, dashArray: "4 4", interactive: false,
        }).addTo(m);
      }
    };
    draw();

    const onClick = (e) => {
      if (pendingPolygon && pendingPolygon.length >= 3) return;
      vertices.push([e.latlng.lat, e.latlng.lng]);
      draw();
    };
    const onDblClick = (e) => {
      if (vertices.length < 3) return;
      L.DomEvent.stopPropagation(e);
      onPolygonComplete && onPolygonComplete(vertices.slice());
    };
    const onContext = (e) => {
      L.DomEvent.preventDefault(e); L.DomEvent.stopPropagation(e);
      vertices = []; draw();
      onPolygonReset && onPolygonReset();
    };
    m.on("click", onClick);
    m.on("dblclick", onDblClick);
    m.on("contextmenu", onContext);
    m.doubleClickZoom.disable();
    return () => {
      m.off("click", onClick);
      m.off("dblclick", onDblClick);
      m.off("contextmenu", onContext);
      m.doubleClickZoom.enable();
      if (preview) m.removeLayer(preview);
      dots.forEach((d) => m.removeLayer(d));
    };
  }, [polygonMode, onPolygonComplete, onPolygonReset, pendingPolygon]);

  // ----- scan progress: grid + polygon preview -----
  useEffect(() => {
    if (!scanGridRef.current || !map.current) return;
    scanGridRef.current.clearLayers();
    if (!scanProgress || !scanProgress.regions || scanProgress.regions.length === 0) {
      if (!scanProgress) {
        if (scanTreesRef.current) scanTreesRef.current.clearLayers();
        scanTreesRenderedRef.current = 0;
      }
      return;
    }
    const STATUS = {
      pending:    { color: "#A8A29E", weight: 1.5, fillOpacity: 0.05, dashArray: "4 4", className: "" },
      capturing:  { color: "#2563EB", weight: 2.5, fillOpacity: 0.10, dashArray: null,  className: "scan-region-pulse" },
      captured:   { color: "#2563EB", weight: 2,   fillOpacity: 0.08, dashArray: null,  className: "" },
      predicting: { color: "#EA580C", weight: 2.5, fillOpacity: 0.10, dashArray: null,  className: "scan-region-pulse" },
      done:       { color: "#0F6E56", weight: 2,   fillOpacity: 0.15, dashArray: null,  className: "" },
      error:      { color: "#DC2626", weight: 2,   fillOpacity: 0.10, dashArray: "2 3", className: "" },
    };
    if (scanProgress.polygon && scanProgress.polygon.length >= 3) {
      L.polygon(scanProgress.polygon.map((p) => [p.lat, p.lng]), {
        color: "#0F6E56", weight: 2.5, fillColor: "#0F6E56", fillOpacity: 0.03, dashArray: "6 3", interactive: false,
      }).addTo(scanGridRef.current);
    }
    scanProgress.regions.forEach((r) => {
      const s = STATUS[r.status] || STATUS.pending;
      const nw = r.sub_bbox.nw, se = r.sub_bbox.se;
      L.rectangle([[nw.lat, nw.lng], [se.lat, se.lng]], {
        color: s.color, weight: s.weight, fillColor: s.color, fillOpacity: s.fillOpacity,
        dashArray: s.dashArray || undefined, className: s.className, interactive: false,
      }).addTo(scanGridRef.current);
      let label = `r${r.row}c${r.col}`;
      if (r.status === "done") label += ` · ${r.tree_count} trees`;
      else if (r.status === "error") label += " · err";
      else label += ` · ${r.status}`;
      const icon = L.divIcon({
        className: "scan-region-label",
        html: `<span>${label}</span>`,
        iconSize: null, iconAnchor: [-4, -4],
      });
      L.marker([nw.lat, nw.lng], { icon, interactive: false }).addTo(scanGridRef.current);
    });
    // Auto-zoom only at start
    if (scanProgress.regions.length && scanTreesRenderedRef.current === 0) {
      const all = scanProgress.regions.flatMap((r) => [
        [r.sub_bbox.nw.lat, r.sub_bbox.nw.lng],
        [r.sub_bbox.se.lat, r.sub_bbox.se.lng],
      ]);
      try { map.current.fitBounds(all, { padding: [80, 80], maxZoom: 17 }); } catch {}
    }
  }, [scanProgress]);

  // ----- scan progress: differential tree append -----
  useEffect(() => {
    if (!scanTreesRef.current) return;
    const treesArr = (scanProgress && scanProgress.trees) || [];
    if (treesArr.length < scanTreesRenderedRef.current) {
      scanTreesRef.current.clearLayers();
      scanTreesRenderedRef.current = 0;
    }
    for (let i = scanTreesRenderedRef.current; i < treesArr.length; i++) {
      const t = treesArr[i];
      if (t.lat == null || t.lng == null) continue;
      const conf = t.confidence ?? 0;
      const color = conf > 0.7 ? "#0F6E56" : conf > 0.5 ? "#5DCAA5" : "#EA9F27";
      L.circleMarker([t.lat, t.lng], {
        radius: 4, color: "#fff", weight: 1, fillColor: color, fillOpacity: 0.9,
        interactive: false, renderer: scanCanvasRef.current,
      }).addTo(scanTreesRef.current);
    }
    scanTreesRenderedRef.current = treesArr.length;
  }, [scanProgress && scanProgress.trees && scanProgress.trees.length]);

  return <div className="map-host" ref={mapRef} />;
}

/* ==================================================================
   Toast
   ================================================================== */
function Toast({ msg, kind }) {
  if (!msg) return null;
  return (
    <div className={`toast ${kind || ""}`}>
      <Icon name={kind === "error" ? "alert" : "check"} size={12} />
      <span>{msg}</span>
    </div>
  );
}

/* ==================================================================
   Legend (bottom-right of map when trees visible)
   ================================================================== */
function MapLegend({ trees, threshold, filter }) {
  if (!trees || trees.length === 0) return null;
  const visible = trees.filter((t) => {
    if (t.confidence < threshold) return false;
    if (t.confidence > 0.7) return filter.high;
    if (t.confidence > 0.5) return filter.med;
    return filter.low;
  });
  return null;  // legend is integrated into stats card + filter popover now
}

/* ==================================================================
   MAIN APP
   ================================================================== */
function App() {
  // ---- view ----
  const [view, setView] = useState("map");
  const [dark, setDark] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // ---- providers ----
  const [tileProvider, setTileProvider] = useState("google");
  const [providersMap, setProvidersMap] = useState(null);

  // ---- backend status / models ----
  const [modelStatus, setModelStatus] = useState(null);
  const backendStatus = modelStatus?._error
    ? "down"
    : (modelStatus && Object.values(modelStatus.models || {}).some((m) => m.available) ? "ok" : "warn");

  // ---- map view: aggregate ----
  const [aggregateStats, setAggregateStats] = useState({ snapshot_count: 0, run_count: 0, total_trees: 0 });
  const [aggregateTrees, setAggregateTrees] = useState([]);
  const [snapshots, setSnapshots] = useState([]);
  const [scans, setScans] = useState([]);
  const [aggLoading, setAggLoading] = useState(false);

  // ---- scan flow ----
  const [scanMode, setScanMode] = useState(false);
  const [polygonMode, setPolygonMode] = useState(false);
  const [scanRunning, setScanRunning] = useState(false);
  const [scanProgress, setScanProgress] = useState(null);
  const [pendingPolygon, setPendingPolygon] = useState(null);
  const scanAbortRef = useRef(null);

  // ---- model + threshold (shared) ----
  const [model, setModel] = useState("yolo");
  const [threshold, setThreshold] = useState(0.25);
  const [filter, setFilter] = useState({ high: true, med: true, low: true });
  const [displayMode, setDisplayMode] = useState("point");

  // ---- image view ----
  const [image, setImage] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [imageTrees, setImageTrees] = useState(null);
  const [imageStats, setImageStats] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [predictStatus, setPredictStatus] = useState("idle");
  const [predictProgress, setPredictProgress] = useState(0);
  const [predictEta, setPredictEta] = useState(null);
  const [geo, setGeo] = useState({
    mode: "corners_2",
    corners_2: { nw: { lat: 51.17, lng: 71.46 }, se: { lat: 51.15, lng: 71.49 } },
  });
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.85);

  // ---- ui ----
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [toastKind, setToastKind] = useState(null);

  // -------- effects --------
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  useEffect(() => {
    window.api.providers().then((r) => setProvidersMap(r.providers)).catch(() => {});
  }, []);

  useEffect(() => {
    window.api.status()
      .then(setModelStatus)
      .catch((e) => setModelStatus({ models: {}, _error: e?.message || "down" }));
  }, []);

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

  useEffect(() => { refreshAggregate(); }, [refreshAggregate]);

  const showToast = useCallback((m, k = "info") => {
    setToast(m); setToastKind(k);
    setTimeout(() => setToast(null), 2500);
  }, []);

  // -------- scan flow --------
  const startScanRect = useCallback(() => {
    setPolygonMode(false); setPendingPolygon(null); setScanMode(true);
  }, []);
  const startScanPoly = useCallback(() => {
    setScanMode(false); setPendingPolygon(null); setPolygonMode(true);
  }, []);
  const cancelDraw = useCallback(() => {
    setScanMode(false); setPolygonMode(false); setPendingPolygon(null);
  }, []);

  const runScan = useCallback(async (bbox, polygon = null) => {
    setScanMode(false); setPolygonMode(false);
    setScanRunning(true);
    setScanProgress({ regions: [], trees: [], done: false, polygon });
    const abort = new AbortController();
    scanAbortRef.current = abort;
    try {
      await window.api.scanRegionStream(
        {
          nw: bbox.nw, se: bbox.se, zoom: 19,
          model, confidence: threshold, maxSubregions: 9,
          provider: tileProvider, polygon: polygon || undefined,
        },
        (ev) => {
          if (ev.type === "plan") {
            setScanProgress((p) => ({
              ...p,
              regions: ev.sub_regions.map((r) => ({
                row: r.row, col: r.col, sub_bbox: { nw: r.nw, se: r.se },
                status: "pending", tree_count: 0, error: null,
              })),
            }));
          } else if (ev.type === "capturing" || ev.type === "capture_done" || ev.type === "predicting") {
            const s = ev.type === "capturing" ? "capturing"
                    : ev.type === "capture_done" ? "captured" : "predicting";
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col ? { ...r, status: s } : r
              ),
            }));
          } else if (ev.type === "sub_complete") {
            const adapted = window.api.adaptAggregateDetections(
              ev.detections.map((d) => ({ ...d, local_id: d.id, model: ev.model, image_id: ev.snapshot_id, job_id: ev.job_id }))
            );
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col
                  ? { ...r, status: "done", tree_count: ev.tree_count } : r
              ),
              trees: [...p.trees, ...adapted],
            }));
          } else if (ev.type === "sub_error") {
            setScanProgress((p) => ({
              ...p,
              regions: p.regions.map((r) =>
                r.row === ev.row && r.col === ev.col
                  ? { ...r, status: "error", error: ev.error } : r
              ),
            }));
          } else if (ev.type === "done") {
            setScanProgress((p) => ({ ...p, done: true }));
            showToast(`Scan complete · ${ev.total_trees.toLocaleString()} trees · ${(ev.duration_ms / 1000).toFixed(1)}s`, "success");
          } else if (ev.type === "fatal") {
            showToast("Scan crashed: " + ev.error, "error");
          }
        },
        { signal: abort.signal },
      );
      await refreshAggregate();
    } catch (e) {
      if (e.name !== "AbortError") {
        showToast("Scan failed: " + e.message, "error");
      }
    } finally {
      setScanRunning(false);
      scanAbortRef.current = null;
      setTimeout(() => setScanProgress(null), 4000);
    }
  }, [model, threshold, tileProvider, refreshAggregate, showToast]);

  const cancelScan = useCallback(() => {
    if (scanAbortRef.current) scanAbortRef.current.abort();
  }, []);

  const handlePolygonComplete = useCallback((poly) => {
    setPendingPolygon(poly);
  }, []);
  const handleStartPolygonScan = useCallback(() => {
    if (!pendingPolygon || pendingPolygon.length < 3) return;
    const lats = pendingPolygon.map((p) => p[0]);
    const lngs = pendingPolygon.map((p) => p[1]);
    const bbox = {
      nw: { lat: Math.max(...lats), lng: Math.min(...lngs) },
      se: { lat: Math.min(...lats), lng: Math.max(...lngs) },
    };
    const polyPoints = pendingPolygon.map(([lat, lng]) => ({ lat, lng }));
    setPendingPolygon(null);
    runScan(bbox, polyPoints);
  }, [pendingPolygon, runScan]);
  const handleClearPolygon = useCallback(() => setPendingPolygon(null), []);

  // -------- scan / snapshot delete --------
  const handleDeleteScan = useCallback(async (scanId, subCount) => {
    if (!confirm(`Delete scan ${scanId.slice(0, 8)} and all ${subCount || "its"} sub-snapshots?`)) return;
    try { await window.api.deleteScan(scanId); await refreshAggregate(); }
    catch (e) { showToast("Delete failed: " + e.message, "error"); }
  }, [refreshAggregate, showToast]);
  const handleDeleteSnapshot = useCallback(async (imageId) => {
    if (!confirm(`Delete snapshot ${imageId}?`)) return;
    try { await window.api.deleteSnapshot(imageId); await refreshAggregate(); }
    catch (e) { showToast("Delete failed: " + e.message, "error"); }
  }, [refreshAggregate, showToast]);

  // -------- image flow --------
  const handleImageUpload = useCallback(async (file) => {
    if (!file) return;
    setUploadError(null); setUploading(true);
    try {
      const meta = await window.api.upload(file);
      setImage({ ...meta, name: meta.filename, url: window.api.imageUrl(meta.image_id) });
      setImageId(meta.image_id);
      setImageTrees(null); setImageStats(null); setJobId(null); setPredictStatus("idle");
      if (meta.is_geotiff) setGeo({ ...geo, mode: "geotiff" });
      else if (!geo.corners_2) {
        setGeo({ mode: "corners_2", corners_2: { nw: { lat: 51.17, lng: 71.46 }, se: { lat: 51.15, lng: 71.49 } } });
      }
      showToast(`Uploaded ${meta.filename}`, "success");
    } catch (e) {
      setUploadError(e.message);
      showToast("Upload failed: " + e.message, "error");
    } finally {
      setUploading(false);
    }
  }, [geo, showToast]);

  const handleImageClear = useCallback(() => {
    setImage(null); setImageId(null); setImageTrees(null); setImageStats(null);
    setJobId(null); setPredictStatus("idle"); setUploadError(null); setShowOverlay(false);
  }, []);

  const handleRunPredict = useCallback(async () => {
    if (!imageId) return;
    setPredictStatus("running"); setPredictProgress(0); setPredictEta(null);
    setImageTrees(null); setImageStats(null); setJobId(null); setPredictError(null);
    const t0 = Date.now();
    let p = 0;
    const interval = setInterval(() => {
      p = Math.min(90, p + 1.5 + Math.random() * 2);
      setPredictProgress(Math.floor(p));
      const elapsed = (Date.now() - t0) / 1000;
      setPredictEta(Math.max(0, (elapsed / Math.max(p, 1)) * 100 - elapsed));
    }, 180);
    try {
      const res = await window.api.predict({ image_id: imageId, model, confidence: threshold, geo });
      clearInterval(interval);
      setPredictProgress(100);
      const adapted = window.api.adaptDetectionsForUI(res.detections);
      setImageTrees(adapted);
      setImageStats(res.stats);
      setJobId(res.job_id);
      setPredictStatus("done");
      showToast(`Detected ${adapted.length} trees in ${res.duration_ms} ms`, "success");
    } catch (e) {
      clearInterval(interval);
      setPredictError(e.message);
      setPredictStatus("idle");
      showToast("Detection failed: " + e.message, "error");
    }
  }, [imageId, model, threshold, geo, showToast]);

  const handleExport = useCallback(async (fmt) => {
    if (!jobId) return;
    try { await window.api.exportFile(jobId, fmt); showToast(`${fmt.toUpperCase()} downloaded`, "success"); }
    catch (e) { showToast("Export failed: " + e.message, "error"); }
  }, [jobId, showToast]);

  // -------- derived map data --------
  const mapTrees = view === "map" ? aggregateTrees : imageTrees;
  const mapImage = view === "image" ? image : null;
  const mapImageBounds = view === "image" ? ((geo.mode === "corners_2" ? geo.corners_2 : null) || (image && image.bounds)) : null;
  const hasMasks = mapTrees && mapTrees.some((t) => t.mask_polygon_geo && t.mask_polygon_geo.length >= 3);
  const hasTrees = mapTrees && mapTrees.length > 0;

  const visibleCount = useMemo(() => {
    if (!mapTrees) return null;
    const visible = mapTrees.filter((t) => {
      if (t.confidence < threshold) return false;
      if (t.confidence > 0.7) return filter.high;
      if (t.confidence > 0.5) return filter.med;
      return filter.low;
    }).length;
    return { visible, total: mapTrees.length };
  }, [mapTrees, threshold, filter]);

  const isEmpty = view === "map" && aggregateStats.total_trees === 0 && !scanRunning && !scanMode && !polygonMode;

  // -------- render --------
  return (
    <div className="app">
      <TopBar
        view={view} setView={setView}
        onOpenSettings={() => setSettingsOpen(!settingsOpen)}
        settingsOpen={settingsOpen}
        backendStatus={backendStatus}
        dark={dark} setDark={setDark}
      />

      <SettingsPopover
        open={settingsOpen}
        tileProvider={tileProvider} setTileProvider={setTileProvider}
        providersMap={providersMap}
        modelStatus={modelStatus}
        onClose={() => setSettingsOpen(false)}
      />

      <div className={view === "image" ? "layout-image" : ""}>
        {view === "image" && (
          <ImageSidebar
            image={image} uploading={uploading} uploadError={uploadError}
            onUpload={handleImageUpload} onClear={handleImageClear}
            geo={geo} setGeo={setGeo}
            status={predictStatus} progress={predictProgress} eta={predictEta}
            onRun={handleRunPredict}
            trees={imageTrees} stats={imageStats}
            model={model} setModel={setModel} modelStatus={modelStatus}
            threshold={threshold} setThreshold={setThreshold}
            filter={filter} setFilter={setFilter}
            displayMode={displayMode} setDisplayMode={setDisplayMode}
            jobId={jobId} onExport={handleExport}
            predictError={predictError}
            showOverlay={showOverlay} setShowOverlay={setShowOverlay}
            overlayOpacity={overlayOpacity} setOverlayOpacity={setOverlayOpacity}
          />
        )}

        <div className="canvas">
          <MapHost
            view={view}
            trees={mapTrees} threshold={threshold} filter={filter} displayMode={displayMode}
            tileProvider={tileProvider} providersMap={providersMap}
            scanMode={scanMode} onScanBbox={(b) => runScan(b)}
            polygonMode={polygonMode}
            onPolygonComplete={handlePolygonComplete}
            pendingPolygon={pendingPolygon}
            onPolygonReset={handleClearPolygon}
            scanProgress={scanProgress}
            image={mapImage} imageBounds={mapImageBounds}
            geo={geo} setGeo={setGeo}
            showOverlay={showOverlay} overlayOpacity={overlayOpacity}
          />

          {view === "map" && (
            <>
              {isEmpty && <Welcome onStart={startScanRect} />}

              {!isEmpty && <StatsCard stats={aggregateStats} onOpenDrawer={() => setDrawerOpen(true)} />}

              {!scanRunning && (
                <ScanActionStack
                  scanMode={scanMode} polygonMode={polygonMode}
                  onStartScan={startScanRect} onStartPolygon={startScanPoly}
                  onCancel={cancelDraw}
                  pendingPolygon={pendingPolygon}
                  onStartPolygonScan={handleStartPolygonScan}
                  onClearPolygon={handleClearPolygon}
                />
              )}

              {hasTrees && (
                <DisplayStrip
                  displayMode={displayMode} setDisplayMode={setDisplayMode}
                  hasMasks={hasMasks} hasTrees={hasTrees}
                  threshold={threshold} setThreshold={setThreshold}
                  filter={filter} setFilter={setFilter}
                  treeCount={visibleCount}
                />
              )}

              <ScanProgressCard scanProgress={scanProgress} onCancel={cancelScan} />

              <HistoryDrawer
                open={drawerOpen}
                onClose={() => setDrawerOpen(false)}
                scans={scans}
                snapshots={snapshots}
                onDeleteScan={handleDeleteScan}
                onDeleteSnapshot={handleDeleteSnapshot}
                loading={aggLoading}
              />
            </>
          )}
        </div>
      </div>

      <Toast msg={toast} kind={toastKind} />
    </div>
  );
}

/* ==================================================================
   Mount
   ================================================================== */
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
