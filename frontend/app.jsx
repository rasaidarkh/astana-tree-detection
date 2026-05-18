/* =========================================================================
   Canopy — Astana urban tree inventory · UI v2
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
    eye:        <><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>,
    eyeOff:     <><path d="M9.9 4.2A9.7 9.7 0 0 1 12 4c6.5 0 10 7 10 7a17.6 17.6 0 0 1-3.3 4.3M6.6 6.6A17.6 17.6 0 0 0 2 11s3.5 7 10 7c1.9 0 3.6-.4 5.1-1.1" /><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" /><line x1="2" y1="2" x2="22" y2="22" /></>,
    edit:       <><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z" /></>,
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
        <svg className="topbar-mark" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M42 6 C 18 10, 8 18, 6 42 C 32 38, 42 30, 42 6 Z" fill="#7cd6a0" />
          <path d="M42 6 C 18 10, 8 18, 6 42 C 16 26, 28 14, 42 6 Z" fill="#9fdfb8" />
        </svg>
        <div>
          <div className="topbar-title">Canopy</div>
          <div className="topbar-sub">Astana urban tree inventory · 2026</div>
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
   LeftPanel — rich info panel in map view (replaces floating cards)
   ================================================================== */
function LeftPanel({
  aggregateStats, scans, snapshots,
  scanMode, polygonMode, onStartScan, onStartPolygon, onCancel,
  pendingPolygon, onStartPolygonScan, onClearPolygon,
  threshold, setThreshold, filter, setFilter,
  visibleCount,
  onOpenManager,
  onToggleScanHidden,
}) {
  const t = (aggregateStats && aggregateStats.total_trees) || 0;
  const s = (aggregateStats && aggregateStats.snapshot_count) || 0;
  const r = (aggregateStats && aggregateStats.run_count) || 0;
  const avgConf = aggregateStats?.avg_confidence;
  const avgCrown = aggregateStats?.avg_crown_m;
  const totalScans = scans?.length || 0;

  // Calculate aggregate canopy: sum of avg trees * avg crown — rough estimate
  // for display only. Real coverage requires per-scan area which we don't
  // aggregate yet. Show as "trees per scan" instead — more honest.
  const treesPerScan = totalScans > 0 ? Math.round(t / totalScans) : 0;

  const drawing = scanMode || polygonMode;
  const hasPending = pendingPolygon && pendingPolygon.length >= 3;

  return (
    <aside className="left-panel">
      <div className="lp-scroll">

        {/* ── primary actions (model is picked in a centered modal on click) ── */}
        {!drawing && !hasPending && (
          <div className="lp-actions">
            <button className="lp-action primary" onClick={onStartScan}>
              <Icon name="grid" size={18} />
              <span>Scan area</span>
            </button>
            <button className="lp-action" onClick={onStartPolygon}>
              <Icon name="polygon" size={18} />
              <span>Polygon</span>
            </button>
          </div>
        )}
        {drawing && !hasPending && (
          <div className="lp-actions" style={{ flexDirection: "column" }}>
            <div className="action-hint" style={{ background: "var(--accent-softer)", borderRadius: "var(--r-3)", width: "100%" }}>
              <span className="pulse"></span>
              <span>
                {scanMode
                  ? "Drag a rectangle on the map"
                  : "Click to add vertices · double-click to finish · right-click to clear"}
              </span>
            </div>
            <button className="lp-action" onClick={onCancel} style={{ width: "100%" }}>
              <Icon name="x" size={14} />
              <span>Cancel</span>
            </button>
          </div>
        )}
        {hasPending && (
          <div className="lp-actions" style={{ flexDirection: "column" }}>
            <div className="action-hint" style={{ background: "var(--accent-softer)", borderRadius: "var(--r-3)", width: "100%" }}>
              <span className="pulse"></span>
              <span>Polygon ready · <b>{pendingPolygon.length}</b> vertices</span>
            </div>
            <button className="lp-action primary" onClick={onStartPolygonScan} style={{ width: "100%", flexDirection: "row" }}>
              <Icon name="play" size={14} />
              <span>Start scan</span>
            </button>
            <button className="lp-action" onClick={onClearPolygon} style={{ width: "100%", flexDirection: "row" }}>
              <Icon name="refresh" size={14} />
              <span>Redraw</span>
            </button>
          </div>
        )}

        {/* ── hero stat ── */}
        <div className="lp-hero">
          <div className="lp-eyebrow">Astana · canopy aggregate</div>
          <div className="lp-headline">
            {t.toLocaleString()}
            <span className="unit">trees</span>
          </div>
          <div className="lp-sub">across {totalScans} scan{totalScans === 1 ? "" : "s"} · {s} snapshot{s === 1 ? "" : "s"}</div>
          <div className="lp-hero-row">
            <span><b>{r}</b>runs</span>
            <span><b>{treesPerScan}</b>avg / scan</span>
            {visibleCount && visibleCount.total > 0 && (
              <span><b>{visibleCount.visible.toLocaleString()}</b>visible</span>
            )}
          </div>
        </div>

        {/* ── secondary metrics ── */}
        {(avgConf != null || avgCrown != null) && (
          <div className="lp-metrics">
            {avgConf != null && (
              <div className="lp-metric">
                <div className="lp-metric-v">
                  {Math.round(avgConf * 100)}<span className="unit">%</span>
                </div>
                <div className="lp-metric-k">avg confidence</div>
              </div>
            )}
            {avgCrown != null && (
              <div className="lp-metric">
                <div className="lp-metric-v">
                  {avgCrown.toFixed(1)}<span className="unit">m</span>
                </div>
                <div className="lp-metric-k">avg crown</div>
              </div>
            )}
          </div>
        )}

        {/* ── filters block ── */}
        <div className="lp-section">
          <div className="lp-section-head">
            <div className="lp-section-title">Filters</div>
          </div>
          <div className="lp-filter-block">
            <div className="field-label">
              <span>Min confidence</span>
              <span className="mono">{Math.round(threshold * 100)}%</span>
            </div>
            <input
              type="range" min={0} max={1} step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="range"
            />
            <div className="tier-chips" style={{ marginTop: 10 }}>
              {[
                { k: "high", label: "High",   color: "var(--conf-high)" },
                { k: "med",  label: "Med",    color: "var(--conf-mid)"  },
                { k: "low",  label: "Low",    color: "var(--conf-low)"  },
              ].map((tt) => (
                <button
                  key={tt.k}
                  className={`tier-chip ${filter[tt.k] ? "on" : ""}`}
                  onClick={() => setFilter({ ...filter, [tt.k]: !filter[tt.k] })}
                >
                  <span className="dot" style={{ background: tt.color, opacity: filter[tt.k] ? 1 : 0.3 }}></span>
                  <span>{tt.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── recent activity ── */}
        <div className="lp-section">
          <div className="lp-section-head">
            <div className="lp-section-title">Recent scans</div>
            <button className="lp-section-action" onClick={onOpenManager}>
              <span>Manage</span>
              <Icon name="chevron" size={11} />
            </button>
          </div>
          {(!scans || scans.length === 0) && (
            <div className="list-empty" style={{ padding: "20px 8px", fontSize: 11.5 }}>
              No scans yet — click <b>Scan area</b> above to start
            </div>
          )}
          {scans && scans.slice(0, 6).map((s) => {
            const date = s.created_at ? new Date(s.created_at) : null;
            const ago = date ? timeAgo(date) : "";
            const name = s.display_name || `Scan ${s.id.slice(0, 8)}`;
            const dotColor = s.status === "completed" ? "var(--success)" : "var(--warning)";
            const hidden = !!s.hidden;
            return (
              <div key={s.id} className={`activity-row ${hidden ? "hidden" : ""}`}>
                <span className="activity-dot" style={{ background: dotColor }}></span>
                <div className="activity-body" onClick={onOpenManager} style={{ cursor: "pointer" }}>
                  <div className="activity-name">{name}</div>
                  <div className="activity-meta">
                    {(s.total_trees || 0).toLocaleString()} trees · {ago} · {s.model}
                  </div>
                </div>
                <button
                  className="activity-eye"
                  onClick={(e) => { e.stopPropagation(); onToggleScanHidden(s.id, hidden); }}
                  title={hidden ? "Show on map" : "Hide from map"}
                >
                  <Icon name={hidden ? "eyeOff" : "eye"} size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function timeAgo(date) {
  const sec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  const d = Math.floor(sec / 86400);
  if (d < 7) return `${d}d ago`;
  return date.toLocaleDateString();
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
function ScanProgressCard({ scanProgress, onCancel, onRename }) {
  // Локальный draft для post-scan rename input (показывается когда .done && sessionId)
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  if (!scanProgress) return null;
  const regions = scanProgress.regions || [];
  const total = regions.length;
  const done = regions.filter((r) => r.status === "done").length;
  const errs = regions.filter((r) => r.status === "error").length;
  const treeCount = scanProgress.trees ? scanProgress.trees.length : 0;
  const cols = Math.max(1, Math.ceil(Math.sqrt(total || 1)));
  const pct = total ? Math.round((done / total) * 100) : 0;
  const isDone = scanProgress.done && scanProgress.sessionId;

  const submit = async () => {
    if (!isDone || !draft.trim()) return;
    setSaving(true);
    try { await onRename(scanProgress.sessionId, draft.trim()); }
    finally { setSaving(false); }
  };

  return (
    <div className="float float-tc scan-progress">
      <div className="scan-progress-head">
        {scanProgress.done
          ? <Icon name="check" size={14} stroke={2.5} />
          : <div className="scan-spinner" />}
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

      {/* Optional post-scan rename — non-blocking, юзер может проигнорировать */}
      {isDone && (
        <div style={{ display: "flex", gap: 6, alignItems: "stretch" }}>
          <input
            className="input"
            placeholder="Name this scan (optional)…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            maxLength={120}
            autoFocus
          />
          <button
            className="btn btn-primary"
            style={{ width: "auto", padding: "0 14px", flexShrink: 0 }}
            onClick={submit}
            disabled={saving || !draft.trim()}
          >
            <Icon name="check" size={12} stroke={2.5} />
            <span>Save</span>
          </button>
        </div>
      )}
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
   ManagerModal — full-feature management for scans + snapshots
   With search, sort, inline rename, delete actions.
   ================================================================== */
function ManagerModal({
  open, onClose,
  scans, snapshots,
  onDeleteScan, onDeleteSnapshot,
  onRenameScan, onRenameSnapshot,
  onToggleScanHidden,
  loading,
}) {
  const [tab, setTab] = useState("scans");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("date_desc");

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const filterAndSort = (items, getDate, getTrees, getName) => {
    let out = items.filter((it) => {
      if (!query.trim()) return true;
      const q = query.toLowerCase();
      return (getName(it) || "").toLowerCase().includes(q)
          || (it.id || it.image_id || "").toLowerCase().includes(q);
    });
    out.sort((a, b) => {
      switch (sort) {
        case "date_asc":   return new Date(getDate(a)) - new Date(getDate(b));
        case "trees_desc": return (getTrees(b) || 0) - (getTrees(a) || 0);
        case "trees_asc":  return (getTrees(a) || 0) - (getTrees(b) || 0);
        case "name":       return (getName(a) || "").localeCompare(getName(b) || "");
        case "date_desc":
        default:           return new Date(getDate(b)) - new Date(getDate(a));
      }
    });
    return out;
  };

  const visibleScans = filterAndSort(
    scans || [], (s) => s.created_at, (s) => s.total_trees,
    (s) => s.display_name || `Scan ${s.id.slice(0, 8)}`,
  );
  const visibleSnaps = filterAndSort(
    snapshots || [], (s) => s.created_at, (s) => s.total_trees,
    (s) => s.display_name || s.filename,
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <Icon name="layers" size={16} />
          <div className="modal-title">Library</div>
          <div className="modal-tabs">
            <button className={`modal-tab ${tab === "scans" ? "active" : ""}`} onClick={() => setTab("scans")}>
              Scans · {scans?.length || 0}
            </button>
            <button className={`modal-tab ${tab === "snapshots" ? "active" : ""}`} onClick={() => setTab("snapshots")}>
              Snapshots · {snapshots?.length || 0}
            </button>
          </div>
          <button className="icon-btn" onClick={onClose}><Icon name="x" size={14} /></button>
        </div>

        <div className="modal-toolbar">
          <input
            className="search-input"
            placeholder={tab === "scans" ? "Search by name or ID…" : "Search snapshots…"}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select className="select-native" value={sort} onChange={(e) => setSort(e.target.value)} style={{ width: 160 }}>
            <option value="date_desc">Newest first</option>
            <option value="date_asc">Oldest first</option>
            <option value="trees_desc">Most trees</option>
            <option value="trees_asc">Fewest trees</option>
            <option value="name">Name (A–Z)</option>
          </select>
        </div>

        <div className="modal-body">
          {loading && <div className="empty-grid">Loading…</div>}

          {!loading && tab === "scans" && (
            <div className="card-grid">
              {visibleScans.length === 0 && (
                <div className="empty-grid">
                  {query ? `No scans matching "${query}"` : (
                    <>
                      No scans yet. Close this modal and click <b>Scan area</b> to start.
                    </>
                  )}
                </div>
              )}
              {visibleScans.map((s) => (
                <ScanCard
                  key={s.id} scan={s}
                  onRename={(name) => onRenameScan(s.id, name)}
                  onDelete={() => onDeleteScan(s.id, s.sub_count)}
                  onToggleHidden={onToggleScanHidden}
                />
              ))}
            </div>
          )}

          {!loading && tab === "snapshots" && (
            <div className="card-grid">
              {visibleSnaps.length === 0 && (
                <div className="empty-grid">
                  {query ? `No snapshots matching "${query}"` : "No snapshots yet."}
                </div>
              )}
              {visibleSnaps.map((s) => (
                <SnapshotCard
                  key={s.image_id} snap={s}
                  onRename={(name) => onRenameSnapshot(s.image_id, name)}
                  onDelete={() => onDeleteSnapshot(s.image_id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function EditableName({ value, fallback, onSave }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value || "");
  const ref = useRef(null);
  useEffect(() => { setDraft(value || ""); }, [value]);
  useEffect(() => { if (editing && ref.current) { ref.current.focus(); ref.current.select(); } }, [editing]);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed !== (value || "")) onSave(trimmed || null);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        ref={ref}
        className="mgr-card-name editing"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") { setDraft(value || ""); setEditing(false); }
        }}
        maxLength={120}
      />
    );
  }
  return (
    <div
      className="mgr-card-name"
      title={value || fallback}
      onDoubleClick={() => setEditing(true)}
      onClick={() => setEditing(true)}
    >
      {value || fallback}
    </div>
  );
}

function ScanCard({ scan, onRename, onDelete, onToggleHidden }) {
  const date = scan.created_at ? new Date(scan.created_at) : null;
  const hidden = !!scan.hidden;
  return (
    <div className={`mgr-card ${hidden ? "is-hidden" : ""}`}>
      <div className="mgr-card-head">
        <EditableName
          value={scan.display_name}
          fallback={`Scan ${scan.id.slice(0, 8)}`}
          onSave={onRename}
        />
        {scan.polygon_json && <span className="tag gray">polygon</span>}
        {scan.status === "running" && <span className="tag running">running</span>}
        <button
          className="icon-btn"
          style={{ width: 28, height: 28 }}
          onClick={() => onToggleHidden(scan.id, hidden)}
          title={hidden ? "Show on map" : "Hide from map"}
        >
          <Icon name={hidden ? "eyeOff" : "eye"} size={13} />
        </button>
      </div>
      <div className="mgr-card-stats">
        <div className="mgr-card-stat">
          <div className="mgr-card-stat-v">{(scan.total_trees || 0).toLocaleString()}</div>
          <div className="mgr-card-stat-k">Trees</div>
        </div>
        <div className="mgr-card-stat">
          <div className="mgr-card-stat-v">{scan.ok_count}/{scan.sub_count}</div>
          <div className="mgr-card-stat-k">Sub-regions</div>
        </div>
      </div>
      <div className="mgr-card-meta">
        <span className="tag-soft">{scan.provider}</span>
        <span className="tag-soft">z{scan.zoom}</span>
        <span className="tag-soft">{scan.model}</span>
        {scan.duration_ms ? <span className="tag-soft">{(scan.duration_ms / 1000).toFixed(0)}s</span> : null}
      </div>
      <div className="mgr-card-coords">
        {scan.nw_lat.toFixed(5)}°, {scan.nw_lng.toFixed(5)}° → {scan.se_lat.toFixed(5)}°, {scan.se_lng.toFixed(5)}°
      </div>
      {date && <div className="mgr-card-coords muted">{date.toLocaleString()}</div>}
      <div className="mgr-card-actions">
        <span className="mgr-card-mono" style={{ flex: 1 }}>{scan.id.slice(0, 12)}</span>
        <button className="ghost-btn danger" onClick={onDelete}>
          <Icon name="trash" size={11} /><span>Delete</span>
        </button>
      </div>
    </div>
  );
}

function SnapshotCard({ snap, onRename, onDelete }) {
  return (
    <div className="mgr-card">
      <div className="mgr-card-head">
        <EditableName
          value={snap.display_name}
          fallback={snap.filename}
          onSave={onRename}
        />
      </div>
      <div className="mgr-card-stats">
        <div className="mgr-card-stat">
          <div className="mgr-card-stat-v">{(snap.total_trees || 0).toLocaleString()}</div>
          <div className="mgr-card-stat-k">Trees</div>
        </div>
        <div className="mgr-card-stat">
          <div className="mgr-card-stat-v">{snap.run_count || 0}</div>
          <div className="mgr-card-stat-k">Runs</div>
        </div>
      </div>
      <div className="mgr-card-meta">
        <span className="tag-soft">{snap.width}×{snap.height}</span>
        {snap.last_model && <span className="tag-soft">{snap.last_model}</span>}
        {snap.is_geotiff && <span className="tag-soft">GeoTIFF</span>}
      </div>
      {snap.nw_lat != null && (
        <div className="mgr-card-coords">
          N {snap.nw_lat.toFixed(5)}° → S {snap.se_lat.toFixed(5)}°
        </div>
      )}
      <div className="mgr-card-actions">
        <span className="mgr-card-mono" style={{ flex: 1 }}>{snap.image_id.slice(0, 12)}</span>
        <button className="ghost-btn danger" onClick={onDelete}>
          <Icon name="trash" size={11} /><span>Delete</span>
        </button>
      </div>
    </div>
  );
}

/* ==================================================================
   Settings popover (top-right, accessed via gear)
   ================================================================== */
function SettingsPopover({ open, tileProvider, setTileProvider, providersMap, modelStatus, model, onClose }) {
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
        <div className="field-help">
          Source for both map display and scan capture. The detection model is
          picked from the sidebar's Detection section, not here.
        </div>
      </div>

      <div className="popover-row">
        <div className="popover-label">Model status</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {Object.entries(models)
            .filter(([k]) => k !== "yolo")
            .map(([k, m]) => (
              <div key={k} className="row" style={{ fontSize: 11 }}>
                <span className="status-dot" style={{ background: m.available ? "var(--success)" : "var(--text-3)" }}></span>
                <span style={{ flex: 1, fontWeight: k === model ? 600 : 400 }}>
                  {m.name || k}
                </span>
                <span className="muted mono" style={{ fontSize: 10 }}>
                  {k === model ? "active · " : ""}{m.available ? (m.loaded ? "ready" : "lazy") : "—"}
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

/* ==================================================================
   Model picker — hierarchical (family → variant).

   Row 1: family selector (YOLO / DeepForest / Mask R-CNN / Ensemble).
   Row 2: variant selector inside the active family.
   Skips Row 2 when family has only one variant (Mask R-CNN, Ensemble).
   ================================================================== */

// Family definitions — ordered. Each variant has a short pill label;
// hover-title shows the full backend label with mAP if registered.
const MODEL_FAMILIES = [
  {
    id: "yolo",
    label: "YOLOv8",
    variants: [
      { kind: "yolo_v4_x",    short: "v4 x · champ" },
      { kind: "yolo_v4_m",    short: "v4 m" },
      { kind: "yolo_v4_s",    short: "v4 s · fast" },
      { kind: "yolo_v3_exp1", short: "v3 exp1" },
      { kind: "yolo_v3_run1", short: "v3 run 1" },
      { kind: "yolo_v3_run2", short: "v3 run 2" },
      { kind: "yolo_v2",      short: "v2 legacy" },
    ],
  },
  {
    id: "deepforest",
    label: "DeepForest",
    variants: [
      { kind: "deepforest_sam2", short: "with SAM 2" },
      { kind: "deepforest",      short: "boxes only" },
    ],
  },
  {
    id: "maskrcnn",
    label: "Mask R-CNN",
    variants: [
      { kind: "maskrcnn", short: "R50-FPN v2" },
    ],
  },
  {
    id: "ensemble",
    label: "Ensemble",
    variants: [
      { kind: "ensemble", short: "WBF (YOLO + DF)" },
    ],
  },
];

// Reverse lookup: ModelKind value → family id
const KIND_TO_FAMILY = {};
for (const fam of MODEL_FAMILIES) {
  for (const v of fam.variants) KIND_TO_FAMILY[v.kind] = fam.id;
}

function _familyForModel(model) {
  return KIND_TO_FAMILY[model] || "yolo";
}

function ModelPicker({ model, setModel, modelStatus }) {
  const models = modelStatus?.models || {};
  const currentFamilyId = _familyForModel(model);
  // Local "shown" family — user can browse other families without
  // changing the active model until they pick a variant.
  const [shownFamily, setShownFamily] = useState(currentFamilyId);
  // Keep shown family in sync if model is changed externally (e.g. user
  // selects from popover or from a deep link)
  useEffect(() => { setShownFamily(currentFamilyId); }, [currentFamilyId]);

  const shownDef = MODEL_FAMILIES.find((f) => f.id === shownFamily) || MODEL_FAMILIES[0];
  const availableVariants = shownDef.variants.filter((v) => models[v.kind]);

  const FamilyBtn = ({ fam }) => {
    const active = shownFamily === fam.id;
    const isCurrent = currentFamilyId === fam.id;  // model currently in this family
    const anyAvail = fam.variants.some((v) => models[v.kind]?.available);
    return (
      <button
        type="button"
        className={`seg-btn ${active ? "active" : ""}`}
        onClick={() => setShownFamily(fam.id)}
        disabled={!anyAvail}
        style={{ flex: 1, justifyContent: "center", position: "relative" }}
        title={fam.label}
      >
        {fam.label}
        {isCurrent && !active && (
          <span
            style={{
              position: "absolute", top: 4, right: 6,
              width: 6, height: 6, borderRadius: "50%",
              background: "var(--accent)",
            }}
            title="model active"
          />
        )}
      </button>
    );
  };

  const VariantBtn = ({ v }) => {
    const m = models[v.kind];
    const available = !!m?.available;
    const active = model === v.kind;
    return (
      <button
        type="button"
        className={`seg-btn ${active ? "active" : ""}`}
        onClick={() => available && setModel(v.kind)}
        disabled={!available}
        style={{ flex: 1, justifyContent: "center", whiteSpace: "nowrap", fontSize: 11 }}
        title={m?.name || v.kind}
      >
        {v.short}
      </button>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div>
        <div className="field-label">Detector</div>
        <div
          className="control-group"
          style={{ background: "var(--surface-2)", boxShadow: "none", borderColor: "var(--border)", marginTop: 4 }}
        >
          {MODEL_FAMILIES.map((fam) => <FamilyBtn key={fam.id} fam={fam} />)}
        </div>
      </div>

      {availableVariants.length > 1 && (
        <div>
          <div className="field-label">{shownDef.label} variant</div>
          <div
            className="control-group"
            style={{
              background: "var(--surface-2)", boxShadow: "none", borderColor: "var(--border)",
              marginTop: 4, flexWrap: "wrap",
            }}
          >
            {availableVariants.map((v) => <VariantBtn key={v.kind} v={v} />)}
          </div>
        </div>
      )}

      <div className="field-help" style={{ marginTop: 2 }}>
        Active: <b>{models[model]?.name || model}</b>
      </div>
    </div>
  );
}

/* ==================================================================
   ScanModelModal — centered modal that asks "which detector" right
   before kicking off Scan area / Polygon flows. Keeps LeftPanel
   minimalist while giving the user a clear at-action choice.
   ================================================================== */
function ScanModelModal({ open, action, model, setModel, modelStatus, onConfirm, onCancel }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") onConfirm();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onCancel, onConfirm]);

  if (!open) return null;

  const actionLabel = action === "scan"
    ? "Scan rectangle"
    : action === "polygon"
      ? "Polygon scan"
      : "Run detection";
  const actionHint = action === "scan"
    ? "After confirmation, drag a rectangle on the map to define the scan area."
    : action === "polygon"
      ? "After confirmation, click vertices on the map · double-click to close polygon."
      : "";

  return (
    <div
      className="modal-overlay"
      onClick={onCancel}
      style={{ alignItems: "center" }}
    >
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 480, padding: 0 }}
      >
        <div className="modal-head" style={{ paddingBottom: 12 }}>
          <Icon name={action === "polygon" ? "polygon" : "grid"} size={16} />
          <div className="modal-title">{actionLabel}</div>
        </div>
        <div style={{ padding: "0 20px 12px 20px" }}>
          <ModelPicker model={model} setModel={setModel} modelStatus={modelStatus} />
          {actionHint && (
            <div className="field-help" style={{ marginTop: 12, padding: "8px 10px", background: "var(--accent-softer)", borderRadius: "var(--r-2)" }}>
              {actionHint}
            </div>
          )}
        </div>
        <div style={{
          display: "flex", gap: 8, padding: "12px 20px 20px 20px",
          borderTop: "1px solid var(--border)", justifyContent: "flex-end",
        }}>
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn primary" onClick={onConfirm}>
            <Icon name="play" size={14} />
            <span style={{ marginLeft: 6 }}>
              {action === "polygon" ? "Start drawing" : "Start"}
            </span>
          </button>
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
              <ModelPicker model={model} setModel={setModel} modelStatus={modelStatus} />
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
  const [dark, setDark] = useState(true);
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
  // Most-recently-completed scan id — backing data for the optional rename
  // chip that appears in the scan progress card after `done` event.
  const [lastFinishedScan, setLastFinishedScan] = useState(null);

  // ---- model + threshold (shared) ----
  // Default to v4_x — the actual measured champion across all 28 experiments
  // (mAP@50 0.315 on merged val, beats prior exp1 0.308). Picker handles
  // graceful fallback if checkpoint isn't registered.
  const [model, setModel] = useState("yolo_v4_x");
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
  const [managerOpen, setManagerOpen] = useState(false);
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

  // Fallback model selection — if backend doesn't have the default model
  // (e.g. running an older version without v4_x_clean registered), pick the
  // first available variant from MODEL_FAMILIES priority order.
  useEffect(() => {
    if (!modelStatus?.models) return;
    if (modelStatus.models[model]?.available) return;  // current is OK
    for (const fam of MODEL_FAMILIES) {
      for (const v of fam.variants) {
        if (modelStatus.models[v.kind]?.available) {
          setModel(v.kind);
          return;
        }
      }
    }
  }, [modelStatus, model]);

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
  // Model is picked at the moment of action via a centered modal. The
  // user clicks "Scan area" / "Polygon" → modal opens with model picker
  // → on confirm, the actual drawing mode is entered.
  const [pendingScanAction, setPendingScanAction] = useState(null); // "scan" | "polygon" | null

  const requestScanRect = useCallback(() => setPendingScanAction("scan"), []);
  const requestScanPoly = useCallback(() => setPendingScanAction("polygon"), []);

  const confirmScanAction = useCallback(() => {
    if (pendingScanAction === "scan") {
      setPolygonMode(false); setPendingPolygon(null); setScanMode(true);
    } else if (pendingScanAction === "polygon") {
      setScanMode(false); setPendingPolygon(null); setPolygonMode(true);
    }
    setPendingScanAction(null);
  }, [pendingScanAction]);

  const cancelScanAction = useCallback(() => setPendingScanAction(null), []);

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
            setScanProgress((p) => ({ ...p, done: true, sessionId: ev.scan_session_id }));
            setLastFinishedScan({ id: ev.scan_session_id, trees: ev.total_trees, when: Date.now() });
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

  const handleRenameScan = useCallback(async (id, name) => {
    try { await window.api.renameScan(id, name); await refreshAggregate(); }
    catch (e) { showToast("Rename failed: " + e.message, "error"); }
  }, [refreshAggregate, showToast]);
  const handleRenameSnapshot = useCallback(async (id, name) => {
    try { await window.api.renameSnapshot(id, name); await refreshAggregate(); }
    catch (e) { showToast("Rename failed: " + e.message, "error"); }
  }, [refreshAggregate, showToast]);

  const handleToggleScanHidden = useCallback(async (id, currentlyHidden) => {
    // optimistic — обновляем locally, потом refreshAggregate подтянет настоящее.
    setScans((prev) => prev.map((s) => s.id === id ? { ...s, hidden: !currentlyHidden ? 1 : 0 } : s));
    try {
      await window.api.setScanHidden(id, !currentlyHidden);
      await refreshAggregate();
    } catch (e) {
      showToast("Toggle failed: " + e.message, "error");
      // Откатываем locally если упало.
      setScans((prev) => prev.map((s) => s.id === id ? { ...s, hidden: currentlyHidden ? 1 : 0 } : s));
    }
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
        model={model}
        onClose={() => setSettingsOpen(false)}
      />

      <ScanModelModal
        open={pendingScanAction !== null}
        action={pendingScanAction}
        model={model} setModel={setModel} modelStatus={modelStatus}
        onConfirm={confirmScanAction}
        onCancel={cancelScanAction}
      />

      <div className={`view-shell ${view === "image" ? "layout-image" : ""}`}>
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

        {view === "map" && (
          <LeftPanel
            aggregateStats={aggregateStats}
            scans={scans}
            snapshots={snapshots}
            scanMode={scanMode} polygonMode={polygonMode}
            onStartScan={requestScanRect} onStartPolygon={requestScanPoly}
            onCancel={cancelDraw}
            pendingPolygon={pendingPolygon}
            onStartPolygonScan={handleStartPolygonScan}
            onClearPolygon={handleClearPolygon}
            threshold={threshold} setThreshold={setThreshold}
            filter={filter} setFilter={setFilter}
            visibleCount={visibleCount}
            onOpenManager={() => setManagerOpen(true)}
            onToggleScanHidden={handleToggleScanHidden}
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

              {hasTrees && (
                <DisplayStrip
                  displayMode={displayMode} setDisplayMode={setDisplayMode}
                  hasMasks={hasMasks} hasTrees={hasTrees}
                  threshold={threshold} setThreshold={setThreshold}
                  filter={filter} setFilter={setFilter}
                  treeCount={visibleCount}
                />
              )}

              <ScanProgressCard
                scanProgress={scanProgress}
                onCancel={cancelScan}
                onRename={handleRenameScan}
              />

              <ManagerModal
                open={managerOpen}
                onClose={() => setManagerOpen(false)}
                scans={scans}
                snapshots={snapshots}
                onDeleteScan={handleDeleteScan}
                onDeleteSnapshot={handleDeleteSnapshot}
                onRenameScan={handleRenameScan}
                onRenameSnapshot={handleRenameSnapshot}
                onToggleScanHidden={handleToggleScanHidden}
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
