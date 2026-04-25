import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import MapComponent from './MapComponent';
import { getMapBootstrap, getIncident, getIncidentUpdates } from '../api';
import { useAuth } from './auth/AuthContext';
import '../styles/dashboard.css';

const WESTERN_CONUS_BBOX = '-125.1,31.0,-101.8,49.5';
const DEFAULT_STYLE = 'mapbox://styles/mapbox/satellite-streets-v12';

const DEFAULT_LAYERS = {
  incidents: true,
  perimeters: true,
  hotspots: true,
  warnings: true,
  evacuations: false,
  prediction: true,
  ndvi: false,
};

const DRAWER_TABS = [
  { id: 'incidents', label: 'Incidents' },
  { id: 'warnings', label: 'Warnings' },
  { id: 'layers', label: 'Layers' },
  { id: 'history', label: 'History' },
];

function formatCount(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : '-';
}

function formatAcres(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  if (n >= 1000) return `${Math.round(n).toLocaleString()} acres`;
  return `${Math.round(n * 10) / 10} acres`;
}

function formatRelative(value) {
  if (!value) return 'unknown';
  const ts = Date.parse(value);
  if (!Number.isFinite(ts)) return 'unknown';
  const minutes = Math.max(0, Math.round((Date.now() - ts) / 60000));
  if (minutes < 60) return `${minutes || 1} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} hr ago`;
  return new Date(ts).toLocaleDateString();
}

function incidentMarkerClass(incident) {
  if (incident.status !== 'active') return 'muted';
  if (Number(incident.acres) >= 1000) return 'major';
  if (Number(incident.acres) >= 100) return 'large';
  return 'active';
}

function sourceHealthLabel(status = {}) {
  if (status.ok && !status.partial && !status.stale) return 'Live';
  if (status.ok) return 'Partial';
  return 'Unavailable';
}

function IncidentCard({ incident, selected, onSelect }) {
  return (
    <button
      type="button"
      className={`map-list-card incident-card ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(incident)}
    >
      <span className={`incident-flame ${incidentMarkerClass(incident)}`} aria-hidden="true" />
      <span className="map-list-card-body">
        <span className="map-list-card-title">{incident.name}</span>
        <span className="map-list-card-subtitle">
          {[incident.county, incident.state].filter(Boolean).join(', ') || 'Unknown location'}
        </span>
        <span className="map-list-card-meta">
          <strong>{formatAcres(incident.acres)}</strong>
          <span>{incident.status}</span>
          <span>{formatRelative(incident.updatedAt)}</span>
        </span>
      </span>
    </button>
  );
}

function WarningCard({ warning, selected, onSelect }) {
  return (
    <button
      type="button"
      className={`map-list-card warning-card ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(warning)}
    >
      <span className="warning-swatch" aria-hidden="true" />
      <span className="map-list-card-body">
        <span className="map-list-card-title">{warning.event}</span>
        <span className="map-list-card-subtitle">{warning.areaDesc || warning.headline}</span>
        <span className="map-list-card-meta">
          <strong>{warning.status || 'Alert'}</strong>
          <span>{formatRelative(warning.updatedAt || warning.effective)}</span>
        </span>
      </span>
    </button>
  );
}

function LayerToggle({ id, label, source, enabled, status, onToggle }) {
  return (
    <label className="layer-toggle">
      <input
        type="checkbox"
        checked={Boolean(enabled)}
        onChange={() => onToggle(id)}
      />
      <span>
        <strong>{label}</strong>
        <small>{source}</small>
      </span>
      {status ? <em>{sourceHealthLabel(status)}</em> : null}
    </label>
  );
}

function EmptyDrawerState({ label }) {
  return (
    <div className="empty-state">
      <strong>No {label} in this view</strong>
      <span>Try refreshing the map or widening the search.</span>
    </div>
  );
}

function Drawer({
  open,
  activeTab,
  setActiveTab,
  incidents,
  warnings,
  selectedIncident,
  selectedWarning,
  onIncidentSelect,
  onWarningSelect,
  layerVisibility,
  layerStatus,
  onToggleLayer,
  onOpenHistory,
  search,
  setSearch,
  onClose,
}) {
  return (
    <aside className={`map-drawer ${open ? 'open' : ''}`} aria-label="Map menu">
      <div className="drawer-header">
        <div>
          <span className="eyebrow">Western CONUS</span>
          <h2>Wildfire Intelligence</h2>
        </div>
        <button className="icon-btn" type="button" onClick={onClose} aria-label="Close menu">x</button>
      </div>

      <div className="drawer-tabs" role="tablist">
        {DRAWER_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'incidents' && (
        <div className="drawer-pane">
          <input
            className="drawer-search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search incidents or counties"
            aria-label="Search incidents"
          />
          <div className="drawer-list">
            {incidents.length ? incidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                selected={selectedIncident?.id === incident.id}
                onSelect={onIncidentSelect}
              />
            )) : <EmptyDrawerState label="incidents" />}
          </div>
        </div>
      )}

      {activeTab === 'warnings' && (
        <div className="drawer-pane">
          <div className="drawer-list">
            {warnings.length ? warnings.map((warning) => (
              <WarningCard
                key={warning.id}
                warning={warning}
                selected={selectedWarning?.id === warning.id}
                onSelect={onWarningSelect}
              />
            )) : <EmptyDrawerState label="warnings" />}
          </div>
        </div>
      )}

      {activeTab === 'layers' && (
        <div className="drawer-pane layer-pane">
          <LayerToggle id="incidents" label="Incidents" source="WFIGS/NIFC" enabled={layerVisibility.incidents} status={layerStatus.incidents} onToggle={onToggleLayer} />
          <LayerToggle id="perimeters" label="Official perimeters" source="WFIGS/FIRIS" enabled={layerVisibility.perimeters} status={layerStatus.perimeters} onToggle={onToggleLayer} />
          <LayerToggle id="hotspots" label="FIRMS hotspots" source="NASA VIIRS/MODIS" enabled={layerVisibility.hotspots} status={layerStatus.hotspots} onToggle={onToggleLayer} />
          <LayerToggle id="warnings" label="Fire weather warnings" source="NWS Alerts API" enabled={layerVisibility.warnings} status={layerStatus.alerts} onToggle={onToggleLayer} />
          <LayerToggle id="evacuations" label="Evacuations" source="Official providers" enabled={layerVisibility.evacuations} status={layerStatus.evacuations} onToggle={onToggleLayer} />
          <LayerToggle id="prediction" label="Ignis prediction" source="Advisory model output" enabled={layerVisibility.prediction} onToggle={onToggleLayer} />
          <LayerToggle id="ndvi" label="NDVI overlay" source="Vegetation context" enabled={layerVisibility.ndvi} onToggle={onToggleLayer} />
        </div>
      )}

      {activeTab === 'history' && (
        <div className="drawer-pane history-pane">
          <h3>Historical Testing</h3>
          <p>Open the existing historical model presets for Palisades, Eaton, Dixie, Caldor, and Camp/Paradise.</p>
          <button className="primary-action" type="button" onClick={onOpenHistory}>
            Open History Tools
          </button>
          <p className="fine-print">
            Historical forecasts stay advisory until matched against real progression polygons.
          </p>
        </div>
      )}
    </aside>
  );
}

function IncidentDetailPanel({
  incident,
  detail,
  activeTab,
  setActiveTab,
  runningPrediction,
  onRunPrediction,
  onClose,
}) {
  if (!incident) return null;
  const updates = detail?.updates || [];
  const predictionEligibility = detail?.predictionEligibility
    || incident.predictionEligibility
    || { eligible: incident.hasPrediction !== false, reasons: [] };
  const canPredict = Boolean(predictionEligibility?.eligible);
  const predictionReason = predictionEligibility?.reasons?.[0]
    || 'Prediction is only enabled when Ignis has enough hotspot or perimeter context.';
  return (
    <aside className="detail-panel open" aria-label="Incident detail">
      <div className="detail-panel-header">
        <div>
          <h2>{incident.name}</h2>
          <p>{[incident.county, incident.state].filter(Boolean).join(', ') || 'Unknown location'}</p>
        </div>
        <button className="icon-btn" type="button" onClick={onClose} aria-label="Close detail">x</button>
      </div>
      <div className="incident-stats">
        <div><span>Acres</span><strong>{formatCount(incident.acres)}</strong></div>
        <div><span>Containment</span><strong>{incident.containmentPct != null ? `${incident.containmentPct}%` : '-'}</strong></div>
      </div>
      <div className="detail-tabs">
        {['prediction', 'updates', 'info'].map((tab) => (
          <button
            key={tab}
            type="button"
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {tab[0].toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'prediction' && (
        <div className="detail-section">
          <div className="prediction-callout">
            <strong>Ignis Advisory Prediction</strong>
            <span>
              Run the ConvLSTM spread-risk workflow against the best available perimeter and hotspot context.
              This stays separate from official perimeter reporting.
            </span>
          </div>
          <div className="quality-grid">
            <div><span>Perimeter</span><strong>{incident.hasPerimeter ? 'Available' : 'Missing'}</strong></div>
            <div><span>Hotspots</span><strong>{incident.hasHotspots ? 'Live context' : 'Sparse'}</strong></div>
            <div><span>Prediction</span><strong>{canPredict ? 'Ready' : 'Limited'}</strong></div>
          </div>
          <button
            className="primary-action"
            type="button"
            disabled={runningPrediction || !canPredict}
            onClick={() => onRunPrediction(incident)}
          >
            {runningPrediction ? 'Running Prediction...' : 'Run Ignis Prediction'}
          </button>
          {!canPredict && (
            <p className="fine-print">{predictionReason}</p>
          )}
          {canPredict && (
            <p className="fine-print">
              Ignis runs from the strongest available incident geometry. FIRMS detections can steer the advisory center
              when the official incident point is too generic.
            </p>
          )}
        </div>
      )}

      {activeTab === 'updates' && (
        <div className="updates-list">
          {updates.length ? updates.map((update) => (
            <article key={update.id} className="update-item">
              <strong>{update.title}</strong>
              <span>{update.source} - {formatRelative(update.createdAt)}</span>
              <p>{update.body}</p>
            </article>
          )) : <EmptyDrawerState label="updates" />}
        </div>
      )}

      {activeTab === 'info' && (
        <div className="info-list">
          <div><span>Source</span><strong>{(incident.sourceNames || []).join(', ') || 'WFIGS'}</strong></div>
          <div><span>Created</span><strong>{incident.createdAt ? new Date(incident.createdAt).toLocaleString() : '-'}</strong></div>
          <div><span>Updated</span><strong>{incident.updatedAt ? new Date(incident.updatedAt).toLocaleString() : '-'}</strong></div>
          <div><span>Source ID</span><strong>{incident.sourceId || '-'}</strong></div>
          <p className="fine-print">Official incident feeds may lag field activity. Ignis prediction is an advisory model layer and should be interpreted separately from official perimeters and evacuation notices.</p>
        </div>
      )}
    </aside>
  );
}

function WarningDetailPanel({ warning, onClose }) {
  if (!warning) return null;
  return (
    <aside className="detail-panel open warning-detail" aria-label="Warning detail">
      <div className="detail-panel-header">
        <div>
          <h2>{warning.event}</h2>
          <p>{warning.areaDesc || 'NWS fire weather alert'}</p>
        </div>
        <button className="icon-btn" type="button" onClick={onClose} aria-label="Close detail">x</button>
      </div>
      <div className="incident-stats three">
        <div><span>Starts</span><strong>{warning.effective ? new Date(warning.effective).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '-'}</strong></div>
        <div><span>Ends</span><strong>{(warning.ends || warning.expires) ? new Date(warning.ends || warning.expires).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '-'}</strong></div>
        <div><span>Status</span><strong>{warning.status || '-'}</strong></div>
      </div>
      <div className="detail-section">
        <h3>Details</h3>
        <p className="warning-copy">{warning.description || warning.headline}</p>
        {warning.instruction ? <p className="warning-copy">{warning.instruction}</p> : null}
      </div>
    </aside>
  );
}

function LegendPanel({ open, onClose }) {
  if (!open) return null;
  return (
    <div className="legend-panel">
      <div className="legend-header">
        <strong>Legend</strong>
        <button className="icon-btn" type="button" onClick={onClose}>x</button>
      </div>
      <div className="legend-row"><span className="legend-symbol incident" /> Active incident</div>
      <div className="legend-row"><span className="legend-symbol perimeter" /> Official perimeter</div>
      <div className="legend-row"><span className="legend-symbol hotspot" /> FIRMS hotspot</div>
      <div className="legend-row"><span className="legend-symbol warning" /> Fire weather warning</div>
      <div className="legend-row"><span className="legend-ramp" /> Ignis advisory risk</div>
    </div>
  );
}

const AdvancedFireDashboard = () => {
  const { user } = useAuth();
  const mapRef = useRef(null);
  const [mapData, setMapData] = useState(null);
  const [isFetching, setIsFetching] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [activeDrawerTab, setActiveDrawerTab] = useState('incidents');
  const [layerVisibility, setLayerVisibility] = useState(DEFAULT_LAYERS);
  const [mapStyle] = useState(DEFAULT_STYLE);
  const [search, setSearch] = useState('');
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [selectedWarning, setSelectedWarning] = useState(null);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState(null);
  const [selectedIncidentUpdates, setSelectedIncidentUpdates] = useState([]);
  const [detailTab, setDetailTab] = useState('prediction');
  const [runningPrediction, setRunningPrediction] = useState(false);
  const [legendOpen, setLegendOpen] = useState(false);
  const [brightness] = useState('');
  const [confidence] = useState('');
  const [userLocation, setUserLocation] = useState(null);
  const [range] = useState(20);

  const loadMapData = useCallback(async () => {
    setIsFetching(true);
    try {
      const response = await getMapBootstrap({ bbox: WESTERN_CONUS_BBOX });
      const payload = response?.data ?? response;
      setMapData(payload);
    } catch (error) {
      console.error('map bootstrap failed', error);
      setMapData({
        updatedAt: new Date().toISOString(),
        incidents: [],
        alerts: [],
        perimeters: { type: 'FeatureCollection', features: [] },
        hotspotFootprints: { type: 'FeatureCollection', features: [] },
        layerStatus: { partial: true, error: error?.message || 'unavailable' },
      });
    } finally {
      setIsFetching(false);
    }
  }, []);

  useEffect(() => {
    loadMapData();
  }, [loadMapData]);

  const incidents = useMemo(() => {
    const all = mapData?.incidents || [];
    const needle = search.trim().toLowerCase();
    return all.filter((incident) => {
      if (!needle) return true;
      return [incident.name, incident.county, incident.state]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(needle);
    });
  }, [mapData, search]);

  const warnings = mapData?.alerts || [];
  const layerStatus = mapData?.layerStatus || {};

  const handleIncidentSelect = useCallback(async (incident) => {
    setSelectedIncident(incident);
    setSelectedWarning(null);
    setDetailTab('prediction');
    setDrawerOpen(false);
    mapRef.current?.flyToIncident?.(incident);
    try {
      const [detailResponse, updatesResponse] = await Promise.allSettled([
        getIncident(incident.id, { bbox: WESTERN_CONUS_BBOX }),
        getIncidentUpdates(incident.id, { bbox: WESTERN_CONUS_BBOX }),
      ]);
      if (detailResponse.status === 'fulfilled') {
        setSelectedIncidentDetail(detailResponse.value?.data ?? detailResponse.value);
      }
      if (updatesResponse.status === 'fulfilled') {
        const payload = updatesResponse.value?.data ?? updatesResponse.value;
        setSelectedIncidentUpdates(payload?.updates || []);
      }
    } catch (error) {
      console.warn('incident detail fetch failed', error);
    }
  }, []);

  const handleWarningSelect = useCallback((warning) => {
    setSelectedWarning(warning);
    setSelectedIncident(null);
    setSelectedIncidentDetail(null);
    setDrawerOpen(false);
    mapRef.current?.flyToAlert?.(warning);
  }, []);

  const handleRunPrediction = useCallback(async (incident) => {
    setRunningPrediction(true);
    try {
      await mapRef.current?.runPredictionForIncident?.(incident);
    } finally {
      setRunningPrediction(false);
    }
  }, []);

  const handleToggleLayer = useCallback((id) => {
    setLayerVisibility((current) => {
      const next = { ...current, [id]: !current[id] };
      if (id === 'ndvi') {
        mapRef.current?.toggleNdvi?.();
      }
      return next;
    });
  }, []);

  const userInitials = ((user?.fullName || user?.name || 'Guest')
    .split(' ')
    .map((name) => name[0])
    .join('') || 'G')
    .slice(0, 2)
    .toUpperCase();

  const selectedDetail = selectedIncidentDetail
    ? { ...selectedIncidentDetail, updates: selectedIncidentUpdates.length ? selectedIncidentUpdates : selectedIncidentDetail.updates }
    : null;

  return (
    <div className="watch-map-shell">
      <header className="watch-topbar">
        <button className="hamburger-btn" type="button" onClick={() => setDrawerOpen((open) => !open)} aria-label="Open menu">
          <span />
          <span />
          <span />
        </button>
        <div className="watch-brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>IgnisAI</span>
        </div>
        <div className="topbar-search">
          <input
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setActiveDrawerTab('incidents');
              setDrawerOpen(true);
            }}
            placeholder="Search fires, counties, warnings"
            aria-label="Search fires, counties, warnings"
          />
        </div>
        <button className="status-pill" type="button" onClick={loadMapData}>
          {isFetching ? 'Refreshing...' : `Updated ${formatRelative(mapData?.updatedAt)}`}
        </button>
        <button className="icon-btn topbar-icon" type="button" onClick={() => setLegendOpen(true)} aria-label="Open legend">?</button>
        <div className="user-chip">
          <div>
            <strong>{user?.fullName || user?.name || 'Guest'}</strong>
            <span>{user?.email || 'guest@example.com'}</span>
          </div>
          <em>{userInitials}</em>
        </div>
      </header>

      <main className="watch-map-stage">
        <MapComponent
          ref={mapRef}
          brightnessFilter={brightness}
          confidenceFilter={confidence}
          mapStyle={mapStyle}
          onFiresUpdated={() => {}}
          setIsFetching={setIsFetching}
          userLocation={userLocation}
          range={range}
          incidents={mapData?.incidents || []}
          alerts={warnings}
          layerVisibility={layerVisibility}
          selectedIncident={selectedIncident}
          selectedAlert={selectedWarning}
          onIncidentSelect={handleIncidentSelect}
          onAlertSelect={handleWarningSelect}
          watchShell
        />

        <Drawer
          open={drawerOpen}
          activeTab={activeDrawerTab}
          setActiveTab={setActiveDrawerTab}
          incidents={incidents}
          warnings={warnings}
          selectedIncident={selectedIncident}
          selectedWarning={selectedWarning}
          onIncidentSelect={handleIncidentSelect}
          onWarningSelect={handleWarningSelect}
          layerVisibility={layerVisibility}
          layerStatus={layerStatus}
          onToggleLayer={handleToggleLayer}
          onOpenHistory={() => {
            setDrawerOpen(false);
            mapRef.current?.toggleHistoryPanel?.();
          }}
          search={search}
          setSearch={setSearch}
          onClose={() => setDrawerOpen(false)}
        />

        <div className="floating-map-controls right">
          <button
            type="button"
            onClick={() => {
              setActiveDrawerTab('layers');
              setDrawerOpen(true);
            }}
          >
            Layers
          </button>
          <button type="button" onClick={() => setLegendOpen(true)}>Legend</button>
        </div>

        <div className="floating-map-controls left">
          <button type="button" onClick={() => mapRef.current?.refreshWildfires?.()}>Refresh</button>
          <button type="button" onClick={() => setUserLocation(null)}>Clear Loc</button>
        </div>

        <LegendPanel open={legendOpen} onClose={() => setLegendOpen(false)} />

        <IncidentDetailPanel
          incident={selectedIncident}
          detail={selectedDetail}
          activeTab={detailTab}
          setActiveTab={setDetailTab}
          runningPrediction={runningPrediction}
          onRunPrediction={handleRunPrediction}
          onClose={() => {
            setSelectedIncident(null);
            setSelectedIncidentDetail(null);
            setSelectedIncidentUpdates([]);
          }}
        />

        <WarningDetailPanel
          warning={selectedWarning}
          onClose={() => setSelectedWarning(null)}
        />

        <div className="map-status-strip">
          <span>{formatCount(incidents.length)} incidents</span>
          <span>{formatCount(warnings.length)} fire weather alerts</span>
          <span>{sourceHealthLabel(layerStatus.hotspots)} FIRMS</span>
          <span>{sourceHealthLabel(layerStatus.alerts)} NWS</span>
        </div>
      </main>
    </div>
  );
};

export default AdvancedFireDashboard;
