import React, { useRef, useState, useCallback } from 'react';
import MapComponent from './MapComponent';
import FireControls from './FireControls';
import { useAuth } from './auth/AuthContext';
import '../styles/dashboard.css';
const MAPBOX_TOKEN =
  (typeof window !== 'undefined' && window._env_?.REACT_APP_MAPBOX_TOKEN) ||
  process.env.REACT_APP_MAPBOX_TOKEN ||
  '';
const AdvancedFireDashboard = () => {
  const { user } = useAuth();

  // --- UI state shared between Map & Controls ---
  const mapRef = useRef(null); // (MapComponent exposes refresh/toggle via ref)
  const [isFetching, setIsFetching] = useState(false);
  const [fireCount, setFireCount]   = useState(null);

  const [brightness, setBrightness]   = useState('');        // '', 'Small'|'Moderate'|'Severe'|'Extreme'
  const [confidence, setConfidence]   = useState('');        // '', 'Low'|'Medium'|'High'|'Very High'
  const [mapStyle, setMapStyle]       = useState('mapbox://styles/mapbox/satellite-streets-v12');

  const [userLocation, setUserLocation] = useState(null);    // {lat, lng}
  const [range, setRange]               = useState(20);      // miles
  const [nearbyFires, setNearbyFires]   = useState([]);      // [{cityName, distance, brightnessCat, confidenceCat}...]

  // Called by FireControls
  const handleRefresh = useCallback(() => {
    mapRef.current?.refreshWildfires?.();
  }, []);

  // Called by MapComponent when it updates features
  const handleFiresUpdated = useCallback((count) => {
    setFireCount(count ?? 0);
  }, []);

  return (
    <div className="advanced-dashboard">
      {/* Simple header (keep your fancier header if you like) */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🔥</div>
            <h1 className="logo-text">IgnisAI</h1>
          </div>
          <div className="user-section">
            <div className="user-info">
              <span className="user-name">{user?.fullName || user?.name || 'Guest'}</span>
              <span className="user-email">{user?.email || 'guest@example.com'}</span>
            </div>
            <div className="user-avatar">
              {((user?.fullName || user?.name || 'GN').split(' ').map(n => n[0]).join(''))}
            </div>
          </div>
        </div>
      </header>

      <div className="dashboard-container">
        {/* LEFT: your real controls */}
        <aside className="dashboard-sidebar">
          <FireControls
            onRefresh={handleRefresh}
            isFetching={isFetching}
            fireCount={fireCount}

            onChangeBrightness={setBrightness}
            onChangeConfidence={setConfidence}
            onChangeMapStyle={setMapStyle}

            mapboxToken={MAPBOX_TOKEN}

            onSelectLocation={setUserLocation}
            range={range}
            onChangeRange={setRange}
            nearbyFires={nearbyFires}
          />
        </aside>

        {/* CENTER: your real map */}
        <main className="dashboard-main">
          <div className="map-container">
            <MapComponent
              ref={mapRef}
              brightnessFilter={brightness}
              confidenceFilter={confidence}
              mapStyle={mapStyle}

              // pass status callbacks so the count/loader work
              onFiresUpdated={handleFiresUpdated}
              setIsFetching={setIsFetching}

              // location & range
              userLocation={userLocation}
              range={range}
              onNearbyFiresUpdate={setNearbyFires}
            />
          </div>
        </main>

        {/* RIGHT: keep your details panel or keep it simple */}
        <aside className="fire-details-panel">
          <div className="fire-details-header">
            <h3 className="fire-details-name">Current View</h3>
          </div>
          <div className="risk-section">
            <h4 className="section-title">Summary</h4>
            <div className="risk-grid">
              <div className="risk-item"><div className="risk-label">Fires in view</div><div className="risk-value">{fireCount ?? '—'}</div></div>
              <div className="risk-item"><div className="risk-label">Brightness Filter</div><div className="risk-value">{brightness || 'All'}</div></div>
              <div className="risk-item"><div className="risk-label">Confidence Filter</div><div className="risk-value">{confidence || 'All'}</div></div>
            </div>
          </div>
          <div className="actions-section">
            <h4 className="section-title">Quick Actions</h4>
            <div className="action-buttons">
              <button className="action-btn primary" onClick={handleRefresh}>Refresh Now</button>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default AdvancedFireDashboard;