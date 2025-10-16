// frontend/src/components/AdvancedFireDashboard.js
import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from './auth/AuthContext';
import '../styles/dashboard.css';

const AdvancedFireDashboard = () => {
  const { user } = useAuth();
  const [selectedFireId, setSelectedFireId] = useState(1);
  const [filters, setFilters] = useState({
    brightness: 'all',
    confidence: 'all',
    mapStyle: 'satellite',
    ndviOverlay: false,
    nearbyRange: 20
  });

  // Mock data for fire incidents
  const activeFires = [
    {
      id: 1,
      name: "Riverside Fire",
      location: "California",
      riskLevel: "High",
      spreadProbability: 87,
      coordinates: [34.0522, -118.2437]
    },
    {
      id: 2,
      name: "Mountain Ridge Fire",
      location: "Colorado",
      riskLevel: "Moderate",
      spreadProbability: 45,
      coordinates: [39.7392, -104.9903]
    },
    {
      id: 3,
      name: "Pine Valley Fire",
      location: "Oregon",
      riskLevel: "Low",
      spreadProbability: 12,
      coordinates: [45.5152, -122.6784]
    }
  ];

  const weatherData = {
    wind: "25 km/h E",
    temperature: "32°C",
    humidity: "15%"
  };

  const riskAssessment = {
    fireWeatherIndex: "Extreme",
    fuelMoisture: "8%",
    terrainRisk: "High"
  };

  const selectedFire = activeFires.find(fire => fire.id === selectedFireId);

  const getRiskColorClass = (riskLevel) => {
    switch (riskLevel.toLowerCase()) {
      case 'high': return 'risk-high';
      case 'moderate': return 'risk-moderate';
      case 'low': return 'risk-low';
      default: return '';
    }
  };

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };

  return (
    <div className="advanced-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🔥</div>
            <h1 className="logo-text">IgnisAI</h1>
          </div>
          <div className="user-section">
            <div className="user-info">
              <span className="user-name">{user?.fullName || user?.name || 'Travis Nguyen'}</span>
              <span className="user-email">{user?.email || 'travis@gmail.com'}</span>
            </div>
            <div className="user-avatar">
              {((user?.fullName || user?.name || 'Travis Nguyen').split(' ').map(n => n[0]).join(''))}
            </div>
          </div>
        </div>
      </header>

      <div className="dashboard-container">
        {/* Left Sidebar */}
        <aside className="dashboard-sidebar">
          <div className="control-panel">
            <h3 className="panel-title">Wildfire Controls</h3>
            
            {/* Brightness Filter */}
            <div className="control-group">
              <label className="control-label">Brightness Filter</label>
              <div className="filter-buttons">
                {['all', 'high', 'medium', 'low'].map(filter => (
                  <button
                    key={filter}
                    className={`filter-btn ${filters.brightness === filter ? 'active' : ''}`}
                    onClick={() => handleFilterChange('brightness', filter)}
                  >
                    {filter === 'all' ? 'All' : 
                     filter === 'high' ? 'High &gt;80%' :
                     filter === 'medium' ? 'Medium 50-80%' : 'Low &lt;50%'}
                  </button>
                ))}
              </div>
            </div>

            {/* Confidence Filter */}
            <div className="control-group">
              <label className="control-label">Confidence Filter</label>
              <select 
                className="form-control dark-select"
                value={filters.confidence}
                onChange={(e) => handleFilterChange('confidence', e.target.value)}
              >
                <option value="all">All Confidence Levels</option>
                <option value="high">High &gt;80%</option>
                <option value="medium">Medium 50-80%</option>
                <option value="low">Low &lt;50%</option>
              </select>
            </div>

            {/* Map Style */}
            <div className="control-group">
              <label className="control-label">Map Style</label>
              <select 
                className="form-control dark-select"
                value={filters.mapStyle}
                onChange={(e) => handleFilterChange('mapStyle', e.target.value)}
              >
                <option value="satellite">Satellite</option>
                <option value="terrain">Terrain</option>
                <option value="streets">Streets</option>
              </select>
            </div>

            {/* NDVI Overlay */}
            <div className="control-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={filters.ndviOverlay}
                  onChange={(e) => handleFilterChange('ndviOverlay', e.target.checked)}
                />
                NDVI Overlay
              </label>
            </div>

            {/* Location Search */}
            <div className="control-group">
              <label className="control-label">Find Location</label>
              <input 
                type="text" 
                className="form-control dark-input" 
                placeholder="Search for location..."
              />
            </div>

            {/* Range Slider */}
            <div className="control-group">
              <label className="control-label">Nearby Fire Range</label>
              <div className="range-container">
                <input
                  type="range"
                  min="5"
                  max="100"
                  value={filters.nearbyRange}
                  className="range-slider"
                  onChange={(e) => handleFilterChange('nearbyRange', e.target.value)}
                />
                <span className="range-value">{filters.nearbyRange} miles</span>
              </div>
            </div>
          </div>

          {/* Active Fires List */}
          <div className="fires-list-panel">
            <h3 className="panel-title">Active Fires</h3>
            <div className="fires-list">
              {activeFires.map(fire => (
                <div
                  key={fire.id}
                  className={`fire-item ${selectedFireId === fire.id ? 'selected' : ''}`}
                  onClick={() => setSelectedFireId(fire.id)}
                >
                  <div className="fire-header">
                    <h4 className="fire-name">{fire.name}</h4>
                    <span className={`risk-badge ${getRiskColorClass(fire.riskLevel)}`}>
                      {fire.riskLevel === 'High' ? 'HIGH RISK' : fire.riskLevel.toUpperCase()}
                    </span>
                  </div>
                  <div className="fire-location">{fire.location}</div>
                  <div className="fire-probability">
                    <span className="probability-value">{fire.spreadProbability}%</span>
                    <span className="probability-label">spread probability</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="dashboard-main">
          {/* Map Area */}
          <div className="map-container">
            <div className="map-placeholder">
              <div className="map-overlay">
                <h3>Interactive Fire Map</h3>
                <p>Map Style: {filters.mapStyle}</p>
                <p>Range: {filters.nearbyRange} miles</p>
              </div>
              {/* Fire markers simulation */}
              <div className="fire-markers">
                {activeFires.map(fire => (
                  <div
                    key={fire.id}
                    className={`fire-marker ${getRiskColorClass(fire.riskLevel)} ${selectedFireId === fire.id ? 'selected' : ''}`}
                    style={{
                      left: `${25 + fire.id * 25}%`,
                      top: `${35 + fire.id * 20}%`
                    }}
                    onClick={() => setSelectedFireId(fire.id)}
                    title={fire.name}
                  >
                    🔥
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>

        {/* Right Panel - Fire Details */}
        <aside className="fire-details-panel">
          {selectedFire && (
            <>
              <div className="fire-details-header">
                <h3 className="fire-details-name">{selectedFire.name}</h3>
                <span className={`risk-badge large ${getRiskColorClass(selectedFire.riskLevel)}`}>
                  {selectedFire.riskLevel === 'High' ? 'HIGH RISK' : selectedFire.riskLevel.toUpperCase()}
                </span>
              </div>

              <div className="fire-prediction-section">
                <h4 className="section-title">Fire Spread Prediction</h4>
                <div className="prediction-warning">
                  ⚠️ This fire is <span className="highlight-text">likely</span> to spread significantly.
                </div>
                <div className="spread-probability">
                  <span className="probability-large">{selectedFire.spreadProbability}%</span>
                  <span className="probability-subtitle">Spread Probability</span>
                </div>
              </div>

              {/* Weather Conditions */}
              <div className="weather-section">
                <h4 className="section-title">Current Conditions</h4>
                <div className="weather-grid">
                  <div className="weather-item">
                    <div className="weather-label">💨 Wind</div>
                    <div className="weather-value">25<br/>km/h E</div>
                  </div>
                  <div className="weather-item">
                    <div className="weather-label">🌡️ Temperature</div>
                    <div className="weather-value">32°C</div>
                  </div>
                  <div className="weather-item">
                    <div className="weather-label">💧 Humidity</div>
                    <div className="weather-value">15%</div>
                  </div>
                </div>
                <div className="data-source">Data source: Real-time weather</div>
              </div>

              {/* Risk Assessment */}
              <div className="risk-section">
                <h4 className="section-title">Risk Assessment</h4>
                <div className="risk-grid">
                  <div className="risk-item">
                    <div className="risk-label">Fire Weather Index</div>
                    <div className="risk-value extreme">{riskAssessment.fireWeatherIndex}</div>
                  </div>
                  <div className="risk-item">
                    <div className="risk-label">Fuel Moisture</div>
                    <div className="risk-value">{riskAssessment.fuelMoisture}</div>
                  </div>
                  <div className="risk-item">
                    <div className="risk-label">Terrain Risk</div>
                    <div className="risk-value high">{riskAssessment.terrainRisk}</div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="actions-section">
                <h4 className="section-title">Quick Actions</h4>
                <div className="action-buttons">
                  <button className="action-btn primary">Generate Report</button>
                  <button className="action-btn secondary">Share Alert</button>
                  <button className="action-btn secondary">View History</button>
                </div>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
};

export default AdvancedFireDashboard;