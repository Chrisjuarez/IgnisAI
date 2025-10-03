import React, { useRef, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './components/auth/AuthContext';
import ProtectedRoute from './components/auth/ProtectedRoute';

import './App.css';
import Header from './components/Header';
import Footer from './components/Footer';
import MapComponent from './components/MapComponent';
import FireControls from './components/FireControls';

import LoginForm from './components/auth/LoginForm';
import RegisterForm from './components/auth/RegisterForm';
import ForgotPasswordForm from './components/auth/ForgotPasswordForm';

function DashboardLayout() {
  const mapRef = useRef(null);
  const [brightnessFilter, setBrightnessFilter] = useState('');
  const [confidenceFilter, setConfidenceFilter] = useState('');
  const [fireCount, setFireCount] = useState(null);
  const [isFetching, setIsFetching] = useState(false);
  const [mapStyle, setMapStyle] = useState('mapbox://styles/mapbox/streets-v12');
  const [userLocation, setUserLocation] = useState(null);
  const [range, setRange] = useState(20);
  const [nearbyFires, setNearbyFires] = useState([]);
  const [selectedFire, setSelectedFire] = useState(null);
  const [firePrediction, setFirePrediction] = useState(null);

  const handleRefresh = () => mapRef.current?.refreshWildfires?.();
  const handleFiresUpdated = count => setFireCount(count);
  const handleNearbyFiresUpdate = fires => setNearbyFires(fires);
  const handleFireSelect = fire => { setSelectedFire(fire); setFirePrediction(null); };
  const handleFireSpreadPrediction = prediction => setFirePrediction(prediction);

  return (
    <div style={styles.appContainer}>
      <Header />
      <div style={styles.mapContainer}>
        <MapComponent
          ref={mapRef}
          brightnessFilter={brightnessFilter}
          confidenceFilter={confidenceFilter}
          onFiresUpdated={handleFiresUpdated}
          setIsFetching={setIsFetching}
          mapStyle={mapStyle}
          userLocation={userLocation}
          range={range}
          onNearbyFiresUpdate={handleNearbyFiresUpdate}
          selectedFire={selectedFire}
          onFireSelect={handleFireSelect}
          firePrediction={firePrediction}
          onFireSpreadPrediction={handleFireSpreadPrediction}
        />
        <FireControls
          onRefresh={handleRefresh}
          isFetching={isFetching}
          fireCount={fireCount}
          onChangeBrightness={setBrightnessFilter}
          onChangeConfidence={setConfidenceFilter}
          onChangeMapStyle={setMapStyle}
          mapboxToken={process.env.REACT_APP_MAPBOX_TOKEN}
          onSelectLocation={setUserLocation}
          range={range}
          onChangeRange={setRange}
          nearbyFires={nearbyFires}
          selectedFire={selectedFire}
        />
      </div>
      <Footer />
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public auth routes */}
          <Route path="/login" element={<LoginForm />} />
          <Route path="/register" element={<RegisterForm />} />
          <Route path="/forgot-password" element={<ForgotPasswordForm />} />

          {/* Protected dashboard */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <DashboardLayout />
              </ProtectedRoute>
            }
          />

          {/* Redirect root to dashboard */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

const styles = {
  appContainer: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh'
  },
  mapContainer: {
    position: 'relative',
    flex: 1,
    overflow: 'hidden'
  }
};

export default App;