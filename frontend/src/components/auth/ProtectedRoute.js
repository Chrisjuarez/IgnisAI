// frontend/src/components/auth/ProtectedRoute.js
import React, { useContext } from 'react';
import { useAuth } from './AuthContext';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useContext(useAuth);

  if (loading) {
    return <div>Loading...</div>;
  }
  return user ? children : <Navigate to="/login" replace />;
};

export default ProtectedRoute;