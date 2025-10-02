// frontend/src/components/auth/LoginForm.js
import React, { useState, useContext } from 'react';
import { AuthContext } from './AuthContext';
import { Link, useNavigate } from 'react-router-dom';
import '../../styles/auth.css';

const LoginForm = () => {
  const { login } = useContext(AuthContext);
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    remember: false
  });
  const [error, setError] = useState('');

  const handleChange = e => {
    const { name, type, value, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setError('');
    try {
      await login(formData.email, formData.password, formData.remember);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-logo">IgnisAI</h1>
          <p className="auth-subtitle">Wildfire Prediction & Analysis</p>
        </div>
        <form className="auth-form" onSubmit={handleSubmit}>
          {error && <div className="error-message">{error}</div>}
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              name="email"
              type="email"
              required
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              required
              placeholder="Enter your password"
              value={formData.password}
              onChange={handleChange}
            />
          </div>
          <div className="form-options">
            <label className="checkbox-label">
              <input
                name="remember"
                type="checkbox"
                checked={formData.remember}
                onChange={handleChange}
              />
              Remember me
            </label>
            <Link to="/forgot-password" className="forgot-link">
              Forgot password?
            </Link>
          </div>
          <button type="submit" className="auth-button">
            Sign In
          </button>
        </form>
        <div className="auth-footer">
          Don't have an account? <Link to="/register">Sign up</Link>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;