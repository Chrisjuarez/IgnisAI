// frontend/src/components/Header.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from './auth/AuthContext';

function Header() {
  const { isAuthenticated, logout, user } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  return (
    <header style={styles.header}>
      <div style={styles.logo}>
        <span style={styles.logoIcon}>🔥</span>
        <span style={styles.logoText}>IgnisAI</span>
      </div>
      <nav style={styles.nav}>
        {isAuthenticated ? (
          <>
            <Link to="/dashboard" style={styles.link}>Dashboard</Link>
            <Link to="/dashboard/legacy" style={styles.link}>Legacy View</Link>
            <div style={styles.userSection}>
              <button 
                onClick={() => setShowUserMenu(!showUserMenu)}
                style={styles.userButton}
              >
                {user?.name || 'User'} ▼
              </button>
              {showUserMenu && (
                <div style={styles.dropdown}>
                  <button onClick={logout} style={styles.dropdownItem}>
                    Logout
                  </button>
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <Link to="/login" style={styles.link}>Login</Link>
            <Link to="/register" style={styles.link}>Register</Link>
          </>
        )}
      </nav>
    </header>
  );
}

const styles = {
  header: {
    display: 'flex',
    backgroundColor: '#1a1a1a',
    color: '#fff',
    padding: '1rem 2rem',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid #333',
    boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)'
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  logoIcon: {
    fontSize: '1.5rem',
    animation: 'fireGlow 2s ease-in-out infinite alternate'
  },
  logoText: {
    fontWeight: 'bold',
    fontSize: '1.5rem',
    background: 'linear-gradient(135deg, #ff4500, #ff6b35)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text'
  },
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: '1.5rem'
  },
  link: {
    color: '#fff',
    textDecoration: 'none',
    fontSize: '1rem',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    transition: 'all 0.3s ease'
  },
  userSection: {
    position: 'relative'
  },
  userButton: {
    background: 'linear-gradient(135deg, #ff4500, #ff6b35)',
    color: 'white',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem'
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    right: '0',
    background: '#2a2a2a',
    border: '1px solid #333',
    borderRadius: '8px',
    marginTop: '0.5rem',
    minWidth: '120px',
    zIndex: 1000
  },
  dropdownItem: {
    width: '100%',
    background: 'none',
    border: 'none',
    color: '#fff',
    padding: '0.75rem 1rem',
    cursor: 'pointer',
    fontSize: '0.9rem',
    textAlign: 'left'
  }
};

export default Header;