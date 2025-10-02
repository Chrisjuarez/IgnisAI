// frontend/src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../components/auth/AuthContext';

function Header() {
  const { isAuthenticated, logout } = useAuth();

  return (
    <header style={styles.header}>
      <div style={styles.logo}>IgnisAI</div>
      <nav style={styles.nav}>
        {isAuthenticated ? (
          <>
            <Link to="/dashboard" style={styles.link}>Dashboard</Link>
            <button onClick={logout} style={styles.button}>Logout</button>
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
    backgroundColor: '#420',
    color: '#fff',
    padding: '0.5rem 1rem',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  logo: {
    fontWeight: 'bold',
    fontSize: '1.2rem'
  },
  nav: {
    display: 'flex',
    gap: '1rem'
  },
  link: {
    color: '#fff',
    textDecoration: 'none',
    fontSize: '1rem'
  },
  button: {
    background: 'none',
    border: 'none',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '1rem'
  }
};

export default Header;