//ignis-ai-frontend/src/components/Header.js
import React from 'react';


function Header() {
  return (
    <header style={styles.header}>
      <div style={styles.logo}>IgnisAI</div>
      <nav style={styles.nav}>
        {/* <a href="#features" style={styles.link}>Features</a>
        <a href="#about" style={styles.link}>About</a> */}
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
    textDecoration: 'none'
  }
};


export default Header;