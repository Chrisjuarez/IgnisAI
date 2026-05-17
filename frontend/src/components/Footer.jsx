//ignis-ai-frontend/src/components/Footer.js
import React from 'react';


function Footer() {
  return (
    <footer style={styles.footer}>
      <p>© 2025 IgnisAI Team. Data courtesy of NASA FIRMS & NOAA.</p>
    </footer>
  );
}


const styles = {
  footer: {
    background: '#f1f1f1',
    color: '#333',
    padding: '0.5rem 1rem',
    textAlign: 'center'
  }
};


export default Footer;