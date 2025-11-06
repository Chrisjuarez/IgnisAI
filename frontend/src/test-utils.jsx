// src/test-utils.jsx
import React from 'react';
import { render } from '@testing-library/react';
import { AuthProvider } from './components/auth/AuthContext';

export function renderWithProviders(ui, options = {}) {
  function Wrapper({ children }) {
    return (
      <AuthProvider>
        {children}
      </AuthProvider>
    );
  }
  return render(ui, { wrapper: Wrapper, ...options });
}

export * from '@testing-library/react';