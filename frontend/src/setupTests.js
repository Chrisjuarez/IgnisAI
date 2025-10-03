// frontend/src/setupTests.js
import React from 'react';
import '@testing-library/jest-dom';
import { TextDecoder, TextEncoder } from 'util';

// Polyfill TextDecoder/TextEncoder for jsdom
global.TextDecoder = TextDecoder;
global.TextEncoder = TextEncoder;

// Mock AuthContext to provide hook and provider
jest.mock('./components/auth/AuthContext', () => {
  const React = require('react');
  return {
    useAuth: () => ({
      user: null,
      isAuthenticated: false,
      loading: false,
      login: jest.fn(),
      register: jest.fn(),
      logout: jest.fn(),
      forgotPassword: jest.fn()
    }),
    AuthProvider: ({ children }) => <div data-testid="auth-provider">{children}</div>
  };
});

// Mock axios
jest.mock('axios', () => ({
  defaults: { baseURL: '', headers: { common: {} } },
  post: jest.fn(() => Promise.resolve({ data: {} })),
  get: jest.fn(() => Promise.resolve({ data: {} })),
  put: jest.fn(() => Promise.resolve({ data: {} })),
  delete: jest.fn(() => Promise.resolve({ data: {} }))
}));

// Mock mapbox-gl
jest.mock('mapbox-gl', () => ({
  Map: function () {
    this.on = jest.fn();
    this.remove = jest.fn();
    this.addLayer = jest.fn();
  },
  NavigationControl: jest.fn(),
  FullscreenControl: jest.fn(),
  Marker: jest.fn()
}));