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
jest.mock('axios', () => {
  const resolvedPayload = { data: { data: [] } };
  const createInstance = () => ({
    post: jest.fn(() => Promise.resolve(resolvedPayload)),
    get: jest.fn(() => Promise.resolve(resolvedPayload)),
    put: jest.fn(() => Promise.resolve(resolvedPayload)),
    delete: jest.fn(() => Promise.resolve(resolvedPayload))
  });

  const defaultInstance = createInstance();
  return {
    defaults: { baseURL: '', headers: { common: {} } },
    ...defaultInstance,
    create: jest.fn(() => createInstance())
  };
});

// Mock mapbox-gl with lightweight map + navigation control so tests can
// exercise zoom/interaction logic without a real WebGL context.
jest.mock('mapbox-gl', () => {
  const doc = typeof globalThis.document !== 'undefined' ? globalThis.document : null;
  const createElement = tag => {
    if (doc) return doc.createElement(tag);
    return {
      tagName: tag,
      children: [],
      setAttribute: () => {},
      addEventListener: () => {},
      appendChild(child) { this.children.push(child); }
    };
  };
  const mockMaps = [];

  class MockSource {
    constructor(config = {}) {
      this.config = config;
      this.setData = jest.fn(data => {
        this.config.data = data;
      });
    }
  }

  class MockMap {
    constructor(options = {}) {
      this.options = options;
      this.container = options.container || createElement('div');
      this._zoom = options.zoom ?? 0;
      this._style = options.style;
      this.sources = {};
      this.layers = {};
      this.eventHandlers = {};
      this.onceHandlers = {};
      this.controls = [];
      this.setLayoutProperty = jest.fn();
      this.setPaintProperty = jest.fn();
      this.setCenter = jest.fn();
      this.setZoom = jest.fn(z => { this._zoom = z; });
      this.getZoom = jest.fn(() => this._zoom);
      this.zoomIn = jest.fn(() => { this._zoom += 1; });
      this.zoomOut = jest.fn(() => { this._zoom -= 1; });
      this.remove = jest.fn();
    }

    addControl(control) {
      this.controls.push(control);
      if (typeof control?.onAdd === 'function') {
        const el = control.onAdd(this);
        if (el && this.container?.appendChild) {
          this.container.appendChild(el);
        }
      }
      return control;
    }

    on(event, layerOrHandler, maybeHandler) {
      const hasLayer = typeof layerOrHandler === 'string';
      const handler = hasLayer ? maybeHandler : layerOrHandler;
      if (!handler) return;
      if (!this.eventHandlers[event]) this.eventHandlers[event] = [];
      this.eventHandlers[event].push({ handler, layerId: hasLayer ? layerOrHandler : null });
      if (event === 'load' && !hasLayer) {
        // simulate async load completion
        setTimeout(() => handler({ target: this }), 0);
      }
    }

    off(event, layerId, handler) {
      if (!this.eventHandlers[event]) return;
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => !(h.layerId === layerId && h.handler === handler));
    }

    once(event, handler) {
      this.onceHandlers[event] = handler;
    }

    trigger(event, payload) {
      (this.eventHandlers[event] || []).forEach(({ handler }) => handler(payload));
    }

    addSource(id, config) {
      this.sources[id] = new MockSource(config);
    }

    getSource(id) {
      return this.sources[id];
    }

    removeSource(id) {
      delete this.sources[id];
    }

    addLayer(layer) {
      this.layers[layer.id] = layer;
    }

    getLayer(id) {
      return this.layers[id];
    }

    removeLayer(id) {
      delete this.layers[id];
    }

    setStyle(style) {
      this._style = style;
      if (this.onceHandlers.styledata) {
        const cb = this.onceHandlers.styledata;
        delete this.onceHandlers.styledata;
        setTimeout(() => cb({ target: this }), 0);
      }
    }

    getStyle() {
      return { layers: Object.values(this.layers) };
    }

  }

  class MockNavigationControl {
    onAdd(map) {
      this.map = map;
      const container = createElement('div');
      container.className = 'mapboxgl-ctrl mapboxgl-ctrl-group';

      const zoomInBtn = createElement('button');
      zoomInBtn.type = 'button';
      zoomInBtn.textContent = '+';
      zoomInBtn.setAttribute('aria-label', 'Zoom in');
      zoomInBtn.addEventListener('click', () => map.zoomIn());

      const zoomOutBtn = createElement('button');
      zoomOutBtn.type = 'button';
      zoomOutBtn.textContent = '-';
      zoomOutBtn.setAttribute('aria-label', 'Zoom out');
      zoomOutBtn.addEventListener('click', () => map.zoomOut());

      container.appendChild(zoomInBtn);
      container.appendChild(zoomOutBtn);
      return container;
    }

    onRemove() {
      this.map = null;
    }
  }

  class MockPopup {
    constructor() {
      this.setLngLat = jest.fn(() => this);
      this.setHTML = jest.fn(() => this);
      this.addTo = jest.fn(() => this);
      this.remove = jest.fn();
    }
  }

  function Map(options) {
    const instance = new MockMap(options);
    mockMaps.push(instance);
    return instance;
  }

  function NavigationControl() {
    return new MockNavigationControl();
  }

  const mockModule = {
    Map,
    NavigationControl,
    Popup: jest.fn(() => new MockPopup()),
    Marker: jest.fn(() => ({
      setLngLat: jest.fn(() => ({ addTo: jest.fn().mockReturnThis() })),
      addTo: jest.fn()
    })),
    FullscreenControl: jest.fn(() => ({})),
    __mockMaps: mockMaps
  };

  mockModule.accessToken = '';
  return mockModule;
});
