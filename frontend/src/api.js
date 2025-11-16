// frontend/src/api.js
import axios from 'axios';

const API_BASE =
  (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) ||
  process.env.REACT_APP_API_URL ||
  '/api';

const api = axios.create({ baseURL: API_BASE });

export const getWildfireData = (opts = {}) =>
  api.get('/wildfires', { params: opts });

export const predictFireSpread = async ({ lat, lng }) => {
  const { data } = await api.get('/predict-fire-spread/vector', {
    params: { lat, lon: lng, thr: 0.5 }
  });
  return data;
};

export default api;
