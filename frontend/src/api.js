// frontend/src/api.js
import axios from 'axios';

// Prefer build-time CRA var, then optional runtime window._env_, then a sane default
const API_BASE =
  process.env.REACT_APP_API_URL ||
  (typeof window !== 'undefined' && window._env_ && window._env_.REACT_APP_API_URL) ||
  'http://localhost:5000/api';

const api = axios.create({ baseURL: API_BASE });


export const getWildfireData = () => api.get('/wildfires');

export const predictFireSpread = async (fireData) => {
  const { data } = await api.post('/predict-fire-spread', fireData);
  return data;
};
export default api;           