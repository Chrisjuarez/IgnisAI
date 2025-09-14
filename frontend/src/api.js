//frontend/src/api.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/api',
});

export const getWildfireData = () => api.get('/wildfires');

export const predictFireSpread = async (fireData) => {
  try {
    const response = await api.post('/predict-fire-spread', fireData);
    return response.data;
  } catch (error) {
    console.error('Error predicting fire spread:', error);
    throw error;
  }
};