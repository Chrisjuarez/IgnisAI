import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    define: {
      'process.env.REACT_APP_API_URL': JSON.stringify(
        env.REACT_APP_API_URL || env.VITE_API_URL || ''
      ),
      'process.env.REACT_APP_MAPBOX_TOKEN': JSON.stringify(
        env.REACT_APP_MAPBOX_TOKEN || env.VITE_MAPBOX_TOKEN || ''
      ),
    },
    server: {
      host: '0.0.0.0',
      port: 3000,
    },
    build: {
      outDir: 'build',
    },
  };
});
