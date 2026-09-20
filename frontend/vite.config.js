import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { copyFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

// Static hosts serve files by path, so a client-side route like /dashboard has
// no file behind it and the host answers with its own 404 - which is what a
// reload hit. The blueprint declares a rewrite, but Render did not apply it to
// the already-created service, so this does not depend on that: most static
// hosts serve 404.html for an unmatched path, and if that file IS the app, the
// router boots and reads the URL as intended.
//
// Harmless if the rewrite is ever applied - the rewrite wins and this is never
// requested.
function spaFallback() {
  return {
    name: 'spa-404-fallback',
    closeBundle() {
      const dir = 'build';
      const index = resolve(dir, 'index.html');
      if (existsSync(index)) {
        copyFileSync(index, resolve(dir, '404.html'));
      }
    },
  };
}



export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react(), spaFallback()],
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
