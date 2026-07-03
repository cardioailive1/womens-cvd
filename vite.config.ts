import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The React client lives in src/client and builds to dist/public.
// In dev, /api is proxied to the Express server on :4000.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://localhost:4000' } },
  build: { outDir: 'dist/public', emptyOutDir: true, sourcemap: false },
});
