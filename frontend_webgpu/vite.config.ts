import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import fs from 'fs'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Serve ../music folder as /music/* static files
    {
      name: 'serve-music',
      configureServer(server) {
        const musicDir = path.resolve(__dirname, '../music');
        // API endpoint: list all .mp3 files
        server.middlewares.use('/api/music-list', (_req, res) => {
          try {
            const files = fs.readdirSync(musicDir)
              .filter(f => /\.(mp3|wav|ogg|m4a|flac)$/i.test(f))
              .sort();
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(files));
          } catch {
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify([]));
          }
        });
        // Serve individual music files
        server.middlewares.use('/music', (req, res, next) => {
          const filePath = path.join(musicDir, decodeURIComponent(req.url || ''));
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.setHeader('Content-Type', 'audio/mpeg');
            fs.createReadStream(filePath).pipe(res);
          } else {
            next();
          }
        });
      },
    },
  ],
})
