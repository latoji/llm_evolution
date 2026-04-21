import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_TARGET =
  process.env["VITE_API_BASE"] ?? "http://localhost:8000";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ingest": { target: API_TARGET, changeOrigin: true },
      "/stats": { target: API_TARGET, changeOrigin: true },
      "/generate": { target: API_TARGET, changeOrigin: true },
      "/db": { target: API_TARGET, changeOrigin: true },
      "/ws": {
        target: API_TARGET.replace(/^http/, "ws"),
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
