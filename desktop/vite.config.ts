import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

const backend = process.env.FLY_BACKEND || "http://127.0.0.1:8790";

export default defineConfig({
  root: path.resolve(__dirname, "ui"),
  base: "/ui/",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "ui/src") } },
  build: { outDir: path.resolve(__dirname, "ui/dist"), emptyOutDir: true },
  server: {
    port: 8081,
    proxy: {
      "/api": backend,
      "/media": backend,
    },
  },
});
