# backend/Dockerfile
FROM node:20-alpine
WORKDIR /app

# install prod deps first for better layer cache
COPY package*.json ./
RUN npm ci --omit=dev --no-audit --no-fund

# copy source
COPY . .

# runtime env
ENV NODE_ENV=production
ENV PORT=5000
ENV TILE_SVC=http://tilesvc:8008

# healthcheck hits /health (Express serves both /health and /api/health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:${PORT}/health" || exit 1

EXPOSE 5000
CMD ["node", "app.js"]
