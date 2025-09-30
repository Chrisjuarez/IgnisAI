# Multi-stage build for production
FROM node:18-alpine AS dependencies
WORKDIR /app
COPY backend/package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS build
WORKDIR /app
COPY backend/package*.json ./
RUN npm ci
COPY backend/ ./
RUN npm run build --if-present

FROM node:18-alpine AS production
WORKDIR /app
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
COPY --from=dependencies --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=build --chown=nodejs:nodejs /app ./
USER nodejs
EXPOSE 5000
CMD ["node", "app.js"]