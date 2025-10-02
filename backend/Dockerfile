# Build stage
FROM node:18-alpine AS build
WORKDIR /app

# Copy package files from backend folder
COPY backend/package*.json ./

# Install all dependencies
RUN npm install

# Copy source code from backend folder
COPY backend ./

# Run build if it exists
RUN npm run build --if-present || echo "No build script found"

# Production stage
FROM node:18-alpine AS production
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

# Copy package files and install production deps only
COPY backend/package*.json ./
RUN npm install --omit=dev

# Copy built app from build stage
COPY --from=build --chown=nodejs:nodejs /app ./

# Switch to non-root user
USER nodejs

EXPOSE 5000
CMD ["node", "app.js"]