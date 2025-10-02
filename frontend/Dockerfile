# Build stage
FROM node:18-alpine AS build
WORKDIR /app

# Copy package files from frontend folder
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code from frontend folder
COPY frontend ./

# Build app
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html