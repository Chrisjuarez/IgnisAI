# ---- Build stage ----
FROM node:18-alpine AS build
WORKDIR /app

ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# Copy only deps first for better caching
COPY package*.json ./
RUN npm ci

# Copy the rest of the app
COPY . ./

# Build CRA
RUN npm run build

# ---- Runtime stage ----
FROM nginx:alpine

# Serve the built app with nginx
COPY --from=build /app/build /usr/share/nginx/html

# Optional (handy locally)
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]