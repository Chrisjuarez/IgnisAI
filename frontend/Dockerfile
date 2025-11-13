# ==== Build stage ====
FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ==== Runtime (NGINX) ====
FROM nginx:1.27-alpine

# Copy built app
COPY --from=build /app/build /usr/share/nginx/html
# Copy NGINX proxy config
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Copy runtime entrypoint (creates /env.js and injects before main bundle)
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]