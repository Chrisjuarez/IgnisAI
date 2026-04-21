# Operations Runbook - IgnisAI 

## Service Overview

IgnisAI consists of three main services:
- **Backend API** (Node.js/Express on port 3001)
- **Frontend Web App** (React on port 3000)  
- **Tilesvc ML API** (FastAPI on port 8008)
- **Database** (MongoDB on port 27017)

## Local Development

### Starting Services

```bash
# Option 1: Docker Compose (Recommended)
docker-compose up --build

# Option 2: Manual startup
# Terminal 1: MongoDB
mongod --dbpath /usr/local/var/mongodb

# Terminal 2: Backend
cd backend
npm install
npm run dev

# Terminal 3: Frontend  
cd frontend
npm install
npm start
```

### Health Checks

```bash
# Backend API health
curl http://localhost:3001/health
# Expected: {"status": "OK", "timestamp": "...", "uptime": "..."}

# Frontend accessibility
curl -I http://localhost:3000
# Expected: HTTP/1.1 200 OK

# MongoDB connection
mongosh --eval "db.runCommand('ping')"
# Expected: { ok: 1 }

# Tilesvc model/input health
curl http://localhost:8008/healthz
# Expected: modelExists=true, staticCatalog.ok=true, calibration.ok=true

# Tilesvc Prometheus-style metrics
curl http://localhost:8008/metrics
```

## Environment Configuration

### Required Environment Variables

**Backend (.env)**
```bash
NODE_ENV=development
PORT=3001
MONGODB_URI=mongodb://localhost:27017/ignisai
JWT_SECRET=your-super-secure-jwt-secret-key-here
JWT_EXPIRES_IN=7d
API_RATE_LIMIT=100
FIRMS_API_KEY=your-nasa-firms-api-key
NOAA_API_KEY=your-noaa-weather-api-key
```

**Frontend (.env)**
```bash
REACT_APP_API_URL=http://localhost:3001
REACT_APP_MAPBOX_TOKEN=your-mapbox-access-token
REACT_APP_ENV=development
```

### Secret Management
- Development: `.env` files (never committed)
- Production: GitHub Actions Encrypted Secrets
- CI/CD: Environment-specific secret stores

## Common Issues & Solutions

### Backend Won't Start
**Problem**: "EADDRINUSE: address already in use :::3001"
```bash
# Find and kill process on port 3001
lsof -ti:3001 | xargs kill -9
# Or use different port
PORT=3002 npm run dev
```

**Problem**: "MongoNetworkError: failed to connect to server"
```bash
# Check MongoDB status
brew services list | grep mongodb
# Start MongoDB if stopped
brew services start mongodb-community
# Or use Docker
docker run -d -p 27017:27017 mongo:6.0
```

### Frontend Issues
**Problem**: "Module not found" errors
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Problem**: MapBox map not rendering
- Verify REACT_APP_MAPBOX_TOKEN is set correctly
- Check browser console for API key errors
- Ensure token has required scopes

### Database Issues
**Problem**: Connection timeout in tests
```bash
# Ensure test database is clean
mongosh ignisai_test --eval "db.dropDatabase()"
# Restart MongoDB service
brew services restart mongodb-community
```

## Production Deployment

### Docker Images
Images are automatically built and pushed to GHCR on main branch:
- `ghcr.io/chrisjuarez/ignisai-backend:latest`
- `ghcr.io/chrisjuarez/ignisai-frontend:latest`

### Deployment Steps
1. Code merged to main branch
2. CI pipeline runs tests and security scans
3. Docker images built and pushed to registry
4. Deploy using docker-compose or Kubernetes

### Rollback Procedure
```bash
# Identify previous working image tag
docker images | grep ignisai

# Update docker-compose to previous tag
version: '3.8'
services:
  backend:
    image: ghcr.io/chrisjuarez/ignisai-backend:sha-abcd123
  frontend:  
    image: ghcr.io/chrisjuarez/ignisai-frontend:sha-abcd123

# Redeploy
docker-compose up -d
```

### Prediction Rollback

If NOAA, FIRMS snapshots, static rasters, calibration, or the model release are unhealthy, disable model calls without taking the map down:

```bash
PREDICTIONS_ENABLED=false docker-compose up -d tilesvc backend
```

The backend returns `predictions_disabled` for prediction routes while observed FIRMS/perimeter layers remain available.

### Model Release Procedure

1. Upload `convlstm_unet_v3_delta_Cd13_Cs15_H64_T6_nautilus.pt` to GitHub Releases.
2. Compute the SHA256 of the release asset and set `MODEL_SHA256`.
3. Set `MODEL_URL` to the exact release asset URL.
4. Keep `MODEL_CONFIG_PATH`, `STATIC_CATALOG_PATH`, and `CALIBRATION_PATH` pinned to artifacts built for the same v3 model.
5. Verify `/healthz` exposes the resolved `modelSha256`, ML package source commit, static catalog version, and calibration status.

## Monitoring & Alerting

### Application Metrics
- **Uptime**: Service availability monitoring
- **Response Times**: API endpoint performance
- **Error Rates**: 4xx/5xx response tracking
- **Resource Usage**: CPU, memory, disk utilization

### Database Monitoring
- **Connection Pool**: Active/idle connections
- **Query Performance**: Slow query identification
- **Storage**: Database size and growth trends
- **Replica Health**: If using MongoDB replica sets

### Log Locations
```bash
# Docker containers
docker logs ignisai-backend
docker logs ignisai-frontend

# Local development
# Backend: Console output + winston logs
# Frontend: Browser console + network tab

# Production: Container orchestrator logs
```

## Security Operations

### Incident Response
1. **Identify**: Monitor alerts, user reports
2. **Contain**: Isolate affected services
3. **Investigate**: Check logs, identify root cause  
4. **Resolve**: Apply fix, test in staging
5. **Document**: Update runbook, prevent recurrence

### Backup & Recovery
```bash
# MongoDB backup
mongodump --host localhost:27017 --db ignisai --out /path/to/backup

# Restore from backup
mongorestore --host localhost:27017 --db ignisai /path/to/backup/ignisai
```

## Performance Optimization

### Backend Optimization
- **Database Indexing**: Ensure indexes on frequently queried fields
- **Caching**: Implement Redis for API response caching
- **Connection Pooling**: Configure MongoDB connection limits

### Frontend Optimization  
- **Bundle Analysis**: Use webpack-bundle-analyzer
- **Code Splitting**: Implement route-based splitting
- **Asset Optimization**: Compress images, minify CSS/JS

## Contact & Support

### On-Call Rotation
- **Primary**: Christian Juarez
- **Backup**: Dylan Nguyen
