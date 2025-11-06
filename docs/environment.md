# Environment Configuration - IgnisAI

## Environment Overview

IgnisAI uses environment-specific configuration to manage secrets, API endpoints, and feature flags across development, testing, and production environments.

## Required Environment Variables

### Backend Environment (.env)

#### Core Configuration
```bash
# Server Configuration
NODE_ENV=development              # development|test|production
PORT=3001                        # API server port
HOST=localhost                   # Server host

# Database Configuration  
MONGODB_URI=mongodb://localhost:27017/ignisai
MONGODB_TEST_URI=mongodb://localhost:27017/ignisai_test
DB_MAX_CONNECTIONS=10
DB_TIMEOUT=5000                  # milliseconds

# Authentication & Security
JWT_SECRET=your-super-secure-jwt-secret-minimum-32-characters
JWT_EXPIRES_IN=7d                # 7 days, 24h, 30m, etc.
BCRYPT_ROUNDS=12                 # Password hashing rounds
API_RATE_LIMIT=100               # requests per windowMs
RATE_LIMIT_WINDOW=900000         # 15 minutes in milliseconds
```

#### External API Keys
```bash
# NASA FIRMS (Fire Information for Resource Management System)
FIRMS_API_KEY=your-nasa-firms-api-key-here
FIRMS_BASE_URL=https://firms.modaps.eosdis.nasa.gov/api/

# NOAA Weather API
NOAA_API_KEY=your-noaa-weather-api-key-here
NOAA_BASE_URL=https://api.weather.gov/

# USGS Topography API
USGS_API_KEY=your-usgs-api-key-optional
USGS_BASE_URL=https://apps.nationalmap.gov/services/
```

#### Service Configuration
```bash
# External API Timeouts
API_TIMEOUT=10000               # 10 seconds
RETRY_ATTEMPTS=3
RETRY_DELAY=1000               # milliseconds

# Caching
CACHE_TTL=300                  # 5 minutes
REDIS_URL=redis://localhost:6379  # Optional Redis cache

# Logging
LOG_LEVEL=info                 # error|warn|info|debug
LOG_FILE=logs/app.log
```

### Frontend Environment (.env)

```bash
# API Configuration
REACT_APP_API_URL=http://localhost:3001
REACT_APP_ENV=development

# MapBox Integration
REACT_APP_MAPBOX_ACCESS_TOKEN=pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJ5b3VyLWFjY2Vzcy10b2tlbiJ9
REACT_APP_MAPBOX_STYLE=mapbox://styles/mapbox/outdoors-v11

# Feature Flags
REACT_APP_ENABLE_ANALYTICS=false
REACT_APP_ENABLE_EXPORT=true
REACT_APP_DEBUG_MODE=true

# Build Configuration
GENERATE_SOURCEMAP=true         # development only
BUILD_PATH=build
```

## Environment-Specific Configurations

### Development Environment
```bash
# .env.development
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug
API_TIMEOUT=30000              # Longer timeout for debugging
CACHE_TTL=60                   # Shorter cache for development
REACT_APP_DEBUG_MODE=true
```

### Testing Environment  
```bash
# .env.test
NODE_ENV=test
MONGODB_URI=mongodb://localhost:27017/ignisai_test
JWT_SECRET=test-secret-key-for-ci-cd-very-long-and-secure
LOG_LEVEL=error                # Suppress logs during tests
API_TIMEOUT=5000               # Faster timeouts in tests
FIRMS_API_KEY=mock-key         # Use mocked responses
NOAA_API_KEY=mock-key
```

### Production Environment
```bash
# Production environment variables (GitHub Secrets)
NODE_ENV=production
PORT=3001
MONGODB_URI=${{ secrets.MONGODB_PRODUCTION_URI }}
JWT_SECRET=${{ secrets.JWT_SECRET_PRODUCTION }}
FIRMS_API_KEY=${{ secrets.FIRMS_API_KEY }}
NOAA_API_KEY=${{ secrets.NOAA_API_KEY }}
REACT_APP_MAPBOX_ACCESS_TOKEN=${{ secrets.MAPBOX_ACCESS_TOKEN }}

# Security hardening
LOG_LEVEL=warn
DEBUG=false
API_RATE_LIMIT=50              # Stricter rate limiting
CACHE_TTL=600                  # Longer cache in production
```

## Secret Management

### GitHub Actions Secrets
Store sensitive values as encrypted secrets in repository settings:

**Repository Secrets:**
- `MONGODB_PRODUCTION_URI`
- `JWT_SECRET_PRODUCTION` 
- `FIRMS_API_KEY`
- `NOAA_API_KEY`
- `MAPBOX_ACCESS_TOKEN`
- `CODECOV_TOKEN` (for coverage reporting)

**Organization Secrets (if applicable):**
- `DOCKER_REGISTRY_TOKEN`
- `SECURITY_SCAN_TOKEN`

### Secret Rotation Policy
- **JWT Secrets**: Rotate every 90 days
- **API Keys**: Monitor usage and rotate when needed
- **Database Credentials**: Rotate every 180 days
- **Access Tokens**: Follow provider recommendations

### Development Secrets
```bash
# Create .env from template
cp .env.example .env

# Generate secure JWT secret
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"

# Test database connection
mongosh $MONGODB_URI --eval "db.runCommand('ping')"
```

## Configuration Validation

### Startup Validation Script
```javascript
// config/validate.js
const requiredVars = {
  development: ['MONGODB_URI', 'JWT_SECRET'],
  test: ['MONGODB_TEST_URI', 'JWT_SECRET'], 
  production: ['MONGODB_URI', 'JWT_SECRET', 'FIRMS_API_KEY', 'NOAA_API_KEY']
};

const validateEnvironment = () => {
  const env = process.env.NODE_ENV || 'development';
  const required = requiredVars[env] || [];
  
  const missing = required.filter(varName => !process.env[varName]);
  
  if (missing.length > 0) {
    console.error(`❌ Missing required environment variables for ${env}:`);
    missing.forEach(varName => console.error(`   - ${varName}`));
    process.exit(1);
  }
  
  console.log(`✅ Environment configuration valid for ${env}`);
};

module.exports = { validateEnvironment };
```

### Environment Loading
```javascript
// Load at app startup
require('dotenv').config();
const { validateEnvironment } = require('./config/validate');

validateEnvironment();
```

## API Key Management

### Obtaining API Keys

#### NASA FIRMS API
1. Register at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Request FIRMS API access
3. Add key to environment: `FIRMS_API_KEY=your-key-here`

#### NOAA Weather API
1. Register at [Weather.gov API](https://www.weather.gov/documentation/services-web-api)
2. No key required for basic access, but rate limited
3. For higher limits, contact NOAA for API key

#### MapBox Access Token
1. Create account at [MapBox](https://account.mapbox.com/)
2. Generate access token with required scopes:
   - `styles:read`
   - `fonts:read` 
   - `datasets:read`
3. Add to environment: `REACT_APP_MAPBOX_ACCESS_TOKEN=pk.eyJ...`

### Usage Monitoring
```bash
# Check API usage and rate limits
curl -H "Authorization: Bearer $FIRMS_API_KEY" \
  "https://firms.modaps.eosdis.nasa.gov/api/usage"

# Monitor MapBox token usage
curl "https://api.mapbox.com/account/v1?access_token=$MAPBOX_TOKEN"
```

## Security Best Practices

### Environment File Security
```bash
# Never commit .env files
echo "*.env" >> .gitignore
echo "*.env.*" >> .gitignore
echo "!.env.example" >> .gitignore

# Secure file permissions
chmod 600 .env
chmod 644 .env.example
```

### Production Hardening
```bash
# Minimum required variables only
NODE_ENV=production
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/ignisai
JWT_SECRET=production-secret-very-long-and-random-string
PORT=3001

# Security headers
HELMET_ENABLED=true
CORS_ORIGIN=https://ignisai.yourdomain.com
SECURE_COOKIES=true
```

### Secret Scanning
CI pipeline includes secret detection:
- **TruffleHog**: Scans commits for accidentally exposed secrets
- **GitHub Secret Scanning**: Built-in detection for common patterns
- **Pre-commit Hooks**: Local validation before commit

## Troubleshooting

### Common Issues

**"JWT_SECRET is required"**
```bash
# Generate a secure secret
export JWT_SECRET=$(openssl rand -base64 64)
echo "JWT_SECRET=$JWT_SECRET" >> .env
```

**"MongoDB connection failed"**
```bash
# Check MongoDB status
brew services list | grep mongodb
mongosh --eval "db.adminCommand('ping')"

# Reset test database
mongosh ignisai_test --eval "db.dropDatabase()"
```

**"MapBox token invalid"**
- Verify token has correct scopes
- Check token hasn't expired
- Ensure REACT_APP_ prefix for frontend variables

**"API rate limit exceeded"**
- Check FIRMS/NOAA usage quotas
- Implement exponential backoff
- Cache responses to reduce API calls

### Environment Debugging
```javascript
// Add to app startup for debugging
console.log('Environment check:');
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('Port:', process.env.PORT);
console.log('MongoDB URI defined:', !!process.env.MONGODB_URI);
console.log('JWT Secret length:', process.env.JWT_SECRET?.length || 0);
console.log('API Keys configured:', {
  firms: !!process.env.FIRMS_API_KEY,
  noaa: !!process.env.NOAA_API_KEY
});
```

## Documentation Links

- [Environment Variables Best Practices](https://12factor.net/config)
- [GitHub Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [MongoDB Connection Strings](https://docs.mongodb.com/manual/reference/connection-string/)
- [MapBox Token Management](https://docs.mapbox.com/help/getting-started/access-tokens/)