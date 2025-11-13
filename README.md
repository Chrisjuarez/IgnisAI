# IgnisAI - Wildfire Risk Assessment Platform

[![CI Status](https://github.com/Chrisjuarez/IgnisAI/workflows/Node%20CI/badge.svg)](https://github.com/Chrisjuarez/IgnisAI/actions)
[![Backend Coverage](https://img.shields.io/badge/backend_coverage-75%25-green)](https://github.com/Chrisjuarez/IgnisAI/actions)
[![Frontend Coverage](https://img.shields.io/badge/frontend_coverage-65%25-yellow)](https://github.com/Chrisjuarez/IgnisAI/actions)

An interactive web-based Geographic Information System (GIS) platform that utilizes artificial intelligence and machine learning to predict wildfire risk in near real-time.

## Architecture

- **Backend**: Node.js/Express API with MongoDB
- **Frontend**: React.js with MapBox GL JS
- **AI/ML**: Python models for risk prediction
- **Data Sources**: NASA FIRMS API, NOAA Weather API

## Quick Start

### Prerequisites
- Node.js 18+
- MongoDB 6.0+
- Docker & Docker Compose

### Development Setup
```bash
# Clone repository
git clone https://github.com/Chrisjuarez/IgnisAI.git
cd IgnisAI

# Backend setup
cd backend
npm install
npm run dev

# Frontend setup (new terminal)
cd ../frontend
npm install
npm start
```

### Using Docker
```bash
docker-compose up --build
```

## Quality Gates

✅ **Backend Coverage**: Minimum 70% statements/branches  
✅ **Frontend Coverage**: Minimum 60% statements/branches  
✅ **Security**: CodeQL analysis, Trivy vulnerability scanning  
✅ **Code Quality**: ESLint, automated formatting  
✅ **Testing**: Unit, Integration, E2E test suites  

## Documentation

- [📋 Testing Plan](docs/test-plan.md)
- [🚀 Operations Runbook](docs/runbook.md) 
- [🔧 Environment Setup](docs/environment.md)
- [📊 Test Results](docs/results.md)
- [🎯 Project Management](docs/project.md)

## Team

- **Christian Juarez**
- **Dylan Nguyen**
- **Travis Nguyen**  
- **Emmanuel Montoya**

## License 

MIT License - see [LICENSE](LICENSE) for details.