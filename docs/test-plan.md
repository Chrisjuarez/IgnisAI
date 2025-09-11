# Test Plan (IgnisAI)

> Aligns with course rubric (Test Plans, Test Case Dev, Test Results, Bugs/Fixes, CICD Tests/Automation).

## 1. Scope & Objectives
- In scope: backend routes (wildfires, weather, ndvi, topography), frontend components (Map, FireControls), ML stub endpoints.
- Out of scope: full production deployment.

## 2. Strategy & Levels
- Unit (Jest/Vitest), Integration (Supertest), E2E smoke (Playwright/Cypress), Negative tests, Performance smoke, Security checks (CodeQL/npm audit).

## 3. Environments & Data
- Local, CI. Use lightweight fixtures and mock external APIs.

## 4. Tooling
- ESLint/Prettier, Jest/Vitest, React Testing Library, Supertest, Cypress/Playwright, NYC coverage, GitHub Actions.

## 5. Entry/Exit Criteria
- Entry: PR opened with tests. Exit: all checks pass, coverage gate met, reviewer approval.

## 6. Traceability
| Story | Criteria | Test ID | Type | Automated |
|------|----------|---------|------|-----------|
| US-1 Map shows wildfires | markers rendered from API | TC-API-001, TC-FE-001 | API/FE | Yes |

## 7. Sample Test Cases
- TC-API-001 `/api/wildfires` 200 + schema
- TC-NEG-001 invalid params → 400
- TC-FE-001 NDVI toggle shows overlay legend
- TC-E2E-001 user loads map, toggles layer
- TC-PERF-001 `/api/wildfires` p95 < 500ms (smoke)

## 8. Coverage Targets
- Sprint 2: BE ≥60%, FE ≥50%; Sprint 3: BE ≥70%, FE ≥60%

## 9. Reporting & Metrics
- CI artifacts: junit, coverage HTML; dashboard of pass rate, coverage, defects, MTTR.

## 10. Defect Policy
- Add failing test before/with each fix; root cause in PR description.

## 11. Risks
- External API instability; mitigate with mocks and contract tests.

## 12. Maintenance
- Review flaky tests weekly; update tests with spec changes.
