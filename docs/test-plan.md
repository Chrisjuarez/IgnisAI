# Test Plan (IgnisAI)

> Aligns with course rubric (Test Plans, Test Case Dev, Test Results, Bugs/Fixes, CICD Tests/Automation).

## 1. Scope & Objectives
- In scope: backend routes (wildfires, weather, ndvi, topography), frontend components (Map, FireControls), tilesvc v3 model serving contracts, input builders, quality metadata, and rollback behavior.
- Out of scope: manual validation of commercial data licenses.

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
- TC-ML-001 NDWS tensor parity: `NpzTileDataset` vs `/input_audit?include_npz=true` NPZ, `np.allclose` on `x_dyn [6,13,64,64]` and `x_stat [15,64,64]`
- TC-ML-002 checkpoint metadata mismatch refuses startup/inference
- TC-ML-003 delta output contract reconstructs `p_next_fire = observed_fire OR thresholded p_new_burn`
- TC-ML-004 static catalog rejects missing channels, nodata-heavy rasters, and all-zero placeholders
- TC-ML-005 Open-Meteo fallback marks `quality.status=degraded`
- TC-FE-002 forecast panel displays advisory-risk copy and layer controls for observed, new-burn, and next-fire context

## 8. Coverage Targets
- Sprint 2: BE ≥60%, FE ≥50%; Sprint 3: BE ≥70%, FE ≥60%

## 9. Reporting & Metrics
- CI artifacts: junit, coverage HTML; dashboard of pass rate, coverage, defects, MTTR.

## 10. Defect Policy
- Add failing test before/with each fix; root cause in PR description.

## 11. Risks
- External API instability; mitigate with mocks, contract tests, NOAA/FIRMS freshness health checks, and `PREDICTIONS_ENABLED=false` rollback.
- Silent channel-order or normalization drift; mitigate with TC-ML-001 as a CI gate. CI must set `IGNIS_STRICT_PARITY=1`, `IGNIS_PARITY_NDWS_TILE`, and `IGNIS_PARITY_AUDIT_NPZ`.

## 12. Maintenance
- Review flaky tests weekly; update tests with spec changes.
