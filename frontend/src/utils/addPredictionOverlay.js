// frontend/src/utils/addPredictionOverlay.js
// Usage:
//   import { addPredictionOverlay } from '../utils/addPredictionOverlay'
//   await addPredictionOverlay(map, { lat, lon, mode: 'raster'|'vector' })

export async function addPredictionOverlay(
  map,
  { lat, lon, mode = 'raster', sourceIdPrefix = 'ignis-pred' }
) {
  if (!map) throw new Error('mapbox-gl map instance is required');
  if (lat == null || lon == null) throw new Error('lat and lon are required');

  if (mode === 'raster') {
    const resp = await fetch(`/api/predict-fire-spread/raster?lat=${lat}&lon=${lon}`);
    if (!resp.ok) throw new Error(`raster request failed: ${resp.status}`);
    const { bounds, image_base64 } = await resp.json();
    const [w, s, e, n] = bounds;
    const url = `data:image/png;base64,${image_base64}`;

    const srcId = `${sourceIdPrefix}-image`;
    const layerId = `${sourceIdPrefix}-image-layer`;

    if (map.getLayer(layerId)) map.removeLayer(layerId);
    if (map.getSource(srcId)) map.removeSource(srcId);

    map.addSource(srcId, {
      type: 'image',
      url,
      coordinates: [[w, n], [e, n], [e, s], [w, s]]
    });

    map.addLayer({
      id: layerId,
      type: 'raster',
      source: srcId,
      paint: { 'raster-opacity': 0.65 }
    });
  } else {
    const resp = await fetch(`/api/predict-fire-spread/vector?lat=${lat}&lon=${lon}`);
    if (!resp.ok) throw new Error(`vector request failed: ${resp.status}`);
    const gj = await resp.json();

    const srcId = `${sourceIdPrefix}-poly`;
    const fillId = `${sourceIdPrefix}-poly-fill`;
    const lineId = `${sourceIdPrefix}-poly-line`;

    if (map.getLayer(fillId)) map.removeLayer(fillId);
    if (map.getLayer(lineId)) map.removeLayer(lineId);
    if (map.getSource(srcId)) map.removeSource(srcId);

    map.addSource(srcId, { type: 'geojson', data: gj });

    map.addLayer({
      id: fillId,
      type: 'fill',
      source: srcId,
      paint: { 'fill-opacity': 0.35 }
    });

    map.addLayer({
      id: lineId,
      type: 'line',
      source: srcId,
      paint: { 'line-width': 1.5 }
    });
  }
}