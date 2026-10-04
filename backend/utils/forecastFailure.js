// backend/utils/forecastFailure.js
//
// How a failed tilesvc call reaches the client: a status that says whose
// problem it is, and a message a person can act on. tilesvc's reason code
// travels along in `reason` for whoever is debugging.

const MESSAGES = {
  timeout: 'The forecast took too long. Try again in a minute.',
  serviceDown: 'The forecast service is not responding. Try again shortly.',
  badRequest: 'Ignis could not run a forecast with these settings.',
  inputs: 'Ignis could not gather the inputs for this forecast.',
};

// tilesvc reason codes, by prefix, in words.
const UNAVAILABLE_INPUTS = [
  ['predictions_disabled', 'Forecasts are turned off on this server.'],
  ['static_', 'Terrain and fuel data for this area could not be loaded.'],
  ['firms_', 'Satellite fire detections for this forecast could not be loaded.'],
  ['calibration_', 'The forecast model is not available right now.'],
  ['model_', 'The forecast model is not available right now.'],
];

function unavailableInputMessage(reason = '') {
  const match = UNAVAILABLE_INPUTS.find(([prefix]) => reason.startsWith(prefix));
  return match ? match[1] : MESSAGES.inputs;
}

/** `{ status, body }` for a tilesvc call that threw. */
function forecastFailure(err) {
  if (err?.code === 'ECONNABORTED' || err?.code === 'ETIMEDOUT') {
    return { status: 504, body: { error: 'forecast_timeout', message: MESSAGES.timeout } };
  }
  const upstream = err?.response;
  if (upstream?.data?.error === 'input_unavailable') {
    const { reason } = upstream.data;
    return { status: 422, body: { error: 'forecast_inputs_unavailable', reason, message: unavailableInputMessage(reason) } };
  }
  if (upstream?.status >= 400 && upstream.status < 500) {
    return { status: 400, body: { error: 'forecast_request_invalid', message: MESSAGES.badRequest } };
  }
  return { status: 502, body: { error: 'forecast_service_unavailable', message: MESSAGES.serviceDown } };
}

module.exports = { forecastFailure };
