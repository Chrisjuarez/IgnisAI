// backend/utils/forecastErrorResponse.js
//
// What to send the client when a call to tilesvc fails: a status that says
// whose problem it is, and a message a person can act on. tilesvc's reason
// code travels along in `reason` for whoever is debugging.

const TIMEOUT_CODES = new Set(['ECONNABORTED', 'ETIMEDOUT']);

// tilesvc's reason codes for missing forecast inputs, by prefix, in words.
const UNAVAILABLE_INPUT_MESSAGES = [
  ['predictions_disabled', 'Forecasts are turned off on this server.'],
  ['no_recent_fire', 'No satellite has detected fire in this area in the last day, so there is nothing to forecast from.'],
  ['static_', 'Terrain and fuel data for this area could not be loaded.'],
  ['firms_', 'Satellite fire detections for this forecast could not be loaded.'],
  ['calibration_', 'The forecast model is not available right now.'],
  ['model_', 'The forecast model is not available right now.'],
];

const SERVICE_UNAVAILABLE = {
  status: 502,
  body: { error: 'forecast_service_unavailable', message: 'The forecast service is not responding. Try again shortly.' },
};

function inputsUnavailable(reason) {
  const known = UNAVAILABLE_INPUT_MESSAGES.find(([prefix]) => String(reason).startsWith(prefix));
  return {
    status: 422,
    body: {
      error: 'forecast_inputs_unavailable',
      reason,
      message: known ? known[1] : 'Ignis could not gather the inputs for this forecast.',
    },
  };
}

/** `{ status, body }` to send when a call to tilesvc throws. */
function forecastErrorResponse(err) {
  if (TIMEOUT_CODES.has(err.code)) {
    return { status: 504, body: { error: 'forecast_timeout', message: 'The forecast took too long. Try again in a minute.' } };
  }

  const tilesvcResponse = err.response;
  if (!tilesvcResponse) {
    // tilesvc did not answer, or answered with something the route rejected.
    return SERVICE_UNAVAILABLE;
  }
  if (tilesvcResponse.data?.error === 'input_unavailable') {
    return inputsUnavailable(tilesvcResponse.data.reason);
  }
  if (tilesvcResponse.status < 500) {
    return { status: 400, body: { error: 'forecast_request_invalid', message: 'Ignis could not run a forecast with these settings.' } };
  }
  return SERVICE_UNAVAILABLE;
}

module.exports = { forecastErrorResponse };
