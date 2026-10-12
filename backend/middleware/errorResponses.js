// backend/middleware/errorResponses.js
//
// The last two handlers in the app: whatever no route answered, and whatever a
// route threw, go back as JSON. Express's defaults are HTML pages, and a thrown
// error's message can name internals.

const { logSafe } = require('../utils/logSafe');

function respondNotFound(req, res) {
  res.status(404).json({ error: 'not_found' });
}

// Express treats a middleware with four parameters as its error handler, so
// `_next` stays even though it is unused. `expose` marks an error whose message
// is safe to show the client; body-parser sets it, and so does the CORS check.
function respondWithError(err, req, res, _next) {
  const status = err.status || err.statusCode || 500;
  if (status >= 500) {
    console.error('Request failed:', req.method, logSafe(req.originalUrl), err);
    return res.status(status).json({ error: 'internal_error' });
  }
  return res.status(status).json({ error: 'request_rejected', message: err.expose ? err.message : undefined });
}

module.exports = { respondNotFound, respondWithError };
