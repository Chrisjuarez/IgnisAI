// backend/utils/logSafe.js
//
// A value from a request, made safe to put in a log line: without line breaks
// it cannot forge extra log entries.

function logSafe(value) {
  return String(value).replace(/[\r\n]/g, '');
}

module.exports = { logSafe };
