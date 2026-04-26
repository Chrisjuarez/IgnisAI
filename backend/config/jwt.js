// backend/config/jwt.js
//
// Single source of truth for JWT configuration.
//
// Why this exists:
//   Previously two files (routes/auth.js and middleware/auth.js) each defined
//   `JWT_SECRET = process.env.JWT_SECRET || 'your-fallback-secret-key'`. That
//   default is catastrophic in production: if JWT_SECRET is unset, the server
//   silently signs tokens with a public, hard-coded string, and *every*
//   attacker who has read this repository can forge a valid token for any
//   user_id of their choosing.
//
// Behavior:
//   * In production (NODE_ENV === 'production') the module throws at require
//     time if JWT_SECRET is missing or shorter than 32 characters. The process
//     dies before listening, so a misconfigured deploy fails its health check
//     instead of running insecurely.
//   * In test (NODE_ENV === 'test') a deterministic test secret is supplied
//     so the existing Jest suite does not need to set the variable.
//   * In development the loader prints a loud warning if the secret is missing
//     and falls back to a randomly-generated secret for the current process,
//     which means tokens issued by `npm run dev` survive a restart only if
//     JWT_SECRET is set. This is intentional — we want devs to notice.
//
// Length rule:
//   HS256 keys < 256 bits (32 bytes) materially weaken the signature. We
//   reject anything shorter than 32 chars across all environments.

'use strict';

const crypto = require('crypto');

const NODE_ENV = process.env.NODE_ENV || 'development';
const MIN_SECRET_LENGTH = 32;
const TEST_SECRET = 'test-secret-key-for-testing-minimum-32-chars';

function loadSecret() {
  const fromEnv = process.env.JWT_SECRET;

  if (fromEnv && fromEnv.length >= MIN_SECRET_LENGTH) {
    return fromEnv;
  }

  if (NODE_ENV === 'production') {
    const why = !fromEnv
      ? 'JWT_SECRET is not set'
      : `JWT_SECRET is ${fromEnv.length} chars (minimum ${MIN_SECRET_LENGTH})`;
    // Throw at require time so the process exits before app.listen.
    throw new Error(
      `[config/jwt] Refusing to start: ${why}. ` +
        'Set a strong JWT_SECRET (>= 32 random chars, e.g. ' +
        '`node -e "console.log(crypto.randomBytes(48).toString(\'base64url\'))"`).'
    );
  }

  if (NODE_ENV === 'test') {
    return TEST_SECRET;
  }

  // Development fallback: generate a per-process secret and warn loudly so the
  // developer knows tokens won't survive a restart.
  const generated = crypto.randomBytes(48).toString('base64url');
  // eslint-disable-next-line no-console
  console.warn(
    '[config/jwt] JWT_SECRET is missing or too short. Generated an ephemeral ' +
      'secret for this process. Tokens will be invalidated on restart. ' +
      'Set JWT_SECRET in your local .env to make sessions persist.'
  );
  return generated;
}

const JWT_SECRET = loadSecret();
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

module.exports = Object.freeze({
  JWT_SECRET,
  JWT_EXPIRES_IN,
  MIN_SECRET_LENGTH,
});
