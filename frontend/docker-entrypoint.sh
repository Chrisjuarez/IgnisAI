#!/bin/sh
set -e

# Defaults for runtime env
: "${REACT_APP_API_URL:=/api}"

# 1) Write /usr/share/nginx/html/env.js from container env
cat >/usr/share/nginx/html/env.js <<EOF
window._env_ = {
  REACT_APP_MAPBOX_TOKEN: '${REACT_APP_MAPBOX_TOKEN}',
  REACT_APP_API_URL: '${REACT_APP_API_URL}'
};
EOF

# 2) Write a tiny shim that forces mapboxgl.accessToken at runtime
cat >/usr/share/nginx/html/config-patch.js <<'JS'
(function () {
  try {
    if (window._env_ && window._env_.REACT_APP_MAPBOX_TOKEN && window.mapboxgl) {
      window.mapboxgl.accessToken = window._env_.REACT_APP_MAPBOX_TOKEN;
    }
  } catch (_) {}
})();
JS

# 3) Ensure both scripts load BEFORE main.*.js (idempotent)
f=/usr/share/nginx/html/index.html
# remove any previous tags to avoid dupes
sed -i 's#<script src="/env.js"></script>##g' "$f"
sed -i 's#<script src="/config-patch.js"></script>##g' "$f"
# insert right before the CRA main bundle
sed -E -i 's#(<script[^>]+src="/static/js/main\.[^"]+\.js"[^>]*></script>)#<script src="/env.js"></script>\n<script src="/config-patch.js"></script>\n\1#' "$f"

exec "$@"