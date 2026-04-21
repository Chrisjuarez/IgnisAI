import json

import pytest

from services.tilesvc import static_catalog
from services.tilesvc.static_catalog import InputUnavailable


def test_static_catalog_path_is_required(monkeypatch):
    monkeypatch.delenv("STATIC_CATALOG_PATH", raising=False)

    with pytest.raises(InputUnavailable, match="STATIC_CATALOG_PATH"):
        static_catalog.load_catalog()


def test_static_catalog_rejects_missing_required_channels(tmp_path, monkeypatch):
    path = tmp_path / "static_catalog.json"
    path.write_text(json.dumps({"channels": {"elev": {"uri": "/tmp/elev.tif"}}}), encoding="utf-8")
    monkeypatch.setenv("STATIC_CATALOG_PATH", str(path))

    with pytest.raises(InputUnavailable) as exc:
        static_catalog.load_catalog()

    assert exc.value.reason == "static_catalog_missing_channels"
    assert "ndvi" in exc.value.details["missing"]


def test_static_catalog_channel_requires_uri():
    with pytest.raises(InputUnavailable) as exc:
        static_catalog._parse_channel("elev", {"units": "m"})

    assert exc.value.reason == "static_catalog_invalid_channel"
