import json

import pytest

from services.tilesvc import static_catalog
from services.tilesvc.static_catalog import InputUnavailable


def test_static_catalog_path_is_required(monkeypatch):
    monkeypatch.delenv("STATIC_CATALOG_PATH", raising=False)

    with pytest.raises(InputUnavailable, match="STATIC_CATALOG_PATH"):
        static_catalog.load_catalog()


def test_missing_optional_static_catalog_uses_placeholder_tensor(monkeypatch):
    monkeypatch.delenv("STATIC_CATALOG_PATH", raising=False)
    monkeypatch.delenv("STATIC_CATALOG_REQUIRED", raising=False)

    tensor, summary = static_catalog.load_static_tensor_for_model(
        None,
        ["elev", "slope", "aspect_cos", "water"],
    )

    assert tensor.shape == (4, static_catalog.SIZE, static_catalog.SIZE)
    assert summary["catalog"]["placeholder"] is True
    assert summary["channels"]["elev"]["placeholder_or_missing"] is True
    assert tensor[3].max() == 0.0


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

def _catalog_payload(version):
    return {
        "version": version,
        "channels": {
            name: {"uri": f"/tmp/{name}.tif"}
            for name in static_catalog.REQUIRED_BASE_STATIC
        },
    }


def test_load_catalog_reuses_parsed_catalog_until_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "static_catalog.json"
    path.write_text(json.dumps(_catalog_payload("v1")), encoding="utf-8")
    monkeypatch.setenv("STATIC_CATALOG_PATH", str(path))
    static_catalog._catalog_by_path.clear()

    first = static_catalog.load_catalog()

    assert static_catalog.load_catalog() is first, "unchanged catalog should not be re-parsed"

    path.write_text(json.dumps(_catalog_payload("v2-rebuilt")), encoding="utf-8")
    reloaded = static_catalog.load_catalog()

    assert reloaded is not first
    assert reloaded["version"] == "v2-rebuilt"
