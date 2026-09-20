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

import numpy as np


def _channel(name):
    return static_catalog.StaticChannel(name=name, uri=f"/tmp/{name}.tif")


def _all_zero():
    return np.zeros((static_catalog.SIZE, static_catalog.SIZE), dtype=np.float32)


def _varied():
    return np.linspace(1.0, 100.0, static_catalog.SIZE ** 2, dtype=np.float32).reshape(
        static_catalog.SIZE, static_catalog.SIZE
    )


def test_wilderness_tiles_are_not_treated_as_broken_rasters():
    # No roads, nobody living there, no standing water. All correct values for
    # backcountry, and previously each one refused the prediction outright.
    for name in ("impervious", "population", "water"):
        stats = static_catalog._validate_channel(name, _all_zero(), _channel(name))
        assert stats["pct_zero"] == 1.0


def test_an_empty_raster_is_still_caught_for_channels_that_cannot_be_zero():
    # Elevation is never uniformly zero over a real 32 km tile, so all-zero
    # here still means the static build produced nothing.
    for name in ("elev", "ndvi", "bi", "erc", "chili"):
        with pytest.raises(InputUnavailable) as exc:
            static_catalog._validate_channel(name, _all_zero(), _channel(name))
        assert exc.value.reason == "static_channel_placeholder"


def test_sparse_channels_with_real_values_still_validate():
    stats = static_catalog._validate_channel("impervious", _varied(), _channel("impervious"))

    assert stats["pct_zero"] == 0.0
    assert stats["finite_ratio"] == 1.0


def test_the_exempt_set_is_only_presence_channels():
    # Guard against someone widening this until the check stops protecting
    # anything. These three measure presence; absence is a real reading.
    assert static_catalog.SPARSE_STATIC_CHANNELS == {"water", "impervious", "population"}
