import json

import numpy as np
import rasterio

from services.static_pipeline.pipeline import BuildOptions, build_static_pipeline, extent_for_tile
from services.tilesvc.grid import lonlat_to_tile
from services.tilesvc.static_builder import CHANNEL_ORDER
from services.tilesvc.static_catalog import load_static_tensor_for_model


def _constant_sources(tmp_path):
    values = {
        "elev": 500.0,
        "ndvi": 0.42,
        "bi": 80.0,
        "erc": 45.0,
        "pdsi": -1.0,
        "chili": 120.0,
        "impervious": 12.0,
        "water": 0.0,
        "population": 150.0,
        "fuel1": 0.75,
        "fuel2": -0.25,
        "fuel3": -2.0,
    }
    channels = {}
    for name, value in values.items():
        channels[name] = {
            "type": "constant",
            "value": value,
            "units": "test",
            "valid_range": [-10000, 10000],
            "resampling": "bilinear",
            "source": {"name": "unit test"},
        }
    channels["water"]["valid_range"] = [0, 100]
    channels["fuel1"]["quality"] = "candidate"
    channels["fuel1"]["parity_status"] = "pending_training_static_audit"
    channels["fuel2"]["quality"] = "candidate"
    channels["fuel2"]["parity_status"] = "pending_training_static_audit"
    channels["fuel3"]["quality"] = "candidate"
    channels["fuel3"]["parity_status"] = "pending_training_static_audit"
    path = tmp_path / "static_sources.json"
    path.write_text(json.dumps({"channels": channels}), encoding="utf-8")
    return path


def test_static_pipeline_builds_local_catalog_and_tilesvc_can_load_it(tmp_path, monkeypatch):
    tile = lonlat_to_tile(-118.55, 34.05)
    extent = extent_for_tile(tile)
    source_config = _constant_sources(tmp_path)
    catalog_path = tmp_path / "static_catalog.production.json"

    result = build_static_pipeline(
        BuildOptions(
            extent=extent.name,
            version="test",
            source_config=source_config,
            work_dir=tmp_path / "work",
            catalog_out=catalog_path,
            upload=False,
            catalog_uri_mode="local",
            custom_extent=extent,
        )
    )

    assert result["ok"] is True
    assert catalog_path.exists()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert sorted(catalog["channels"]) == sorted(
        ["elev", "ndvi", "bi", "erc", "pdsi", "chili", "impervious", "water", "population", "fuel1", "fuel2", "fuel3"]
    )
    assert catalog["channels"]["fuel1"]["quality"] == "candidate"

    elev_path = catalog["channels"]["elev"]["uri"]
    with rasterio.open(elev_path) as src:
        assert str(src.crs) == "EPSG:5070"
        assert src.nodata == -9999.0
        assert src.tags()["ignis_channel"] == "elev"
        assert src.width > 0
        assert src.height > 0

    monkeypatch.setenv("STATIC_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("STATIC_CATALOG_REQUIRED", "1")
    stat, summary = load_static_tensor_for_model(tile, CHANNEL_ORDER)

    assert stat.shape == (15, 64, 64)
    assert np.isfinite(stat).all()
    assert summary["channels"]["fuel1"]["quality"] == "candidate"
    assert summary["catalog"]["fuel_channel_status"]["fuel1"]["parity_status"] == "pending_training_static_audit"


def test_static_pipeline_dry_run_rejects_missing_bucket_for_s3_catalog(tmp_path):
    source_config = _constant_sources(tmp_path)
    tile = lonlat_to_tile(-118.55, 34.05)
    extent = extent_for_tile(tile)

    try:
        build_static_pipeline(
            BuildOptions(
                extent=extent.name,
                version="test",
                source_config=source_config,
                work_dir=tmp_path / "work",
                upload=False,
                dry_run=True,
                catalog_uri_mode="s3",
                custom_extent=extent,
            )
        )
    except ValueError as exc:
        assert "bucket_uri is required" in str(exc)
    else:
        raise AssertionError("expected missing bucket ValueError")
