import pytest

pytest.importorskip("fastapi")

from services.tilesvc import app


def test_noaa_health_dir_uses_template_parent_when_cache_dir_unset(monkeypatch):
    monkeypatch.delenv("NOAA_GRID_CACHE_DIR", raising=False)
    monkeypatch.setenv("NOAA_GRID_CACHE_TEMPLATE", "/data/noaa_grid_cache/{iso_hour}_{lat}_{lon}.npz")

    assert app._noaa_cache_health_dir() == "/data/noaa_grid_cache"
