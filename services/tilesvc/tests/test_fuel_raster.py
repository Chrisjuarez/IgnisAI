"""
A missing FBFM40 raster used to return barren everywhere, which made every
fuel-reading engine score zero on every fire and read as a real forecast of no
growth. These pin the behaviour that replaced it.
"""

import numpy as np
import pytest

from services.tilesvc import fuel_raster
from services.tilesvc.fuel_raster import (
    FUEL_RASTER_ENV,
    candidate_paths,
    fuel_codes_for_tile,
    fuel_raster_path,
)
from services.tilesvc.grid import lonlat_to_tile
from services.tilesvc.static_catalog import InputUnavailable


@pytest.fixture(autouse=True)
def isolated_lookup(tmp_path, monkeypatch):
    monkeypatch.delenv(FUEL_RASTER_ENV, raising=False)
    monkeypatch.setenv("IGNIS_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)


def test_a_missing_raster_raises_instead_of_reporting_barren():
    tile = lonlat_to_tile(-118.555, 34.078)
    with pytest.raises(InputUnavailable) as excinfo:
        fuel_codes_for_tile(tile)

    exc = excinfo.value
    assert exc.reason == "fuel_raster_missing"
    # The message has to be actionable: every earlier version of this failure
    # was silent, and a bare "not found" would send someone hunting.
    assert "fetch-fuel" in str(exc)
    assert exc.details["searched"]


def test_the_cache_root_is_searched_before_the_repo_data_dir():
    paths = candidate_paths()
    assert len(paths) == 2
    assert "cache" in str(paths[0]) and paths[0].name.endswith(".tif")
    assert paths[1].parts[:2] == ("data", "source-rasters")


def test_an_explicit_path_wins_over_every_fallback(tmp_path, monkeypatch):
    explicit = tmp_path / "somewhere-else.tif"
    explicit.write_bytes(b"")
    monkeypatch.setenv(FUEL_RASTER_ENV, str(explicit))
    assert candidate_paths() == [explicit]
    assert fuel_raster_path() == explicit


def test_a_blank_env_var_falls_back_rather_than_resolving_to_cwd(monkeypatch):
    monkeypatch.setenv(FUEL_RASTER_ENV, "   ")
    assert len(candidate_paths()) == 2


def test_codes_load_and_land_in_the_fbfm40_range(monkeypatch):
    """
    Guards the swap this module exists to prevent: fuel1/2/3 are latent
    channels, and reading those instead of the raster would produce values
    outside the FBFM40 vocabulary.
    """
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    from services.tilesvc.grid import SIZE

    path = fuel_raster.candidate_paths()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = lonlat_to_tile(-118.555, 34.078)
    from services.tilesvc.grid import tile_affine

    data = np.full((SIZE, SIZE), 142, dtype=np.int16)  # SH2, a real burnable model
    data[0, 0] = -9999  # LANDFIRE's negative nodata sentinel
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
        dtype="int16", crs="EPSG:5070", transform=tile_affine(tile),
    ) as dst:
        dst.write(data, 1)

    codes = fuel_codes_for_tile(tile)
    assert codes.dtype == np.int16
    assert codes[0, 0] == fuel_raster.NO_DATA_CODE, "negative sentinel must become barren"
    assert (codes == 142).sum() > 0
    # Every FBFM40 code is either a burnable model (101-204) or non-burnable
    # (91-99). Anything else means we read the wrong raster.
    valid = ((codes >= 101) & (codes <= 204)) | ((codes >= 91) & (codes <= 99))
    assert valid.all(), f"out-of-vocabulary codes: {np.unique(codes[~valid])}"
