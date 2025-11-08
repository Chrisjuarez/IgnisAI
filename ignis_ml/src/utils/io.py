from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

def write_cog(path, arr, affine: Affine, crs="EPSG:5070"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":"COG", "dtype":arr.dtype.name, "height":arr.shape[-2],
        "width":arr.shape[-1], "count":1, "crs":crs, "transform":affine,
        "compress":"LZW", "nodata":255
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(path)
