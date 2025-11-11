from dataclasses import dataclass
import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.geometry import Polygon
from affine import Affine

CONUS_EQD = "EPSG:5070"
WGS84 = "EPSG:4326"

@dataclass
class GridSpec:
    crs: str = CONUS_EQD
    res_m: int = 1000
    size: int = 64

def aoi_bounds_from_center(lat, lon, gs: GridSpec):
    tf = Transformer.from_crs(WGS84, gs.crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    half = (gs.res_m * gs.size) / 2
    return (x-half, y-half, x+half, y+half)

def affine_from_bounds(gs: GridSpec, bounds):
    xmin, ymin, xmax, ymax = bounds
    return Affine(gs.res_m, 0, xmin, 0, -gs.res_m, ymax)

def world_to_rowcol(lats, lons, gs: GridSpec, bounds):
    tf = Transformer.from_crs(WGS84, gs.crs, always_xy=True)
    xs, ys = tf.transform(lons, lats)
    xmin, ymin, xmax, ymax = bounds
    cols = ((xs - xmin) // gs.res_m).astype(int)
    rows = ((ymax - ys) // gs.res_m).astype(int)
    return rows, cols

def rasterize_points(lat, lon, val, gs: GridSpec, bounds, H, W, mode="binary"):
    r,c = world_to_rowcol(np.array(lat), np.array(lon), gs, bounds)
    arr = np.zeros((H,W), dtype=np.float32)
    ok = (r>=0)&(r<H)&(c>=0)&(c<W)
    if mode == "binary":
        arr[r[ok], c[ok]] = 1.0
    else:
        for rr,cc,v in zip(r[ok], c[ok], np.array(val)[ok]):
            arr[rr,cc] = max(arr[rr,cc], float(v))
    return arr

def polygon_from_mask(prob, gs: GridSpec, bounds, threshold=0.5, simplify=0):
    import numpy as np
    from skimage import measure
    xmin, ymin, xmax, ymax = bounds
    bw = (prob >= threshold).astype(np.uint8)
    contours = measure.find_contours(bw, 0.5)
    polys = []
    for pts in contours:
        r = pts[:,0]; c = pts[:,1]
        x = xmin + (c+0.5)*gs.res_m
        y = ymax - (r+0.5)*gs.res_m
        poly = Polygon(np.column_stack([x,y]))
        if poly.area > 0: polys.append(poly)
    if not polys: return None
    m = unary_union(polys).buffer(0)
    if simplify>0: m = m.simplify(simplify)
    # project to lat/lon
    tf = Transformer.from_crs(gs.crs, WGS84, always_xy=True)
    def reproj(pg):
        X,Y = tf.transform(*np.array(pg.exterior.coords).T)
        return Polygon(np.column_stack([X,Y]))
    if m.geom_type == "Polygon":
        return mapping(reproj(m))
    return mapping(unary_union([reproj(g) for g in m.geoms]))
