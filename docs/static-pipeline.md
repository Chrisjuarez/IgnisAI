# Static COG Pipeline

IgnisAI uses a static catalog to tell `tilesvc` where to read the production static rasters used by the v3 model. The catalog is small JSON; the actual rasters are Cloud Optimized GeoTIFFs in S3.

## What It Builds

The first production target is `western_conus` in `EPSG:5070` at `500 m` resolution. The pipeline writes these base channels:

`elev`, `ndvi`, `bi`, `erc`, `pdsi`, `chili`, `impervious`, `water`, `population`, `fuel1`, `fuel2`, `fuel3`

`tilesvc` derives `slope`, `aspect_cos`, and `aspect_sin` from `elev` at request time, giving the model its expected 15 static channels.

## Configure Sources

Copy the example source config and replace the `REPLACE_*` source URIs:

```bash
cp config/static_sources.western_conus.example.json config/static_sources.western_conus.json
```

The source config supports:

- `opentopography_globaldem` for DEM fetches from OpenTopography.
- `raster` for already-published source rasters/COGs readable by GDAL/rasterio.
- `nlcd_water` for deriving a percent water mask from NLCD land-cover class `11`.
- `landfire_fuel_component` for candidate `fuel1`, `fuel2`, and `fuel3` transforms from LANDFIRE FBFM40.

Required environment:

```bash
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export IGNIS_STATIC_BUCKET=your-ignis-static-bucket
export IGNIS_STATIC_PREFIX=ignis/static
export OPENTOPO_API_KEY=...
```

NASA Earthdata credentials are needed when the source rasters are pulled from Earthdata before being passed to this pipeline:

```bash
export EARTHDATA_USERNAME=...
export EARTHDATA_PASSWORD=...
```

## Dry Run

```bash
python -m services.static_pipeline build \
  --extent western_conus \
  --version 20260421 \
  --bucket s3://$IGNIS_STATIC_BUCKET/$IGNIS_STATIC_PREFIX \
  --source-config config/static_sources.western_conus.json \
  --dry-run
```

## Build And Upload

```bash
python -m services.static_pipeline build \
  --extent western_conus \
  --version 20260421 \
  --bucket s3://$IGNIS_STATIC_BUCKET/$IGNIS_STATIC_PREFIX \
  --source-config config/static_sources.western_conus.json \
  --catalog-out config/static_catalog.production.json
```

Output layout:

```text
s3://<bucket>/ignis/static/western_conus/v1/<version>/<channel>.tif
s3://<bucket>/ignis/static/western_conus/v1/<version>/static_catalog.production.json
```

Commit `config/static_catalog.production.json` after a successful upload. Do not commit generated `.tif` files.

## Runtime

Render tilesvc should use:

```bash
STATIC_CATALOG_PATH=/app/config/static_catalog.production.json
STATIC_CATALOG_REQUIRED=0
```

Keep `STATIC_CATALOG_REQUIRED=0` until `/healthz` shows the catalog is valid and `/input_audit` shows no missing placeholder channels. Then switch it to `1`.

`fuel1`, `fuel2`, and `fuel3` are marked as candidate LANDFIRE-derived channels. Strict production should remain blocked until the static parity audit passes against representative training tiles.
