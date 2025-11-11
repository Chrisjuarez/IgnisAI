# services/tilesvc/cache.py
import os
import io
import numpy as np
import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# These should match your buckets (create in same region)
BUCKET_STATIC  = os.getenv("IGNIS_STATIC_BUCKET", "ignis-static")
BUCKET_DYNAMIC = os.getenv("IGNIS_DYNAMIC_BUCKET", "ignis-dynamic")

S3 = boto3.client("s3", region_name=AWS_REGION)


def s3_key_static(ix: int, iy: int) -> str:
    """Static per-tile cache (elevation, slope, aspect)."""
    return f"5070/64km/static/{ix}_{iy}.npz"


def s3_key_dynamic(date_str: str, ix: int, iy: int, T: int) -> str:
    """Dynamic snapshot (optional pre-warmed cache)."""
    return f"5070/64km/{date_str}/{ix}_{iy}_T{T}.npz"


def load_npz_s3(bucket: str, key: str) -> dict:
    """Load a .npz from S3 into a dict of arrays."""
    b = io.BytesIO()
    S3.download_fileobj(bucket, key, b)
    b.seek(0)
    return dict(np.load(b))


def save_npz_s3(bucket: str, key: str, **arrays):
    """Save arrays to a compressed .npz in S3."""
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    b.seek(0)
    S3.upload_fileobj(b, bucket, key, ExtraArgs={"ContentType": "application/octet-stream"})