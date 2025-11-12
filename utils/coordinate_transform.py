# Coordinate transformation utility for the PHANTOM project

import numpy as np
import rasterio
from pyproj import Transformer

def transform_coordinates(x, y, src_crs, dst_crs):
    """Transform coordinates from one CRS to another."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)

def get_crs_from_geotiff(file_path):
    """Extract the Coordinate Reference System (CRS) from a GeoTIFF file."""
    with rasterio.open(file_path) as src:
        return src.crs

def transform_geotiff_coordinates(file_path, x, y, dst_crs):
    """Transform coordinates from a GeoTIFF's CRS to a specified destination CRS."""
    src_crs = get_crs_from_geotiff(file_path)
    return transform_coordinates(x, y, src_crs, dst_crs)