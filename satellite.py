import ee
import pandas as pd

_initialized = False

def init_ee(project=None):
    global _initialized
    if not _initialized:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        _initialized = True

def get_satellite_features(lat, lon, plot_id, date_range=("2026-05-01", "2026-08-01")):
    point = ee.Geometry.Point([lon, lat])

    # Sentinel-2 surface reflectance, cloud-filtered
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(point)
          .filterDate(*date_range)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .median())

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Soil moisture from SMAP
    smap = (ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
            .filterBounds(point)
            .filterDate(*date_range)
            .select("ssm")
            .mean())

    # Land surface temperature from MODIS
    lst = (ee.ImageCollection("MODIS/061/MOD11A1")
           .filterBounds(point)
           .filterDate(*date_range)
           .select("LST_Day_1km")
           .mean())

    combined = ndvi.addBands(smap).addBands(lst)
    result = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()

    return {
        "plot_id": plot_id,
        "lat": lat,
        "lon": lon,
        "ndvi": result.get("NDVI"),
        "soil_moisture": result.get("ssm"),
        "land_surface_temp_k": result.get("LST_Day_1km", 0) * 0.02 if result.get("LST_Day_1km") else None,
    }

import math

def build_region(lat, lon, width_m, height_m):
    
    lat_offset = (height_m / 2) / 111320
    lon_offset = (width_m / 2) / (111320 * math.cos(math.radians(lat)))
    return ee.Geometry.Rectangle([lon - lon_offset, lat - lat_offset, lon + lon_offset, lat + lat_offset])


def get_truecolor_map_url(lat, lon, width_m=750, height_m=750, date_range=("2026-05-01", "2026-08-01")):
    
    region = build_region(lat, lon, width_m, height_m)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(*date_range)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .median())

    vis_params = {
        "region": region,
        "dimensions": 512,
        "bands": ["B4", "B3", "B2"],  # true-color RGB bands
        "min": 0,
        "max": 3000,
    }
    return s2.getThumbURL(vis_params)


def get_ndvi_map_url(lat, lon, width_m=750, height_m=750, date_range=("2026-05-01", "2026-08-01")):
    
    region = build_region(lat, lon, width_m, height_m)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(*date_range)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .median())
    ndvi = s2.normalizedDifference(["B8", "B4"])

    vis_params = {
        "region": region,
        "dimensions": 512,
        "min": -0.2,
        "max": 0.8,
        "palette": ["red", "orange", "yellow", "yellowgreen", "green"],
    }
    return ndvi.getThumbURL(vis_params)


if __name__ == "__main__":
    init_ee(project="your-project-id-here")  # swap in the actual Earth Engine project ID
    
    plots = [
        {"plot_id": "plot_1", "lat": 37.4275, "lon": -122.1697},  # swap in your actual test plot coords
    ]
    rows = []
    for p in plots:
        row = get_satellite_features(p["lat"], p["lon"], p["plot_id"])
        row["ndvi_map_url"] = get_ndvi_map_url(p["lat"], p["lon"])
        rows.append(row)

    pd.DataFrame(rows).to_csv("satellite_features.csv", index=False)
    print("wrote satellite_features.csv")