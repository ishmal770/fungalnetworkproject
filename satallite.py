import ee
import pandas as pd

ee.Initialize(project='projectcoesus493716')
# initializing the earth engine api

#pulling plot 
def get_satellite_features(lat, lon, plot_id, date_range=("2026-05-01", "2026-08-01")):
    point = ee.Geometry.Point([lon, lat]) # creating earth engine location from cordinates ( longitudes, latitudes)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") #gets sentienel 2 satellite dataset
        .filterBounds(point) #keeps images in our plot locations
        .filterDate(*date_range) #.keeps images with relevant dates
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) # gets rid of images with a 20% cloud presence for better readings 
        .median() # removes noise and outliers on image by finding median of image pixels
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI") #calculates difference of vegetation using s2 bands (near-infared and red) by using (NIR-Red) / (NIR + Red) to measure plant health

    smap = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture") #used to measure moisture density 
        .filterBounds(point)
        .filterDate(*date_range)
        .select("ssm")
        .mean()
    )

    lst = (
        ee.ImageCollection("MODIS/061/MOD11A1") #land surface temp
        .filterBounds(point)
        .filterDate(*date_range)
        .select("LST_Day_1km") #get daytime land surface temp at 1km res.
        .mean()
    )

    combined = ndvi.addBands(smap).addBands(lst) #.addBands stacks all the three measurements above into one image object
    result = combined.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo() # reduceRegion() is a step that extracts the averages of all the pixel values of our point. getInfo triggers the computation of the imaging 

    return {
        "plot_id": plot_id,
        "lat": lat,
        "lon": lon,
        "ndvi": result.get("NDVI"),
        "soil_moisture": result.get("ssm"),
        "land_surface_temp_k": result.get("LST_Day_1km", 0)* 0.02 if result.get("LST_Day_1km") else None, # *0.02 to convert into Kelvin
    } # return all the main plot points into a dict

if __name__ == "__main__":
    plots = [
        {
            "plot_id": "plot_1", "lat": 37.4275, "lon": -122.1697 #REPLACE THESE VALUES WITH USER INPUT !!!!!!!!!
        },
    ]

    rows = [get_satellite_features(p["lat"], p["lon"], p["plot_id"]) for p in plots]
    pd.DataFrame(rows).to_csv("satellite_features.csv", index=False)
    print("wrote satellite_features.csv")
