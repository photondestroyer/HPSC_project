from osgeo import gdal, ogr, osr
import numpy as np
import os

def get_utm_epsg_from_latlon(lat, lon):
    zone = int((lon + 180) / 6) + 1
    # EPSG: 326xx for Northern Hemisphere, 327xx for Southern
    base = 32600 if lat >= 0 else 32700
    return base + zone

def create_dem_boundary_utm(input_tif, output_shp):
    ds = gdal.Open(input_tif)
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    
    # Get Source CRS
    src_proj = osr.SpatialReference()
    src_proj.ImportFromWkt(ds.GetProjection())
    
    # 1. Determine Target CRS (UTM)
    # If source is already projected (not LatLon), keep it. 
    # If LatLon, auto-calc UTM.
    if src_proj.IsProjected():
        target_proj = src_proj
        print(f"Source is already projected: {target_proj.GetName()}")
    else:
        # Get center point to determine UTM zone
        center_x = gt[0] + (width * gt[1]) / 2
        center_y = gt[3] + (height * gt[5]) / 2
        
        target_epsg = get_utm_epsg_from_latlon(center_y, center_x)
        target_proj = osr.SpatialReference()
        target_proj.ImportFromEPSG(target_epsg)
        print(f"Auto-detected Target UTM EPSG: {target_epsg}")

    # Transformation pipeline
    transform = osr.CoordinateTransformation(src_proj, target_proj)

    # 2. Calculate Corners (Pixel Logic) -> (Geo Logic)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # We use pixel edges, not centers, to ensure full coverage
    corners_pixel = [
        (0, 0),
        (width, 0),
        (width, height),
        (0, height)
    ]
    
    utm_coords = []
    
    print("\n--- ANUGA Domain Coordinates (Copy this list) ---")
    print("[")
    
    for x_pix, y_pix in corners_pixel:
        # Apply Affine Transform
        x_geo = gt[0] + x_pix * gt[1] + y_pix * gt[2]
        y_geo = gt[3] + x_pix * gt[4] + y_pix * gt[5]
        
        # Reproject to UTM
        point = transform.TransformPoint(x_geo, y_geo)
        # Store as (x, y) tuple ignoring z
        utm_coords.append((point[0], point[1]))
        print(f"  [{point[0]:.4f}, {point[1]:.4f}],")
        
    print("]")
    print("-------------------------------------------------")

    # 3. Create Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
        
    out_ds = driver.CreateDataSource(output_shp)
    layer = out_ds.CreateLayer("boundary", target_proj, geom_type=ogr.wkbPolygon)
    
    # Create Geometry
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in utm_coords:
        ring.AddPoint(x, y)
    # Close the ring
    ring.AddPoint(utm_coords[0][0], utm_coords[0][1])
    
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    feature_defn = layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    
    # Cleanup
    feature = None
    out_ds = None
    ds = None
    
    return utm_coords

# Example Usage
if __name__ == "__main__":
    # Replace with your DEM path
    dem_path = "ROI_2_DEM_UTM.tif" 
    shp_path = "domain_boundary.shp"
    
    # This will print the list needed for ANUGA's create_domain_from_region
    coords = create_dem_boundary_utm(dem_path, shp_path)