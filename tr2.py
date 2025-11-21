import pandas as pd
from shapely import wkt
import anuga
import os
from anuga import Domain, Geo_reference
from osgeo import gdal, osr
import numpy as np

# --- HELPER FUNCTION ---
def get_tiff_details(tiff_path):
    """Extract zone, proj4, EPSG, and hemisphere from a GeoTIFF."""
    if not os.path.exists(tiff_path): 
        raise FileNotFoundError(f"{tiff_path}")
    ds = gdal.Open(tiff_path)
    if ds is None: 
        raise ValueError("GDAL could not open the file.")
    
    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    
    if srs.IsProjected():
        zone = srs.GetUTMZone()
        # Fallback for zone parsing
        if zone == 0:
            name = srs.GetAttrValue("PROJCS")
            if name:
                import re
                match = re.search(r"zone\s+(\d+)", name, re.IGNORECASE)
                if match: zone = int(match.group(1))
        
        proj4_str = srs.ExportToProj4()
        epsg = srs.GetAuthorityCode(None)
        
        # Determine hemisphere from EPSG
        # ANUGA expects 'northern' or 'southern' (full words)
        hemisphere = 'northern'
        if epsg:
            if epsg.startswith('327'):
                hemisphere = 'southern'
            elif epsg.startswith('326'):
                hemisphere = 'northern'
        
        return zone, proj4_str, epsg, hemisphere
    else:
        raise ValueError("TIFF is not projected.")

def create_tiff_with_epsg(input_tiff, output_tiff, epsg_code):
    """Create a copy of the TIFF with explicit EPSG code."""
    print(f"Creating TIFF with EPSG:{epsg_code}: {output_tiff}")
    try:
        # Create SRS from EPSG
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg_code))
        
        # Open source and create destination
        src_ds = gdal.Open(input_tiff)
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.CreateCopy(output_tiff, src_ds, strict=0)
        
        # Set projection with explicit EPSG
        dst_ds.SetProjection(srs.ExportToWkt())
        
        # Close datasets
        dst_ds = None
        src_ds = None
        
        print(f"Created corrected TIFF: {output_tiff}")
        return True
    except Exception as e:
        print(f"Failed to create corrected TIFF: {e}")
        return False

class TopographyLoader:
    def __init__(self, domain):
        self.domain = domain

    def load_geotiff(self, tiff_path: str, quantity_name: str = 'elevation'):
        if not os.path.exists(tiff_path): 
            raise FileNotFoundError(f"{tiff_path}")
        print(f"Loading GeoTIFF: {tiff_path} into '{quantity_name}'...")
        try:
            self.domain.get_quantity(quantity_name).set_values_from_tif_file(tiff_path)
            print("Topography loaded successfully.")
        except Exception as e:
            print(f"Failed to load GeoTIFF. Reason: {e}")
            raise

class AnugaGeometryLoader:
    def parse_qgis_bounding_box(self, csv_path: str) -> list:
        df = pd.read_csv(csv_path)
        wkt_string = df['WKT'].iloc[0]
        polygon_geom = wkt.loads(wkt_string)
        
        # Buffer to avoid NoData edges
        polygon_geom = polygon_geom.buffer(-5000.0) 

        coords_tuples = list(polygon_geom.exterior.coords)
        anuga_polygon = [list(pt) for pt in coords_tuples]
        
        if anuga_polygon[0] == anuga_polygon[-1]:
            anuga_polygon.pop()
        return anuga_polygon

if __name__ == "__main__":
    csv_file = "bounder.csv"
    geotiff_file = "DEM_UTM.tif"

    # 1. GET TIFF CRS with hemisphere
    print("Reading GeoTIFF projection...")
    zone_num, tiff_proj4, epsg_code, hemisphere = get_tiff_details(geotiff_file)
    print(f"TIFF Zone: {zone_num}, EPSG: {epsg_code}, Hemisphere: {hemisphere}")
    
    # 2. For India zone 43, use EPSG 32643 (northern hemisphere)
    target_epsg = "32643"  # UTM Zone 43N
    
    # 3. Create corrected TIFF with explicit EPSG
    corrected_tiff = "DEM_UTM_EPSG32643.tif"
    if not os.path.exists(corrected_tiff):
        create_tiff_with_epsg(geotiff_file, corrected_tiff, target_epsg)
    else:
        print(f"Using existing corrected TIFF: {corrected_tiff}")
    
    # Verify the corrected TIFF
    _, _, corrected_epsg, corrected_hemisphere = get_tiff_details(corrected_tiff)
    print(f"Corrected TIFF - EPSG: {corrected_epsg}, Hemisphere: {corrected_hemisphere}")
    
    # 4. LOAD POLYGON
    geo_loader = AnugaGeometryLoader() 
    bounding_polygon = geo_loader.parse_qgis_bounding_box(csv_file)
    print(f"Parsed Polygon Vertices (First 3): {bounding_polygon[:3]}...")

    # 5. CREATE DOMAIN with larger triangles for faster computation
    num_segments = len(bounding_polygon)
    tags = {'exterior': list(range(num_segments))}

    print("Creating mesh (this may take a moment)...")
    domain = anuga.create_domain_from_regions(
        bounding_polygon, 
        boundary_tags=tags, 
        maximum_triangle_area=50000  # Larger triangles = faster mesh generation
    )
    
    print(f"Domain created with {domain.get_number_of_triangles()} triangles")
    print(f"Initial domain geo_reference: {domain.geo_reference}")
    
    # 6. SET ZONE AND HEMISPHERE DIRECTLY on the domain's geo_reference
    domain.geo_reference.set_zone(zone_num)
    domain.geo_reference.set_hemisphere(corrected_hemisphere)
    
    print(f"After setting zone and hemisphere: {domain.geo_reference}")

    # 7. LOAD TOPOGRAPHY from corrected TIFF
    topo_loader = TopographyLoader(domain)
    print('-'*45)
    try:
        topo_loader.load_geotiff(corrected_tiff)
        
        elev = domain.get_quantity('elevation')
        print(f"Elevation Range: {elev.get_minimum_value():.2f} to {elev.get_maximum_value():.2f}")
        print("✓ SUCCESS: Topography loaded without errors!")
        
    except Exception as e:
        print(f"Execution halted: {e}")
        raise

    # 8. SETUP INITIAL CONDITIONS
    domain.set_quantity('friction', 0.03)  # Manning's n roughness coefficient
    domain.set_quantity('stage', expression='elevation')  # Dry initial condition
    
    # 9. SETUP BOUNDARY CONDITIONS
    # Dirichlet boundary with negative stage = outflow condition
    # [stage, xmomentum, ymomentum]
    Bo = anuga.Dirichlet_boundary([-5.0, 0.0, 0.0])  # Outflow: stage below bed
    
    # Apply outflow boundary to all exterior boundaries
    print("\nBoundary tags:", domain.get_boundary_tags())
    domain.set_boundary({'exterior': Bo})
    
    print("Boundary conditions set: All exterior boundaries = Dirichlet outflow")
    
    # 10. CONFIGURE DOMAIN PARAMETERS
    domain.set_minimum_storable_height(0.00001)  # Only store depth > 10^-5 m
    domain.set_name('simulation_output')  # Output file name
    
    print("\n" + "="*50)
    print("DOMAIN SETUP COMPLETE")
    print("="*50)
    print(f"Number of triangles: {domain.get_number_of_triangles()}")
    print(f"Elevation range: {elev.get_minimum_value():.2f} to {elev.get_maximum_value():.2f} m")
    print(f"Boundary condition: Dirichlet outflow on all exterior boundaries")
    print(f"Initial condition: Dry bed (stage = elevation)")
    print(f"Friction coefficient: 0.03")
    print("="*50)


    # Visualize the triangulated mesh
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Get mesh coordinates and triangles
points = domain.get_vertex_coordinates()
triangles = domain.get_triangles()

# Reshape points to x, y coordinates
x = points[:, 0]
y = points[:, 1]

# Create triangulation for plotting
triang = tri.Triangulation(x, y, triangles)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Mesh triangulation
ax1.triplot(triang, 'b-', linewidth=0.3, alpha=0.5)
ax1.set_xlabel('Easting (m)', fontsize=12)
ax1.set_ylabel('Northing (m)', fontsize=12)
ax1.set_title(f'Triangulated Mesh\n({domain.get_number_of_triangles():,} triangles)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Elevation on mesh
elevation_values = domain.get_quantity('elevation').get_values(location='centroids')
tcf = ax2.tripcolor(triang, elevation_values, cmap='terrain', shading='flat')
ax2.set_xlabel('Easting (m)', fontsize=12)
ax2.set_ylabel('Northing (m)', fontsize=12)
ax2.set_title('Elevation on Mesh', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
cbar = plt.colorbar(tcf, ax=ax2)
cbar.set_label('Elevation (m)', fontsize=11)

plt.tight_layout()
plt.show()

print(f"\nMesh Statistics:")
print(f"  Total triangles: {domain.get_number_of_triangles():,}")
print(f"  Total vertices: {len(x):,}")
print(f"  Domain area: {domain.get_area():,.2f} m²")
print(f"  Domain area: {domain.get_area()/1e6:.2f} km²")