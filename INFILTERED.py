# Core libraries
import anuga
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
import os
import sys
import time
from pathlib import Path

# ============================================================================
# CRITICAL FIX: Apply SWW unlimited time dimension patch
# ============================================================================
# This fixes the "IndexError: index 0 is out of bounds" when using large yieldsteps
import sww_unlimited_time_fix
sww_unlimited_time_fix.apply_patch()
# ============================================================================

# Geospatial libraries
from osgeo import gdal, osr
from shapely import wkt
from pyproj import Transformer, CRS

# Check for GPU support
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] CuPy found - GPU acceleration ENABLED")
    print(f" GPU Device: {cp.cuda.Device()}")
    print(f" CuPy version: {cp.__version__}")
    
    # Get GPU info
    mempool = cp.get_default_memory_pool()
    mem_info = cp.cuda.Device().mem_info
    print(f" GPU Memory Available: {mem_info[0]/1e9:.2f} GB free / {mem_info[1]/1e9:.2f} GB total")
except ImportError:
    print("[X] CuPy not found - Running on CPU only")
    print(" To enable GPU: pip install cupy-cuda11x (or cupy-cuda12x)")

print(f"\nANUGA version: {anuga.__version__ if hasattr(anuga, '__version__') else 'Unknown'}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


# import os
print(os.getcwd())

os.chdir('/home/intern1/ishan/ANUGA-ing')
print(os.getcwd())

class RainfallSimulationConfig:
    """Configuration for rainfall-based overland flow simulation."""
    def __init__(self, start_date_str=None, end_date_str=None):
        # Input files
        self.dem_file = "filled_smoothed_topography_EPSG32643.tif" # DEM in UTM Zone 43N
        self.rainfall_csv = "IMERG_ROI1_2020.csv" # IMERG data
        self.bounding_box_csv = "bounder.csv" # Domain boundary
        
        # Coordinate system
        self.target_epsg = 32643 # UTM Zone 43N
        self.source_epsg = 4326 # WGS84 (lat/lon)
        
        # Domain parameters
        self.maximum_triangle_area = 25000 # m² (mesh resolution)
        self.buffer_distance = -5000.0 # meters (inward buffer)
        
        # Physical parameters
        self.friction_coefficient = 0.04 # Manning's n
        self.minimum_storable_height = 1e-8 # meters
        
        # Simulation time parameters
        # User-specified dates in DD-MM-YYYY format
        self.start_date_str = start_date_str # e.g., "01-01-2022"
        self.end_date_str = end_date_str # e.g., "10-01-2022"
        
        # These will be set from data or user input
        self.start_date = None # Will be set as datetime object
        self.end_date = None # Will be set as datetime object
        
        self.output_interval_hours = 24.0 # Save results every hour
        
        # GPU and performance
        self.use_gpu = GPU_AVAILABLE
        self.save_checkpoints = True
        self.checkpoint_interval_hours = 24.0 * 10 # Checkpoint every 5 days
        
        # Output
        self.output_dir = 'infiltration_runoff_June_ROI1_2020'
        self.simulation_name = 'rainfall_infiltration_simulation'

# USER INPUT: Specify your desired simulation period here (DD-MM-YYYY format)
# Example: config = RainfallSimulationConfig(start_date_str="01-01-2022", end_date_str="10-01-2022")
# Leave None to use first 10 days of available data
config = RainfallSimulationConfig(start_date_str="15-05-2020", end_date_str="30-06-2020")

print("[OK] Configuration loaded")
print(f" DEM: {config.dem_file}")
print(f" Rainfall data: {config.rainfall_csv}")
print(f" Target EPSG: {config.target_epsg}")
print(f" GPU enabled: {config.use_gpu}")
if config.start_date_str:
    print(f" User-specified start date: {config.start_date_str}")
if config.end_date_str:
    print(f" User-specified end date: {config.end_date_str}")
else:
    print(f" Date range: Will use available data (default: first 10 days)")

print("\n" + "="*70)
print("LOADING IMERG RAINFALL DATA")
print("="*70)

# Load rainfall CSV
print(f"\nReading {config.rainfall_csv}...")
rain_df = pd.read_csv(config.rainfall_csv)

print(f"[OK] Loaded {len(rain_df):,} records")
print(f"\nColumns: {list(rain_df.columns)}")
print(f"\nFirst few rows:")
print(rain_df.head())

# Parse time column (format: DD-MM-YYYY HH:MM)
print("\nParsing timestamps...")
rain_df['datetime'] = pd.to_datetime(rain_df['time'], format='%d-%m-%Y %H:%M')

# Get unique timestamps and spatial points
unique_times = sorted(rain_df['datetime'].unique())
unique_lats = sorted(rain_df['lat'].unique())
unique_lons = sorted(rain_df['long'].unique())

print(f"[OK] Parsed {len(unique_times)} unique timestamps")
print(f" Available time range: {unique_times[0]} to {unique_times[-1]}")
print(f" Total duration: {(unique_times[-1] - unique_times[0]).days} days, {(unique_times[-1] - unique_times[0]).seconds//3600} hours")

print(f"\n[OK] Spatial grid: {len(unique_lats)} latitudes × {len(unique_lons)} longitudes")
print(f" Latitude range: {min(unique_lats):.3f}° to {max(unique_lats):.3f}°")
print(f" Longitude range: {min(unique_lons):.3f}° to {max(unique_lons):.3f}°")

# Statistics
print(f"\n[OK] Precipitation statistics:")
print(f" Min: {rain_df['precipitation'].min():.6f} mm/hr")
print(f" Max: {rain_df['precipitation'].max():.6f} mm/hr")
print(f" Mean: {rain_df['precipitation'].mean():.6f} mm/hr")
print(f" Non-zero values: {(rain_df['precipitation'] > 0).sum():,} ({(rain_df['precipitation'] > 0).sum()/len(rain_df)*100:.1f}%)")

# Set simulation period based on user input or defaults
print(f"\n{'='*70}")
print("SETTING SIMULATION PERIOD")
print(f"{'='*70}")

if config.start_date_str:
    # Parse user-specified start date (DD-MM-YYYY format)
    config.start_date = pd.to_datetime(config.start_date_str, format='%d-%m-%Y')
    print(f"[OK] Using user-specified start date: {config.start_date}")
    
    # Validate start date is within available data
    if config.start_date < unique_times[0]:
        print(f" [WARNING] WARNING: Start date {config.start_date} is before available data starts {unique_times[0]}")
        print(f" Using earliest available date: {unique_times[0]}")
        config.start_date = unique_times[0]
    elif config.start_date > unique_times[-1]:
        raise ValueError(f"Start date {config.start_date} is after available data ends {unique_times[-1]}")
else:
    # Use first available date
    config.start_date = unique_times[0]
    print(f"[OK] Using first available date: {config.start_date}")

if config.end_date_str:
    # Parse user-specified end date (DD-MM-YYYY format)
    config.end_date = pd.to_datetime(config.end_date_str, format='%d-%m-%Y')
    # Set to end of day (23:59:59) to include all hours of the end date
    config.end_date = config.end_date.replace(hour=23, minute=59, second=59)
    print(f"[OK] Using user-specified end date: {config.end_date}")
    
    # Validate end date is within available data
    if config.end_date > unique_times[-1]:
        print(f" [WARNING] WARNING: End date {config.end_date} is after available data ends {unique_times[-1]}")
        print(f" Using latest available date: {unique_times[-1]}")
        config.end_date = unique_times[-1]
    elif config.end_date < config.start_date:
        raise ValueError(f"End date {config.end_date} is before start date {config.start_date}")
else:
    # Default: 10 days from start date (or end of available data, whichever is earlier)
    default_end = config.start_date + timedelta(days=10)
    config.end_date = min(default_end, unique_times[-1])
    print(f"[OK] Using default period (10 days): {config.end_date}")

print(f"\n{'='*70}")
print(f"FINAL SIMULATION PERIOD: {config.start_date} to {config.end_date}")
print(f"Duration: {(config.end_date - config.start_date).days} days, {(config.end_date - config.start_date).seconds//3600} hours")
print(f"{'='*70}")


print("\n" + "="*70)
print("COORDINATE TRANSFORMATION: WGS84 → UTM Zone 43N")
print("="*70)

# Create transformer from WGS84 to UTM Zone 43N
print(f"\nCreating coordinate transformer...")
print(f" Source: EPSG:{config.source_epsg} (WGS84 lat/lon)")
print(f" Target: EPSG:{config.target_epsg} (UTM Zone 43N)")

transformer = Transformer.from_crs(
CRS.from_epsg(config.source_epsg),
CRS.from_epsg(config.target_epsg),
always_xy=True
)

# Transform coordinates
print("\nTransforming rainfall grid coordinates...")
rain_df['easting'], rain_df['northing'] = transformer.transform(
rain_df['long'].values,
rain_df['lat'].values
)

print(f"[OK] Coordinates transformed")
print(f"\nUTM coordinates:")
print(f" Easting range: {rain_df['easting'].min():.2f} to {rain_df['easting'].max():.2f} m")
print(f" Northing range: {rain_df['northing'].min():.2f} to {rain_df['northing'].max():.2f} m")

# Show example transformation
print(f"\nExample transformation:")
sample = rain_df.iloc[0]
print(f" Lat/Lon: ({sample['lat']:.6f}°, {sample['long']:.6f}°)")
print(f" UTM: ({sample['easting']:.2f} m E, {sample['northing']:.2f} m N)")

def parse_qgis_bounding_box(csv_path, buffer_distance=-5000.0):
    """Parse QGIS bounding box from CSV with optional buffering."""
    df = pd.read_csv(csv_path)
    wkt_string = df['WKT'].iloc[0]
    polygon_geom = wkt.loads(wkt_string)

    if buffer_distance != 0:
        polygon_geom = polygon_geom.buffer(buffer_distance)

    coords_tuples = list(polygon_geom.exterior.coords)
    anuga_polygon = [list(pt) for pt in coords_tuples]

    if anuga_polygon[0] == anuga_polygon[-1]:
        anuga_polygon.pop()

    return anuga_polygon

print("\n" + "="*70)
print("CREATING ANUGA DOMAIN")
print("="*70)

# Load boundary polygon
print(f"\nLoading boundary from {config.bounding_box_csv}...")
bounding_polygon = parse_qgis_bounding_box(config.bounding_box_csv, config.buffer_distance)
print(f"[OK] Parsed {len(bounding_polygon)} vertices")

# Create mesh
print(f"\nCreating computational mesh (max triangle area: {config.maximum_triangle_area} m²)...")
num_segments = len(bounding_polygon)
tags = {'exterior': list(range(num_segments))}

start_time = time.time()
domain = anuga.create_domain_from_regions(
bounding_polygon,
boundary_tags=tags,
maximum_triangle_area=config.maximum_triangle_area
)
mesh_time = time.time() - start_time

print(f"[OK] Domain created in {mesh_time:.2f}s")
print(f" Triangles: {domain.get_number_of_triangles():,}")
print(f" Vertices: {domain.get_number_of_nodes():,}")
print(f" Area: {domain.get_area()/1e6:.2f} km²")

# Set coordinate reference
domain.geo_reference.set_zone(43) # UTM Zone 43N
domain.geo_reference.set_hemisphere('northern')
print(f"[OK] Set UTM Zone 43N (Northern Hemisphere)")

# Configure flow algorithm for GPU
domain.set_flow_algorithm('DE0')
domain.set_low_froude(0)
print(f"[OK] Flow algorithm: DE0 (GPU compatible)")


print("\n" + "="*70)
print("LOADING TOPOGRAPHY")
print("="*70)

print(f"\nLoading DEM from {config.dem_file}...")
domain.get_quantity('elevation').set_values_from_tif_file(config.dem_file)
elev = domain.get_quantity('elevation')

print(f"[OK] Elevation loaded")
print(f" Min elevation: {elev.get_minimum_value():.2f} m")
print(f" Max elevation: {elev.get_maximum_value():.2f} m")
print(f" Mean elevation: {np.mean(elev.centroid_values):.2f} m")

# Set initial conditions
print(f"\nSetting initial conditions...")
domain.set_quantity('friction', config.friction_coefficient)
domain.set_quantity('stage', expression='elevation') # Dry bed
print(f"[OK] Friction coefficient: {config.friction_coefficient}")
print(f"[OK] Initial condition: Dry bed (stage = elevation)")

# Set boundary conditions
print(f"\nSetting boundary conditions...")
Bo = anuga.Dirichlet_boundary([-10.0, 0.0, 0.0]) # Outflow
domain.set_boundary({'exterior': Bo})
print(f"[OK] Boundary: Dirichlet outflow on all exterior boundaries")

# Configure domain
domain.set_minimum_storable_height(config.minimum_storable_height)
domain.set_name(config.simulation_name)
domain.set_datadir(config.output_dir)
os.makedirs(config.output_dir, exist_ok=True)

# Enable .sww file storage with the ANUGA core fix applied
# FIX APPLIED: Modified anuga_core-main/anuga/file/sww.py to use unlimited time dimension
# This resolves the "IndexError: index 0 is out of bounds" issue with large yieldsteps
domain.set_store(True)  # ✅ NOW ENABLED - unlimited dimension fix applied
domain.set_store_vertices_uniquely(False)
print(f"✅ SWW file storage ENABLED (using unlimited time dimension fix)")
print(f"   Fix location: anuga_core-main/anuga/file/sww.py line 664-668")



import gc # Explicit garbage collection

print("\n" + "="*70)
print("PREPARING RAINFALL OPERATOR (DISK-BACKED MEMORY-MAPPED)")
print("="*70)

# Get domain centroid coordinates
centroids = domain.get_centroid_coordinates()
domain_x = centroids[:, 0]
domain_y = centroids[:, 1]

print(f"\nDomain mesh:")
print(f" Triangles: {len(domain_x):,}")
print(f" X range: {domain_x.min():.2f} to {domain_x.max():.2f} m")
print(f" Y range: {domain_y.min():.2f} to {domain_y.max():.2f} m")

# Filter rainfall data for simulation period
print(f"\nFiltering rainfall data for simulation period...")
rain_sim = rain_df[
    (rain_df['datetime'] >= config.start_date) & 
    (rain_df['datetime'] <= config.end_date)
].copy()

print(f"[OK] Filtered to {len(rain_sim):,} records")

# Get unique times in simulation period (as NumPy array)
sim_times = np.array(sorted(rain_sim['datetime'].unique()))
print(f"[OK] {len(sim_times)} time steps")

# Convert to seconds since start (vectorized)
rain_sim['seconds'] = (rain_sim['datetime'] - config.start_date).dt.total_seconds()
time_seconds = (pd.Series(sim_times) - config.start_date).dt.total_seconds().values

print(f"\nTime steps: 0 to {time_seconds[-1]:.0f} seconds ({time_seconds[-1]/3600:.1f} hours)")

# Get unique lat/lon grid spacing (convert to NumPy arrays immediately)
unique_lats = np.array(sorted(rain_sim['lat'].unique()))
unique_lons = np.array(sorted(rain_sim['long'].unique()))
lat_spacing = np.diff(unique_lats).mean() if len(unique_lats) > 1 else 0.1
lon_spacing = np.diff(unique_lons).mean() if len(unique_lons) > 1 else 0.1

print(f"\n[OK] Rainfall grid spacing:")
print(f" Latitude spacing: {lat_spacing:.4f}° (~{lat_spacing * 111000:.0f} m)")
print(f" Longitude spacing: {lon_spacing:.4f}° (~{lon_spacing * 111000 * np.cos(np.radians(unique_lats[0])):.0f} m)")

# DISK-BACKED OPTIMIZATION: Write rainfall to memory-mapped file
print(f"\n Creating disk-backed rainfall storage (MEMORY-MAPPED)...")
print(f" Method: Each cell uses average of 4 corner grid points")
print(f" Storage: Memory-mapped binary file (zero RAM for data)")

# Create output directory for rainfall cache
rainfall_cache_dir = os.path.join(config.output_dir, 'rainfall_infiltration_cache')
os.makedirs(rainfall_cache_dir, exist_ok=True)

# Memory-mapped file path
rainfall_mmap_file = os.path.join(rainfall_cache_dir, 'rainfall_data.dat')
time_index_file = os.path.join(rainfall_cache_dir, 'time_index.npy')

# OPTIMIZATION 1: Transform domain centroids ONCE (not in loop)
from pyproj import Transformer
transformer_inv = Transformer.from_crs(
    CRS.from_epsg(config.target_epsg),
    CRS.from_epsg(config.source_epsg),
    always_xy=True
)
domain_lon, domain_lat = transformer_inv.transform(domain_x, domain_y)
print(f" [OK] Transformed {len(domain_x):,} domain points to lat/lon")

# OPTIMIZATION 2: Pre-compute cell indices for all domain points (vectorized)
lat_indices = np.searchsorted(unique_lats, domain_lat)
lon_indices = np.searchsorted(unique_lons, domain_lon)

# Clip indices to valid ranges
lat_indices = np.clip(lat_indices, 1, len(unique_lats) - 1)
lon_indices = np.clip(lon_indices, 1, len(unique_lons) - 1)

# Get the 4 corner indices for each domain point (vectorized)
lat_ll_idx = lat_indices - 1 # lower-left
lat_ur_idx = lat_indices # upper-right
lon_ll_idx = lon_indices - 1 # lower-left
lon_ur_idx = lon_indices # upper-right

print(f" [OK] Pre-computed cell indices for all domain points")

# OPTIMIZATION 3: Create efficient lookup structure for rainfall data
print(f" [OK] Creating efficient rainfall grid lookup...")

# Create mapping from actual lat/lon values to grid indices
lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}

# Pre-compute indices for ALL rainfall data at once (vectorized)
lat_indices_all = rain_sim['lat'].map(lat_to_idx).values
lon_indices_all = rain_sim['long'].map(lon_to_idx).values

# Get datetime and precip as arrays for fast comparison
datetime_array = rain_sim['datetime'].values
precip_array = rain_sim['precipitation'].values.astype(np.float32)

# MEMORY-CRITICAL: Pre-allocate rain_grid ONCE and reuse (float32)
rain_grid = np.zeros((len(unique_lats), len(unique_lons)), dtype=np.float32)

# Pre-allocate corner value arrays ONCE (float32)
v_ll = np.empty(len(domain_x), dtype=np.float32)
v_lr = np.empty(len(domain_x), dtype=np.float32)
v_ul = np.empty(len(domain_x), dtype=np.float32)
v_ur = np.empty(len(domain_x), dtype=np.float32)

# Create memory-mapped array: shape = (n_timesteps, n_triangles), dtype=float32
n_timesteps = len(sim_times)
n_triangles = len(domain_x)

# CHECK IF FILES ALREADY EXIST (SKIP REGENERATION AFTER KERNEL CRASH)
if os.path.exists(rainfall_mmap_file) and os.path.exists(time_index_file):
    print(f"\n[OK] Found existing rainfall cache files!")
    print(f" Rainfall data: {rainfall_mmap_file} ({os.path.getsize(rainfall_mmap_file)/1e9:.2f} GB)")
    print(f" Time index: {time_index_file}")
    print(f"\n Loading from cache (skipping regeneration)...")

    # Open existing memory-mapped file in read-only mode
    rainfall_mmap_readonly = np.memmap(rainfall_mmap_file, dtype=np.float32, mode='r',
                                       shape=(n_timesteps, n_triangles))

    print(f" [OK] Loaded memory-mapped file:")
    print(f"      Shape: ({n_timesteps:,} timesteps, {n_triangles:,} triangles)")
    print(f"      RAM overhead: ~0 MB (OS page cache only)")

    # Verify data integrity
    test_rainfall = rainfall_mmap_readonly[0, :]
    print(f"\n [OK] Data verification (first timestep):")
    print(f"      Min: {test_rainfall.min()*3600*1000:.6f} mm/hr")
    print(f"      Max: {test_rainfall.max()*3600*1000:.6f} mm/hr")
    print(f"      Mean: {test_rainfall.mean()*3600*1000:.6f} mm/hr")

    print(f"\n      Using cached rainfall data (kernel crash recovery)")
    print(f"      To regenerate: delete '{rainfall_cache_dir}/' directory\n")

    # Set flag to skip generation
    SKIP_GENERATION = True
else:
    print(f"\n Cache files not found - will generate new data")
    print(f"\n Creating memory-mapped file: {rainfall_mmap_file}")
    print(f" Shape: ({n_timesteps:,} timesteps, {n_triangles:,} triangles)")
    print(f" Size: {n_timesteps * n_triangles * 4 / 1e9:.2f} GB (on disk, NOT in RAM)")
    SKIP_GENERATION = False

if not SKIP_GENERATION:
    # Create memory-mapped array (write mode)
    rainfall_mmap = np.memmap(rainfall_mmap_file, dtype=np.float32, mode='w+', 
                              shape=(n_timesteps, n_triangles))

    # Process each timestep and write directly to disk
    gc_interval = 500
    for i, t in enumerate(sim_times):
        # VECTORIZED: Find all records for this timestep
        mask = (datetime_array == t)
        lat_idx_t = lat_indices_all[mask]
        lon_idx_t = lon_indices_all[mask]
        precip_t = precip_array[mask]

        # Zero out the grid IN-PLACE
        rain_grid.fill(0.0)

        # Vectorized assignment to grid
        rain_grid[lat_idx_t, lon_idx_t] = precip_t

        # Use pre-allocated arrays (in-place copy)
        np.copyto(v_ll, rain_grid[lat_ll_idx, lon_ll_idx])
        np.copyto(v_lr, rain_grid[lat_ll_idx, lon_ur_idx])
        np.copyto(v_ul, rain_grid[lat_ur_idx, lon_ll_idx])
        np.copyto(v_ur, rain_grid[lat_ur_idx, lon_ur_idx])

        # Average of 4 corners and convert mm/hr to m/s
        rainfall_array = ((v_ll + v_lr + v_ul + v_ur) * (0.25 / 3600000.0)).astype(np.float32)

        # Write directly to memory-mapped file (row i)
        rainfall_mmap[i, :] = rainfall_array

        # Explicit garbage collection at intervals
        if (i + 1) % gc_interval == 0:
            rainfall_mmap.flush() # Write to disk
            gc.collect()
            print(f" Progress: {i+1}/{n_timesteps} timesteps | Written to disk, GC performed")
        elif (i + 1) % (gc_interval * 2) == 0:
            print(f" Progress: {i+1}/{n_timesteps} timesteps")

    # Final flush to ensure all data is written
    rainfall_mmap.flush()
    del rainfall_mmap # Close the file
    gc.collect()

    print(f"\n[OK] Created memory-mapped rainfall file")
    print(f" File: {rainfall_mmap_file}")
    print(f" Size on disk: {os.path.getsize(rainfall_mmap_file) / 1e9:.2f} GB")
    print(f" RAM usage: ~0 MB (memory-mapped, loaded on-demand)")

    # Save time index for fast lookup
    np.save(time_index_file, time_seconds)
    print(f" Time index saved: {time_index_file}")

    # Clean up temporary arrays
    del v_ll, v_lr, v_ul, v_ur, rain_grid, lat_indices_all, lon_indices_all
    del datetime_array, precip_array
    gc.collect()

    print(f"\n[OK] Cleaned up temporary arrays")

    # Reopen in read-only mode for verification
    rainfall_mmap_readonly = np.memmap(rainfall_mmap_file, dtype=np.float32, mode='r', 
                                       shape=(n_timesteps, n_triangles))

# Test reading from memory-mapped file (works for both cached and newly generated)
print(f"\n[OK] Testing memory-mapped file access...")
test_rainfall = rainfall_mmap_readonly[0, :] # Read first timestep
print(f" Min rainfall: {test_rainfall.min()*3600*1000:.6f} mm/hr")
print(f" Max rainfall: {test_rainfall.max()*3600*1000:.6f} mm/hr")
print(f" Mean rainfall: {test_rainfall.mean()*3600*1000:.6f} mm/hr")
print(f" Non-zero points: {np.sum(test_rainfall > 0):,} ({np.sum(test_rainfall > 0)/len(test_rainfall)*100:.1f}%)")

# Verify against original data
first_time_data = rain_sim[rain_sim['datetime'] == sim_times[0]]
print(f"\n[OK] Verifying against original IMERG data at first timestep...")
print(f" Original IMERG precipitation at t=0:")
print(f"  Min: {first_time_data['precipitation'].min():.6f} mm/hr")
print(f"  Max: {first_time_data['precipitation'].max():.6f} mm/hr")
print(f"  Mean: {first_time_data['precipitation'].mean():.6f} mm/hr")
print(f" After cell-based averaging (4-corner average):")
print(f"  Actual range: {test_rainfall.min()*3600*1000:.6f} to {test_rainfall.max()*3600*1000:.6f} mm/hr")

# Clean up test data
del test_rainfall
gc.collect()

print(f"\n[OK] Rainfall data ready")
print(f" [TIP] Data stored on disk, not in RAM!")
print(f" [TIP] Operator will read data on-demand with zero memory overhead")
print(f" Status: {'Reused from cache' if SKIP_GENERATION else 'Newly generated'}")



class Spatial_Temporal_Rainfall_Operator(anuga.Rate_operator):
    """
    Custom rainfall operator that applies cell-based constant rainfall values.
    
    Uses ANUGA's Rate_operator which properly handles source terms.
    Reads rainfall data from memory-mapped file for zero RAM overhead.
    
    **CRITICAL FIX**: Implements __call__() to ensure update_rate() is called during simulation
    """
    
    def __init__(self, domain, rainfall_mmap_file, time_seconds, 
                 description=None, label=None, logging=False, verbose=False):
        """
        Initialize the spatial-temporal rainfall operator.
        
        Parameters:
        -----------
        domain : ANUGA Domain
            The computational domain
        rainfall_mmap_file : str
            Path to memory-mapped rainfall data file
        time_seconds : array
            Array of time values (seconds) for which rainfall data exists
        """
        self.time_seconds = time_seconds
        self.n_triangles = domain.get_number_of_triangles()
        self.n_timesteps = len(time_seconds)
        self.current_time_idx = -1
        self.call_counter = 0  # For debugging
        
        # Open memory-mapped file in read-only mode
        self.rainfall_mmap = np.memmap(rainfall_mmap_file, dtype=np.float32, mode='r',
                                       shape=(self.n_timesteps, self.n_triangles))
        
        print(f"[OK] Memory-mapped rainfall file opened (read-only)")
        print(f" File: {rainfall_mmap_file}")
        print(f" Shape: ({self.n_timesteps:,} timesteps, {self.n_triangles:,} triangles)")
        print(f" RAM overhead: ~0 MB (OS page cache only)")
        
        # Initialize with first rainfall rate (read from mmap)
        initial_rate = self.rainfall_mmap[0, :].copy()  # Copy to avoid mmap reference
        
        # Initialize Rate_operator with initial rate
        anuga.Rate_operator.__init__(self, domain, rate=initial_rate,
                                     description=description, label=label,
                                     logging=logging, verbose=verbose)
        
        print(f"[OK] Cell-Based Rainfall Operator initialized (DISK-BACKED)")
        print(f" Domain triangles: {self.n_triangles:,}")
        print(f" Time steps: {len(self.time_seconds)}")
        print(f" Method: Memory-mapped file access (zero RAM overhead)")
        print(f" DEBUG: First 5 __call__() invocations will print debug info")
    
    def __call__(self):
        """
        Apply rainfall - this is called by ANUGA during evolution.
        
        This overrides Rate_operator.__call__() to update rainfall rate
        from memory-mapped file and then apply it.
        
        **CRITICAL**: Without this override, update_rate() would never be called!
        """
        self.call_counter += 1
        
        # Update the rainfall rate based on current simulation time
        self.update_rate()
        
        # Call parent's __call__() to actually apply the rate
        anuga.Rate_operator.__call__(self)
        
        # DEBUG: Print first few calls
        if self.call_counter <= 5:
            t = self.domain.get_time()
            print(f"    DEBUG Rainfall __call__() #{self.call_counter}: t={t:.1f}s")
            print(f"          Current rainfall rate: min={self.rate.min()*3600*1000:.6f} mm/hr, "
                  f"max={self.rate.max()*3600*1000:.6f} mm/hr")
    
    def update_rate(self):
        """
        Update the rainfall rate based on current simulation time.
        Reads data from memory-mapped file on-demand.
        """
        t = self.domain.get_time()
        
        # Find the appropriate rainfall array index for current time
        idx = np.searchsorted(self.time_seconds, t)
        
        if idx >= len(self.time_seconds):
            idx = len(self.time_seconds) - 1
        elif idx > 0 and abs(t - self.time_seconds[idx-1]) < abs(t - self.time_seconds[idx]):
            idx = idx - 1
        
        # Read rainfall data from memory-mapped file (fast, OS-cached)
        rainfall_rate = self.rainfall_mmap[idx, :]  # Read row idx
        
        # Ensure non-negative
        rainfall_rate = np.maximum(rainfall_rate, 0.0)
        
        # Update the rate (Rate_operator will apply this)
        self.set_rate(rainfall_rate)
        
        # Update monitoring
        if idx != self.current_time_idx:
            self.current_time_idx = idx
            if self.verbose and idx % 24 == 0:  # Print every 24 hours
                print(f" Rainfall updated at t={t/3600:.1f}h: "
                      f"min={rainfall_rate.min()*3600*1000:.3f} mm/hr, "
                      f"max={rainfall_rate.max()*3600*1000:.3f} mm/hr "
                      f"(read from disk)")
    
    def parallel_safe(self):
        """Required for ANUGA operator framework."""
        return True
    
    def statistics(self):
        """Return statistics about current rainfall."""
        if hasattr(self, 'rate') and self.rate is not None:
            if isinstance(self.rate, np.ndarray):
                rate_array = self.rate
            else:
                rate_array = np.full(self.n_triangles, self.rate)
            
            return {
                'min_rate_mm_hr': float(rate_array.min() * 3600 * 1000),
                'max_rate_mm_hr': float(rate_array.max() * 3600 * 1000),
                'mean_rate_mm_hr': float(rate_array.mean() * 3600 * 1000)
            }
        return {}
    
    def __del__(self):
        """Clean up memory-mapped file on deletion."""
        if hasattr(self, 'rainfall_mmap'):
            del self.rainfall_mmap

print("[OK] Custom Cell-Based Rainfall Operator class defined (DISK-BACKED)")


print("\n" + "="*70)
print("PRECIPITATION ANALYSIS OVER SIMULATION PERIOD")
print("="*70)

# OPTIMIZED: Read from memory-mapped file for zero RAM overhead
# Pre-allocate arrays for statistics
timestep_mins = np.zeros(n_timesteps, dtype=np.float32)
timestep_maxs = np.zeros(n_timesteps, dtype=np.float32)
timestep_means = np.zeros(n_timesteps, dtype=np.float32)
timestep_nonzero_pct = np.zeros(n_timesteps, dtype=np.float32)
timestep_hours = time_seconds / 3600.0

print(f"Analyzing {n_timesteps:,} timesteps from memory-mapped file...")
print(f" (Reading on-demand, ~0 MB RAM overhead)")

# Vectorized computation reading from mmap (OS caches frequently accessed data)
gc_interval = 500 # Run GC every N timesteps during analysis
for i in range(n_timesteps):
    # Read rainfall data from mmap (fast, OS-cached)
    rates_m_s = rainfall_mmap_readonly[i, :]

    # Convert to mm/hr for analysis
    rates_mm_hr = rates_m_s * 3600 * 1000

    timestep_mins[i] = rates_mm_hr.min()
    timestep_maxs[i] = rates_mm_hr.max()
    timestep_means[i] = rates_mm_hr.mean()
    timestep_nonzero_pct[i] = (rates_m_s > 0).sum() / len(rates_m_s) * 100

    # MEMORY-CRITICAL: GC at intervals to prevent accumulation
    if (i + 1) % gc_interval == 0:
        gc.collect()

# Final GC after analysis
gc.collect()
print("[OK] Analysis complete")

# ========== OVERALL STATISTICS ==========
print(f"\n{'='*70}")
print("OVERALL PRECIPITATION STATISTICS")
print(f"{'='*70}")
print(f"Total timesteps analyzed: {n_timesteps:,}")
print(f"Duration: {timestep_hours[-1]:.1f} hours ({timestep_hours[-1]/24:.1f} days)")

print(f"\n Across All Timesteps:")
print(f" Min rainfall (any cell, any time): {timestep_mins.min():.6f} mm/hr")
print(f" Max rainfall (any cell, any time): {timestep_maxs.max():.6f} mm/hr")
print(f" Mean of timestep means: {timestep_means.mean():.6f} mm/hr")
print(f" Mean coverage (cells with rain): {timestep_nonzero_pct.mean():.2f}%")

# ========== PEAK RAINFALL EVENTS ==========
max_mean_idx = timestep_means.argmax()
max_max_idx = timestep_maxs.argmax()

print(f"\n Peak Rainfall Events:")
print(f" Highest mean rainfall: {timestep_means[max_mean_idx]:.6f} mm/hr at t={timestep_hours[max_mean_idx]:.1f} hrs")
print(f" Highest max rainfall: {timestep_maxs[max_max_idx]:.6f} mm/hr at t={timestep_hours[max_max_idx]:.1f} hrs")

# ========== TIME SERIES SUMMARY ==========
print(f"\n Temporal Patterns:")
# Identify dry periods (mean rainfall < 0.01 mm/hr)
dry_mask = timestep_means < 0.01
wet_mask = ~dry_mask
n_dry = dry_mask.sum()
n_wet = wet_mask.sum()

print(f" Dry timesteps (< 0.01 mm/hr mean): {n_dry:,} ({n_dry/n_timesteps*100:.1f}%)")
print(f" Wet timesteps (≥ 0.01 mm/hr mean): {n_wet:,} ({n_wet/n_timesteps*100:.1f}%)")

if n_wet > 0:
    print(f"\n Statistics for Wet Timesteps Only:")
    print(f" Mean rainfall: {timestep_means[wet_mask].mean():.6f} mm/hr")
    print(f" Max rainfall: {timestep_means[wet_mask].max():.6f} mm/hr")
    print(f" Mean coverage: {timestep_nonzero_pct[wet_mask].mean():.2f}%")

    # ========== SAMPLE TIMESTEPS ==========
    # Show detailed info for first, middle, and last timesteps
    sample_indices = [0, n_timesteps // 2, -1]
    print(f"\n Sample Timestep Details:")
    for idx in sample_indices:
        if idx == -1:
            idx = n_timesteps - 1
        t_hr = timestep_hours[idx]
        print(f" t={t_hr:7.1f}h: mean={timestep_means[idx]:8.6f} mm/hr, "
              f"max={timestep_maxs[idx]:8.6f} mm/hr, "
              f"coverage={timestep_nonzero_pct[idx]:5.1f}%")

print(f"\n{'='*70}\n")

# Clean up analysis arrays
del timestep_mins, timestep_maxs, timestep_means, timestep_nonzero_pct
del timestep_hours
gc.collect()
print("[OK] Analysis arrays cleaned up")


print("\n" + "="*70)
print("ADDING RAINFALL OPERATOR TO DOMAIN")
print("="*70)

# Create and add the cell-based rainfall operator (DISK-BACKED)
rainfall_op = Spatial_Temporal_Rainfall_Operator(
    domain,
    rainfall_mmap_file=rainfall_mmap_file,
    time_seconds=time_seconds,
    description="IMERG hourly cell-based rainfall (memory-mapped)",
    label="IMERG_rainfall",
    verbose=True
)

print("\n[OK] Cell-based rainfall operator added to domain (DISK-BACKED)")
print("     Each domain triangle receives constant rainfall based on")
print("     4-corner average from IMERG grid cells")
print("     Rainfall data read on-demand from disk (zero RAM overhead)")


print("\n" + "="*70)
print("GPU ACCELERATION SETUP")
print("="*70)

if not GPU_AVAILABLE:
    print("\n❌ ERROR: GPU not available!")
    print("   CuPy not found or GPU not accessible")
    sys.exit(1)

# Force GPU mode (user requirement: GPU only)
try:
    # Test GPU
    import cupy as cp
    print("\n🔍 Testing GPU accessibility...")
    test_array = cp.random.random((1000, 1000))
    test_result = cp.sum(test_array)
    del test_array  # Free memory
    print(f"✅ GPU computation test passed (result: {float(test_result):.2f})")
    
    # Get GPU info
    gpu_device = cp.cuda.Device()
    print(f"\n📊 GPU Device Information:")
    
    # Get device name safely
    try:
        device_name = cp.cuda.runtime.getDeviceProperties(gpu_device.id)['name'].decode('utf-8')
        print(f"   Name: {device_name}")
    except Exception:
        print(f"   Device ID: {gpu_device.id}")
    
    print(f"   Compute Capability: {gpu_device.compute_capability}")
    
    mem_info = gpu_device.mem_info
    print(f"   Total Memory: {mem_info[1]/1e9:.2f} GB")
    print(f"   Available Memory: {mem_info[0]/1e9:.2f} GB")
    
    # Enable GPU mode BEFORE any simulation operations
    print(f"\n🚀 Enabling GPU mode...")
    domain.set_multiprocessor_mode(2)  # 2 = GPU mode
    
    # Verify GPU interface was created
    print(f"   Multiprocessor mode: {domain.get_multiprocessor_mode()}")
    
    if domain.gpu_interface is None:
        print(f"\n❌ CRITICAL ERROR: GPU interface is None after set_multiprocessor_mode(2)")
        print(f"   This usually means:")
        print(f"   1. ANUGA was not compiled with CUDA support")
        print(f"   2. GPU kernels failed to load")
        print(f"   3. CuPy version incompatibility")
        print(f"\n🔧 Attempting manual GPU interface initialization...")
        
        # Try to manually initialize GPU interface
        try:
            from anuga.shallow_water.shallow_water_domain import GPU_interface
            domain.gpu_interface = GPU_interface(domain)
            print(f"   ✅ Manual GPU interface created!")
        except Exception as manual_err:
            print(f"   ❌ Manual initialization failed: {manual_err}")
            print(f"\n💡 Your ANUGA installation may not support GPU acceleration.")
            print(f"   To check: python -c \"import anuga; print(hasattr(anuga.Domain, 'set_multiprocessor_mode'))\"")
            sys.exit(1)
    
    if domain.gpu_interface is not None:
        print(f"✅ GPU ACCELERATION ENABLED!")
        print(f"   GPU interface type: {type(domain.gpu_interface).__name__}")
        print(f"   GPU interface module: {domain.gpu_interface.__class__.__module__}")
        
        # Check for required GPU kernels
        required_kernels = [
            'protect_against_infinitesimal_and_negative_heights_kernel',
            'compute_fluxes_ext_short_kernel',
            'saxpy_centroid_values_kernel'
        ]
        
        print(f"\n🔍 Checking GPU kernels:")
        for kernel_name in required_kernels:
            has_kernel = hasattr(domain.gpu_interface, kernel_name)
            status = "✅" if has_kernel else "❌"
            print(f"   {status} {kernel_name}: {has_kernel}")
        
        # Show GPU memory usage
        mempool = cp.get_default_memory_pool()
        print(f"\n💾 GPU Memory Pool:")
        print(f"   Used: {mempool.used_bytes()/1e9:.3f} GB")
        print(f"   Total allocated: {mempool.total_bytes()/1e9:.3f} GB")
        
        print(f"\n💡 Monitor GPU with: watch -n 1 nvidia-smi")
    else:
        print(f"\n❌ FATAL ERROR: GPU interface still None after all attempts")
        print(f"   Your ANUGA installation does not support GPU acceleration")
        print(f"   Exiting (user requirement: GPU only)")
        sys.exit(1)
    
except ImportError as e:
    print(f"\n❌ FATAL ERROR: Cannot import required GPU modules: {e}")
    print(f"   Exiting (user requirement: GPU only)")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ FATAL ERROR during GPU setup: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n   Exiting (user requirement: GPU only)")
    sys.exit(1)

print("\n" + "="*70)
print("DOMAIN READY FOR SIMULATION")
print("="*70)
print(f"Triangles: {domain.get_number_of_triangles():,}")
print(f"Area: {domain.get_area()/1e6:.2f} km²")
print(f"Mode: {' GPU (CUDA/CuPy)' if config.use_gpu else ' CPU (OpenMP)'}")
print("="*70)

class GreenAmptInfiltrationOperator(anuga.Rate_operator):
    """
    Green-Ampt infiltration operator for ANUGA.
    
    Implements soil infiltration following the Green-Ampt method. This operator 
    removes water from the surface and tracks cumulative infiltration depth.
    
    The Green-Ampt equation:
        f(t) = K_s * (1 + (ψ*Δθ) / L_f)
        
    Where:
        K_s = hydraulic conductivity (m/s)
        ψ = wetting front capillary pressure head (m)
        Δθ = moisture deficit (porosity - initial moisture) (-)
        L_f = wetting front depth (m)
    
    Actual infiltration = min(potential_infiltration, available_water)
    
    **IMPORTANT**: This uses __call__() override (not update_rate) to match ANUGA's design
    **FIX**: Only infiltrates where water_depth > 1 mm to avoid numerical issues
    **FIX 2**: Uses proper minimum L_f to prevent infinite infiltration rates at start
    **FIX 3**: Uses hard-coded constant max infiltration rate (25.9 mm/hr)
    """
    
    # Default soil properties (from Rawls et al., 1992)
    SOIL_PROPS = {
        'sand': {'lambda': 0.694, 'h_b': 0.0726, 'K_s': 5.83e-5},
        'loamy sand': {'lambda': 0.553, 'h_b': 0.0869, 'K_s': 1.70e-5},
        'sandy loam': {'lambda': 0.378, 'h_b': 0.1466, 'K_s': 7.19e-6},
        'loam': {'lambda': 0.252, 'h_b': 0.1115, 'K_s': 3.67e-6},
        'silt loam': {'lambda': 0.234, 'h_b': 0.2076, 'K_s': 1.89e-6},
        'sandy clay loam': {'lambda': 0.319, 'h_b': 0.2808, 'K_s': 1.19e-6},
        'clay loam': {'lambda': 0.242, 'h_b': 0.2589, 'K_s': 7.22e-7},
        'silty clay loam': {'lambda': 0.177, 'h_b': 0.3256, 'K_s': 5.00e-7},
        'sandy clay': {'lambda': 0.223, 'h_b': 0.2917, 'K_s': 3.33e-7},
        'silty clay': {'lambda': 0.150, 'h_b': 0.3419, 'K_s': 2.50e-7},
        'clay': {'lambda': 0.165, 'h_b': 0.3730, 'K_s': 1.67e-7},
    }
    
    def __init__(self, domain, 
                 hydraulic_conductivity='loam',
                 soil_type='loam',
                 soil_porosity=0.45,
                 initial_soil_moisture=0.15,
                 soil_bulk_density=1590.0,
                 rock_density=2650.0,
                 volume_fraction_coarse_fragments=0.0,
                 surface_water_minimum_depth=1.0e-8,
                 infiltration_threshold=0.0001,  # minimum depth for infiltration (m)
                 description=None, label=None, logging=False, verbose=False):
        """
        Initialize Green-Ampt infiltration operator.
        
        Parameters:
        -----------
        domain : ANUGA Domain
            The computational domain
        hydraulic_conductivity : str, float, or array
            Either soil type name, constant value (m/s), or array per triangle
        soil_type : str
            Soil type for capillary pressure calculation
        soil_porosity : float
            Total porosity (0-1)
        initial_soil_moisture : float
            Initial soil moisture content (0-1)
        soil_bulk_density : float (kg/m³)
            Dry bulk density of soil
        rock_density : float (kg/m³)
            Density of rock constituent
        volume_fraction_coarse_fragments : float
            Fraction of coarse material (0-1)
        surface_water_minimum_depth : float (m)
            Minimum surface water depth for stability
        infiltration_threshold : float (m)
            Minimum water depth required for infiltration to occur (default 0.001 m = 1 mm)
        """
        self.n_triangles = domain.get_number_of_triangles()
        self._min_water = surface_water_minimum_depth
        self._infiltration_threshold = infiltration_threshold
        self.verbose = verbose
        
        # Set hydraulic conductivity
        if isinstance(hydraulic_conductivity, str):
            # Use soil type
            if hydraulic_conductivity in self.SOIL_PROPS:
                self.K_s = np.full(self.n_triangles, 
                                  self.SOIL_PROPS[hydraulic_conductivity]['K_s'])
            else:
                raise ValueError(f"Unknown soil type: {hydraulic_conductivity}")
        elif isinstance(hydraulic_conductivity, (int, float)):
            self.K_s = np.full(self.n_triangles, float(hydraulic_conductivity))
        else:
            self.K_s = np.asarray(hydraulic_conductivity)
        
        # Calculate moisture deficit
        self.moisture_deficit = self._calc_moisture_deficit(
            soil_bulk_density, rock_density, 
            volume_fraction_coarse_fragments, 
            initial_soil_moisture
        )
        
        # Calculate capillary pressure head
        if soil_type in self.SOIL_PROPS:
            props = self.SOIL_PROPS[soil_type]
            self.capillary_pressure = self._calc_pressure_head(
                props['lambda'], props['h_b']
            )
        else:
            raise ValueError(f"Unknown soil type: {soil_type}")
        
        # Calculate minimum wetting front depth to prevent infinite rates
        # L_f_min should give infiltration rate ~ 2-5x K_s at start
        # From f = K_s * (1 + ψΔθ/L_f), if we want f = 3*K_s initially:
        # 3 = 1 + ψΔθ/L_f_min  =>  L_f_min = ψΔθ/2
        self._min_wetting_front = (self.capillary_pressure * self.moisture_deficit) / 2.0
        
        # HARD-CODED: Maximum infiltration rate cap (prevent unrealistic spikes)
        self._max_infiltration_rate_mm_hr = 10 # mm/hr (for display)
        self._max_infiltration_rate = 10 * (1e-3 * 3600) # m/s (for calculation)
        
        # Initialize infiltration tracking
        # Cumulative infiltration depth (m of water)
        self.infiltration_depth = np.zeros(self.n_triangles)
        
        # Cumulative infiltrated volume (m³)
        self.infiltrated_volume = 0.0
        
        # Track ACTUAL infiltration rate used by ANUGA (mm/hr)
        self.current_infiltration_rate_mm_hr = 0.0
        self.n_cells_infiltrating = 0
        
        # Counter for debugging
        self.call_counter = 0
        
        # Initialize Rate_operator with zero rate (will update dynamically in __call__)
        anuga.Rate_operator.__init__(self, domain, rate=0.0,
                                     description=description, label=label,
                                     logging=logging, verbose=verbose)
        
        print(f"[OK] Green-Ampt Infiltration Operator initialized")
        print(f" Domain triangles: {self.n_triangles:,}")
        print(f" Hydraulic conductivity: {self.K_s.min()*1000:.6f} to {self.K_s.max()*1000:.6f} mm/s")
        print(f" Moisture deficit (Δθ): {self.moisture_deficit:.3f}")
        print(f" Capillary pressure (ψ): {self.capillary_pressure:.3f} m")
        print(f" Min wetting front depth: {self._min_wetting_front*1000:.2f} mm")
        print(f" Max infiltration rate: {self._max_infiltration_rate_mm_hr:.2f} mm/hr (constant)")
        print(f" Infiltration threshold: {self._infiltration_threshold*1000:.1f} mm")
        print(f" Method: GREEN-AMPT equation with __call__() override")
        print(f" DEBUG: First 5 __call__() invocations will print debug info")
    
    @staticmethod
    def _calc_moisture_deficit(soil_bulk_density, rock_density, 
                               volume_fraction_coarse, initial_moisture):
        """Calculate moisture deficit (Δθ = porosity - initial moisture)."""
        # Calculate porosity
        porosity = 1.0 - (soil_bulk_density / rock_density)
        porosity *= (1.0 - volume_fraction_coarse)
        
        # Moisture deficit
        return porosity - initial_moisture
    
    @staticmethod
    def _calc_pressure_head(lam, h_b):
        """
        Calculate capillary pressure head using Brooks-Corey relationship.
        
        ψ = (2 + 3λ) / (1 + 3λ) * h_b / 2
        """
        return (2.0 + 3.0 * lam) / (1.0 + 3.0 * lam) * h_b * 0.5
    
    def __call__(self):
        """
        Apply infiltration - this is called by ANUGA during evolution.
        
        This overrides Rate_operator.__call__() to implement Green-Ampt infiltration.
        Computes potential infiltration using Green-Ampt equation and
        removes water from surface (limited by available water).
        
        Only infiltrates where water depth > infiltration_threshold (default 1 mm).
        """
        self.call_counter += 1
        
        # Get current timestep
        dt = self.domain.get_timestep()
        
        if dt <= 0:
            return
        
        # Get surface water depth (stage - elevation)
        stage = self.domain.quantities['stage'].centroid_values
        elevation = self.domain.quantities['elevation'].centroid_values
        water_depth = np.maximum(stage - elevation, 0.0)
        
        # Only infiltrate where water depth exceeds threshold (default 1 mm)
        infiltration_mask = water_depth > self._infiltration_threshold
        
        # Initialize actual infiltration depth array
        actual_infilt_depth = np.zeros(self.n_triangles)
        
        # Only compute infiltration where mask is True
        if np.any(infiltration_mask):
            # Calculate wetting front depth (L_f = cumulative infiltration / moisture deficit)
            # Use MAXIMUM of (computed L_f, minimum L_f) to prevent infinite rates
            computed_L_f = self.infiltration_depth[infiltration_mask] / self.moisture_deficit
            wetting_front_depth = np.maximum(computed_L_f, self._min_wetting_front)
            
            # Green-Ampt potential infiltration rate (m/s)
            # f = K_s * (1 + ψ*Δθ / L_f)
            potential_rate = self.K_s[infiltration_mask] * (
                1.0 + (self.capillary_pressure * self.moisture_deficit) / wetting_front_depth
            )
            
            # Cap the maximum infiltration rate to prevent unrealistic spikes (constant scalar)
            potential_rate = np.minimum(potential_rate, self._max_infiltration_rate)
            
            # Available water (above minimum threshold)
            available_water = water_depth[infiltration_mask] - self._min_water
            np.clip(available_water, 0.0, None, out=available_water)
            
            # Maximum infiltration depth this timestep (limited by available water)
            potential_infilt_depth = potential_rate * dt
            actual_infilt_depth[infiltration_mask] = np.minimum(potential_infilt_depth, available_water)
        
        # Remove water from surface
        self.domain.quantities['stage'].centroid_values[:] -= actual_infilt_depth
        
        # Update cumulative infiltration
        self.infiltration_depth += actual_infilt_depth
        
        # Track total infiltrated volume
        triangle_areas = self.domain.areas
        volume_this_step = np.sum(actual_infilt_depth * triangle_areas)
        self.infiltrated_volume += volume_this_step
        
        # Store ACTUAL infiltration rate used this timestep (mm/hr)
        if np.any(infiltration_mask):
            # Calculate rate from actual_infilt_depth / dt (convert m/s to mm/hr)
            self.current_infiltration_rate_mm_hr = (actual_infilt_depth[infiltration_mask] / dt).mean() * 3600 * 1000
            self.n_cells_infiltrating = np.sum(infiltration_mask)
        else:
            self.current_infiltration_rate_mm_hr = 0.0
            self.n_cells_infiltrating = 0
        
        # Update domain's fractional step volume (for mass conservation)
        self.domain.fractional_step_volume_integral -= volume_this_step
        
        # DEBUG: Print first few calls
        if self.call_counter <= 5:
            t = self.domain.get_time()
            n_infiltrating = np.sum(infiltration_mask)
            print(f"    DEBUG __call__() #{self.call_counter}: t={t:.1f}s, dt={dt:.3f}s")
            print(f"          Triangles infiltrating: {n_infiltrating:,} / {self.n_triangles:,}")
            print(f"          Mean water depth: {water_depth.mean():.6f} m")
            if n_infiltrating > 0:
                infilt_rate_mm_hr = (actual_infilt_depth[infiltration_mask]/dt).mean()*3600*1000
                print(f"          Mean infiltration rate: {infilt_rate_mm_hr:.3f} mm/hr")
            print(f"          Volume this step: {volume_this_step:.2f} m³")
            print(f"          Total infiltrated: {self.infiltrated_volume:.2f} m³")
        
        # Periodic logging
        if self.verbose and self.call_counter % 1000 == 0:
            t = self.domain.get_time()
            n_infiltrating = np.sum(infiltration_mask)
            if n_infiltrating > 0:
                actual_rate_mm_hr = (actual_infilt_depth[infiltration_mask] / dt).mean() * 3600 * 1000
                print(f" Infiltration at t={t/3600:.1f}h: "
                      f"rate={actual_rate_mm_hr:.3f} mm/hr (avg), "
                      f"depth={self.infiltration_depth.mean()*1000:.2f} mm (cumulative avg), "
                      f"volume={self.infiltrated_volume:.1f} m³, "
                      f"cells={n_infiltrating:,}")
    
    def parallel_safe(self):
        """Required for ANUGA operator framework."""
        return True
    
    def statistics(self):
        """Return statistics about infiltration."""
        return {
            'mean_infiltration_depth_m': float(self.infiltration_depth.mean()),
            'max_infiltration_depth_m': float(self.infiltration_depth.max()),
            'total_infiltrated_volume_m3': float(self.infiltrated_volume),
            'mean_wetting_front_depth_m': float(
                (self.infiltration_depth / self.moisture_deficit).mean()
            )
        }

print("[OK] Green-Ampt Infiltration Operator class defined")
print(" Uses __call__() override (not update_rate) to match ANUGA's Rate_operator design")
print(" First 5 __call__() invocations will print debug info")
print(" FIX: Only infiltrates where water_depth > 1 mm (prevents removing initial tiny depths)")
print(" FIX 2: Uses realistic min wetting front depth")
print(" FIX 3: Uses hard-coded constant max infiltration rate (8 mm/hr )")


print("\n" + "="*70)
print("ADDING GREEN-AMPT INFILTRATION OPERATOR")
print("="*70)

# Create infiltration operator with soil parameters
infiltration_op = GreenAmptInfiltrationOperator(
domain,
hydraulic_conductivity='loam', # Can use 'sand', 'loam', 'clay', etc., or custom value in m/s
soil_type='loam',
soil_porosity=0.45,
initial_soil_moisture=0.35, # 15% initial moisture
soil_bulk_density=1590.0, # kg/m³
rock_density=2650.0, # kg/m³
volume_fraction_coarse_fragments=0.0,
surface_water_minimum_depth=1.0e-8, # meters
description="Green-Ampt soil infiltration",
label="infiltration",
verbose=True
)

print("\n[OK] Infiltration operator created")
print(f" Soil type: loam")
print(f" Infiltration will be applied INSIDE the simulation loop")
print(f" Method: Remove water from surface at each internal timestep")


# Monitoring arrays
time_series = []
max_depth_series = []
mean_depth_series = []
total_volume_series = []
timestep_series = []

# NEW: Infiltration monitoring
infiltration_depth_series = []
infiltrated_volume_series = []

# Checkpointing setup
checkpoint_counter = 0
last_checkpoint_time = 0

def save_checkpoint():
    """Save simulation state to checkpoint file."""
    global checkpoint_counter
    
    checkpoint_file = os.path.join(config.output_dir, f'checkpoint_{checkpoint_counter:03d}.pkl')
    
    checkpoint_data = {
        'time': domain.get_time(),
        'quantities': {
            'stage': domain.get_quantity('stage').centroid_values.copy(),
            'xmomentum': domain.get_quantity('xmomentum').centroid_values.copy(),
            'ymomentum': domain.get_quantity('ymomentum').centroid_values.copy(),
        },
        'timeseries': {
            'time': np.array(time_series),
            'max_depth': np.array(max_depth_series),
            'mean_depth': np.array(mean_depth_series),
            'total_volume': np.array(total_volume_series),
            'timestep': np.array(timestep_series),
            'infiltration_depth': np.array(infiltration_depth_series),
            'infiltrated_volume': np.array(infiltrated_volume_series)
        }
    }
    
    import pickle
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"\n [OK] Checkpoint saved: {checkpoint_file}")
    checkpoint_counter += 1

print("[OK] Monitoring and checkpointing configured (with infiltration tracking)")


print("\n" + "="*70)
print("STARTING SIMULATION with Rainfall + Infiltration")
print("="*70)

# Simulation parameters
yieldstep = config.output_interval_hours * 3600  # Convert to seconds
finaltime = (config.end_date - config.start_date).total_seconds()
checkpoint_interval = config.checkpoint_interval_hours * 3600  # Convert to seconds

print(f"\nSimulation parameters:")
print(f"  Duration: {finaltime/3600:.1f} hours ({finaltime/86400:.1f} days)")
print(f"  Output interval: {yieldstep/3600:.1f} hours")
print(f"  Checkpoint interval: {checkpoint_interval/3600:.1f} hours")

# Start simulation
print(f"\n{'='*70}")
print("SIMULATION RUNNING...")
print(f"{'='*70}\n")

sim_start = time.time()
step_count = 0

try:
    # Pre-compute elevation and areas (they don't change)
    elevation = domain.get_quantity('elevation').centroid_values
    areas = domain.areas
    
    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        # Get quantities (stage is the only changing quantity)
        stage = domain.get_quantity('stage').centroid_values
        
        # Vectorized depth calculation
        depth = np.maximum(stage - elevation, 0.0)
        
        # Compute statistics - INCLUDE ALL WATER DEPTHS (no filtering)
        max_depth = depth.max()
        
        # Mean depth calculation: compute BOTH wet-cells-only AND all-cells average
        wet_mask = depth > 0.00001  # 0.01mm threshold for "wet" classification
        mean_depth_wet = depth[wet_mask].mean() if wet_mask.any() else 0.0  # Mean of wet cells only
        mean_depth_all = depth.mean()  # Mean of ALL cells (includes zeros)
        
        # Volume includes ALL water (no masking)
        total_volume = (depth * areas).sum()
        current_timestep = domain.get_timestep()
        
        # Get infiltration statistics
        mean_infiltration = infiltration_op.infiltration_depth.mean()
        total_infiltrated = infiltration_op.infiltrated_volume
        
        # Get ACTUAL infiltration rate used by ANUGA (not back-calculated)
        actual_rate_mm_hr = infiltration_op.current_infiltration_rate_mm_hr
        n_infiltrating_cells = infiltration_op.n_cells_infiltrating
        
        # Store timeseries - save the all-cells mean for consistency with volume
        time_series.append(t)
        max_depth_series.append(max_depth)
        mean_depth_series.append(mean_depth_all)  # CHANGED: Save all-cells mean
        total_volume_series.append(total_volume)
        timestep_series.append(current_timestep)
        infiltration_depth_series.append(mean_infiltration)
        infiltrated_volume_series.append(total_infiltrated)
        
        # Progress report - show both means for clarity
        step_count += 1
        elapsed = time.time() - sim_start
        progress = (t / finaltime) * 100
        eta = (elapsed / t) * (finaltime - t) if t > 0 else 0
        
        print(f"Time: {t/3600:.1f}h / {finaltime/3600:.1f}h ({progress:.1f}%) | "
              f"Max depth: {max_depth:.5f}m | "
              f"Mean depth (all): {mean_depth_all:.5f}m | "
              f"Mean depth (wet): {mean_depth_wet:.5f}m | "
              f"Volume: {total_volume:.4f}m³ | "
              f"Infiltrated: {total_infiltrated:.1f}m³ ({mean_infiltration*1000:.2f}mm avg) | "
              f"Rate: {actual_rate_mm_hr:.2f}mm/hr ({n_infiltrating_cells:,} cells) | "
              f"dt: {current_timestep:.5f}s | "
              f"ETA: {eta/60:.0f}min")
        
        # Checkpoint
        if config.save_checkpoints and (t - last_checkpoint_time) >= checkpoint_interval:
            save_checkpoint()
            last_checkpoint_time = t
    
    # Save timeseries data
    np.savez(os.path.join(config.output_dir, 'timeseries_data.npz'),
             time=np.array(time_series),
             max_depth=np.array(max_depth_series),
             mean_depth=np.array(mean_depth_series),  # This is mean of ALL cells
             total_volume=np.array(total_volume_series),
             timestep=np.array(timestep_series),
             infiltration_depth=np.array(infiltration_depth_series),
             infiltrated_volume=np.array(infiltrated_volume_series))
    
    # Final save
    save_checkpoint()
    np.savez(os.path.join(config.output_dir, 'timeseries_data.npz'),
             time=np.array(time_series),
             max_depth=np.array(max_depth_series),
             mean_depth=np.array(mean_depth_series),  # This is mean of ALL cells
             total_volume=np.array(total_volume_series),
             timestep=np.array(timestep_series),
             infiltration_depth=np.array(infiltration_depth_series),
             infiltrated_volume=np.array(infiltrated_volume_series))
    
    total_time = time.time() - sim_start
    
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Time steps: {step_count}")
    print(f"Average time per step: {total_time/step_count:.2f} seconds")
    print(f"\nWater Balance:")
    print(f"  Surface water volume: {total_volume:.1f} m³")
    print(f"  Infiltrated volume: {total_infiltrated:.1f} m³")
    print(f"  Mean infiltration depth: {mean_infiltration*1000:.2f} mm")
    print(f"\nResults saved to: {config.output_dir}/")
    print(f"  - {config.simulation_name}.sww (full spatial results)")
    print(f"  - timeseries_data.npz (time series + infiltration data)")
    print(f"  - checkpoint_*.pkl (checkpoints)")

except KeyboardInterrupt:
    print("\n\n[WARNING] SIMULATION INTERRUPTED BY USER")
    print("Saving current state...")
    save_checkpoint()
    np.savez(os.path.join(config.output_dir, 'timeseries_data.npz'),
             time=np.array(time_series),
             max_depth=np.array(max_depth_series),
             mean_depth=np.array(mean_depth_series),  # This is mean of ALL cells
             total_volume=np.array(total_volume_series),
             timestep=np.array(timestep_series),
             infiltration_depth=np.array(infiltration_depth_series),
             infiltrated_volume=np.array(infiltrated_volume_series))
    print("[OK] State saved successfully")
    

print("\n" + "="*70)
print("STARTING SIMULATION with Rainfall + Infiltration")
print("="*70)

yieldstep = config.output_interval_hours * 3600  # Convert to seconds
finaltime = (config.end_date - config.start_date).total_seconds()
checkpoint_interval = config.checkpoint_interval_hours * 3600  # Convert to seconds

print(f"\nSimulation parameters:")
print(f"  Start date: {config.start_date}")
print(f"  End date: {config.end_date}")
print(f"  Duration: {finaltime/86400:.2f} days ({finaltime/3600:.1f} hours)")
print(f"  Yieldstep: {yieldstep/3600:.1f} hours")
print(f"  Checkpoint interval: {checkpoint_interval/86400:.1f} days")

# ========== CHECKPOINT RECOVERY ==========
print(f"\n{'='*70}")
print("CHECKING FOR EXISTING CHECKPOINTS")
print(f"{'='*70}")

import glob
import pickle

resume_from_checkpoint = False
resume_time = 0.0

# Search for checkpoint files
checkpoint_pattern = os.path.join(config.output_dir, 'checkpoint_*.pkl')
checkpoint_files = glob.glob(checkpoint_pattern)

if checkpoint_files:
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"\nFound {len(checkpoint_files)} checkpoint file(s)")
    print(f"Latest checkpoint: {os.path.basename(latest_checkpoint)}")
    print(f"  Modified: {time.ctime(os.path.getmtime(latest_checkpoint))}")
    
    try:
        # Load checkpoint data
        with open(latest_checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        resume_time = checkpoint_data['time']
        print(f"\n✓ Checkpoint loaded successfully!")
        print(f"  Simulation time: {resume_time/3600:.2f} hours ({resume_time/86400:.2f} days)")
        
        # Restore domain state
        domain.set_quantity('stage', checkpoint_data['quantities']['stage'], location='centroids')
        domain.set_quantity('xmomentum', checkpoint_data['quantities']['xmomentum'], location='centroids')
        domain.set_quantity('ymomentum', checkpoint_data['quantities']['ymomentum'], location='centroids')
        domain.set_time(resume_time)
        
        print(f"  Restored domain quantities (stage, xmomentum, ymomentum)")
        
        # Restore timeseries data
        ts = checkpoint_data['timeseries']
        time_series.extend(ts['time'].tolist())
        max_depth_series.extend(ts['max_depth'].tolist())
        mean_depth_series.extend(ts['mean_depth'].tolist())
        total_volume_series.extend(ts['total_volume'].tolist())
        timestep_series.extend(ts['timestep'].tolist())
        infiltration_depth_series.extend(ts['infiltration_depth'].tolist())
        infiltrated_volume_series.extend(ts['infiltrated_volume'].tolist())
        
        print(f"  Restored {len(ts['time'])} timeseries data points")
        
        # Update checkpoint counter (extract number from filename)
        checkpoint_num = int(os.path.basename(latest_checkpoint).split('_')[1].split('.')[0])
        checkpoint_counter = checkpoint_num + 1
        last_checkpoint_time = resume_time
        
        print(f"  Next checkpoint will be #{checkpoint_counter:03d}")
        
        # Calculate remaining time
        remaining_time = finaltime - resume_time
        print(f"\n📊 Resuming simulation:")
        print(f"  Completed: {resume_time/86400:.2f} / {finaltime/86400:.2f} days ({100*resume_time/finaltime:.1f}%)")
        print(f"  Remaining: {remaining_time/86400:.2f} days ({remaining_time/3600:.1f} hours)")
        
        resume_from_checkpoint = True
        
    except Exception as e:
        print(f"\n⚠ Error loading checkpoint: {e}")
        print("  Starting fresh simulation...")
        resume_from_checkpoint = False
else:
    print("\nNo checkpoint files found. Starting fresh simulation.")

print(f"{'='*70}\n")

# ========== START OR RESUME SIMULATION ==========
sim_start = time.time()
step_count = 0
last_print_time = sim_start
next_checkpoint = last_checkpoint_time + checkpoint_interval if resume_from_checkpoint else checkpoint_interval

if resume_from_checkpoint:
    print(f"🔄 RESUMING from {resume_time/3600:.2f} hours")
else:
    print(f"🚀 STARTING fresh simulation")

print(f"{'='*70}\n")

try:
    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        step_count += 1
        
        # Get current state
        stage = domain.get_quantity('stage').centroid_values
        elevation = domain.get_quantity('elevation').centroid_values
        depth = stage - elevation
        depth = np.maximum(depth, 0)  # No negative depths
        
        # Statistics (all cells)
        max_depth = depth.max()
        mean_depth_all = depth.mean()
        
        # Wet cells only (for meaningful statistics)
        wet_mask = depth > 1e-4
        n_wet = wet_mask.sum()
        mean_depth_wet = depth[wet_mask].mean() if n_wet > 0 else 0.0
        
        # Total volume
        areas = domain.get_quantity('elevation').get_areas()
        volume = (depth * areas).sum()
        
        # Get infiltration stats from operator
        current_infil_rate = infiltration_op.current_infiltration_rate_mm_hr
        n_infiltrating = infiltration_op.n_cells_infiltrating
        total_infiltrated_volume = infiltration_op.infiltrated_volume
        mean_infiltration_depth = infiltration_op.infiltration_depth.mean()
        
        # Store in arrays
        time_series.append(t)
        max_depth_series.append(max_depth)
        mean_depth_series.append(mean_depth_all)  # Save all-cells mean
        total_volume_series.append(volume)
        timestep_series.append(domain.timestep)
        infiltration_depth_series.append(mean_infiltration_depth)
        infiltrated_volume_series.append(total_infiltrated_volume)
        
        # Print progress
        current_time = time.time()
        if step_count % 10 == 0 or (current_time - last_print_time) > 60:
            elapsed = current_time - sim_start
            speed_ratio = (t - resume_time) / elapsed if elapsed > 0 else 0
            
            print(f"\n⏱  t = {t/3600:8.2f} hrs ({t/86400:6.2f} days) | Step {step_count:5d}")
            print(f"   💧 Max depth: {max_depth:8.4f} m | Mean depth (all): {mean_depth_all:8.4f} m | Mean depth (wet): {mean_depth_wet:8.4f} m")
            print(f"   🌊 Total volume: {volume/1e6:10.3f} M m³ | Wet cells: {n_wet:8d} / {len(depth):8d} ({100*n_wet/len(depth):5.2f}%)")
            print(f"   🌧  Infiltration rate: {current_infil_rate:6.3f} mm/hr | Infiltrating cells: {n_infiltrating:8d} ({100*n_infiltrating/len(depth):5.2f}%)")
            print(f"   📉 Total infiltrated: {total_infiltrated_volume/1e6:10.3f} M m³ | Mean infiltration depth: {mean_infiltration_depth*1000:8.3f} mm")
            print(f"   🕐 Timestep: {domain.timestep:8.3f} s | Speed: {speed_ratio:6.2f}x | Elapsed: {elapsed/60:8.2f} min")
            
            last_print_time = current_time
        
        # Save checkpoint if needed
        if config.save_checkpoints and t >= next_checkpoint:
            save_checkpoint()
            next_checkpoint = t + checkpoint_interval
            print(f"\n💾 Checkpoint #{checkpoint_counter-1:03d} saved at t={t/3600:.2f} hours")

except KeyboardInterrupt:
    print("\n\n⚠️  Simulation interrupted by user")
    print("💾 Saving final checkpoint...")
    save_checkpoint()
    print("✓ Checkpoint saved. You can resume from this point later.")
    raise

sim_end = time.time()
elapsed = sim_end - sim_start

print("\n" + "="*70)
print("SIMULATION COMPLETED")
print("="*70)
print(f"Total simulation time: {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours)")
print(f"Average speed ratio: {finaltime/elapsed:.2f}x (simulated time / real time)")
print(f"Final time: {t/86400:.2f} days")
print(f"Total steps: {step_count}")
print(f"Final max depth: {max_depth:.4f} m")
print(f"Final mean depth (all cells): {mean_depth_all:.4f} m")
print(f"Final mean depth (wet cells): {mean_depth_wet:.4f} m")
print(f"Final volume: {volume/1e6:.3f} M m³")
print(f"Total infiltrated volume: {total_infiltrated_volume/1e6:.3f} M m³")
print(f"Mean infiltration depth: {mean_infiltration_depth*1000:.3f} mm")
print("="*70)