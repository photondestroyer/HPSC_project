#!/usr/bin/env python3
"""
Create ANUGA Simulation Animation from ALL Checkpoint Files
============================================================

This script loads ALL checkpoint files (checkpoint_000.pkl, checkpoint_001.pkl, etc.)
and creates a complete animation showing water depth evolution from start to finish.

Works with INFILTERED.py simulation output (rainfall + infiltration).

Usage:
    conda activate anugs
    python animate_from_checkpoints.py

Features:
    - Loads all checkpoints in sequence
    - Reconstructs full simulation timeline
    - Creates smooth depth animation
    - Shows rainfall and infiltration effects
    - Saves as high-quality GIF/MP4

Author: AI Assistant
Date: 2025-11-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation
from matplotlib.colors import LogNorm, Normalize
import os
import sys
import glob
import pickle
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION (Match your INFILTERED.py settings)
# ============================================================================

OUTPUT_DIR = 'rainfall_infiltration_runoff'  # Where checkpoints are saved
DEM_FILE = 'filled_smoothed_topography_EPSG32643.tif'  # For mesh reconstruction
BOUNDING_BOX_CSV = 'bounder.csv'  # Domain boundary
MAX_TRIANGLE_AREA = 25000  # Same as simulation
BUFFER_DISTANCE = -5000.0  # Same as simulation

# Animation settings
FRAMES_PER_CHECKPOINT = 10  # How many frames to interpolate between checkpoints
FPS = 10  # Frames per second in output video
DPI = 100  # Resolution
CMAP = 'Blues'  # Colormap for depth

print("="*70)
print("ANUGA ANIMATION FROM CHECKPOINTS - FULL TIME SERIES")
print("="*70)

# ============================================================================
# STEP 1: Find and Load All Checkpoints
# ============================================================================

print(f"\n{'='*70}")
print("STEP 1: LOADING CHECKPOINTS")
print(f"{'='*70}")

checkpoint_pattern = os.path.join(OUTPUT_DIR, 'checkpoint_*.pkl')
checkpoint_files = sorted(glob.glob(checkpoint_pattern))

if not checkpoint_files:
    print(f"\n❌ ERROR: No checkpoint files found in {OUTPUT_DIR}/")
    print(f"   Looking for: {checkpoint_pattern}")
    sys.exit(1)

print(f"\n✓ Found {len(checkpoint_files)} checkpoint files:")
for i, cp in enumerate(checkpoint_files):
    file_size = os.path.getsize(cp) / (1024**2)  # MB
    print(f"   {i+1}. {os.path.basename(cp)} ({file_size:.1f} MB)")

# Load all checkpoint data
print(f"\n📂 Loading checkpoint data...")
checkpoints = []
times_hours = []

for i, cp_file in enumerate(checkpoint_files):
    print(f"   Loading {os.path.basename(cp_file)}...", end=' ')
    with open(cp_file, 'rb') as f:
        data = pickle.load(f)
    
    time_hours = data['time'] / 3600.0
    times_hours.append(time_hours)
    checkpoints.append(data)
    
    print(f"✓ (t={time_hours:.1f}h, {len(data['quantities']['stage']):,} cells)")

print(f"\n✅ Loaded {len(checkpoints)} checkpoints")
print(f"   Time range: {times_hours[0]:.1f}h → {times_hours[-1]:.1f}h")
print(f"   Duration: {times_hours[-1] - times_hours[0]:.1f} hours ({(times_hours[-1] - times_hours[0])/24:.1f} days)")

# ============================================================================
# STEP 2: Reconstruct Domain Mesh (Required for Spatial Visualization)
# ============================================================================

print(f"\n{'='*70}")
print("STEP 2: RECONSTRUCTING DOMAIN MESH")
print(f"{'='*70}")

try:
    import anuga
    from shapely import wkt
    import pandas as pd
    
    print("\n📐 Loading domain boundary...")
    
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
    
    bounding_polygon = parse_qgis_bounding_box(BOUNDING_BOX_CSV, BUFFER_DISTANCE)
    print(f"   ✓ Loaded boundary ({len(bounding_polygon)} vertices)")
    
    print("\n🔨 Creating mesh (this may take a minute)...")
    num_segments = len(bounding_polygon)
    tags = {'exterior': list(range(num_segments))}
    
    start_time = time.time()
    domain = anuga.create_domain_from_regions(
        bounding_polygon,
        boundary_tags=tags,
        maximum_triangle_area=MAX_TRIANGLE_AREA
    )
    mesh_time = time.time() - start_time
    
    print(f"   ✓ Mesh created in {mesh_time:.2f}s")
    print(f"      Triangles: {domain.get_number_of_triangles():,}")
    print(f"      Vertices: {domain.get_number_of_nodes():,}")
    
    # Set coordinate reference
    domain.geo_reference.set_zone(43)
    domain.geo_reference.set_hemisphere('northern')
    
    print("\n📊 Loading topography...")
    domain.get_quantity('elevation').set_values_from_tif_file(DEM_FILE)
    elevation = domain.get_quantity('elevation').centroid_values
    print(f"   ✓ Elevation loaded (min={elevation.min():.1f}m, max={elevation.max():.1f}m)")
    
    # Get mesh coordinates for triangulation
    x = domain.centroid_coordinates[:, 0]
    y = domain.centroid_coordinates[:, 1]
    vertices = domain.get_nodes()
    triangles = domain.get_triangles()
    
    print(f"\n✅ Domain mesh ready for animation")
    
    mesh_available = True

except Exception as e:
    print(f"\n❌ ERROR: Could not reconstruct domain mesh: {e}")
    print(f"\n   To create spatial animations, we need:")
    print(f"   1. ANUGA installed (conda activate anugs)")
    print(f"   2. Same DEM file: {DEM_FILE}")
    print(f"   3. Same boundary file: {BOUNDING_BOX_CSV}")
    print(f"   4. Same mesh parameters (MAX_TRIANGLE_AREA={MAX_TRIANGLE_AREA})")
    print(f"\n   Falling back to summary plots only...")
    mesh_available = False

# ============================================================================
# STEP 3: Create Animations (if mesh available)
# ============================================================================

if not mesh_available:
    print(f"\n⚠️  Cannot create spatial animations without mesh")
    print(f"   Creating summary plots instead...")
    
    # Create summary timeseries plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simulation Summary from Checkpoints', fontsize=14, fontweight='bold')
    
    all_times = []
    all_max_depths = []
    all_mean_depths = []
    all_volumes = []
    
    for cp in checkpoints:
        if 'timeseries' in cp:
            ts = cp['timeseries']
            all_times.extend(ts['time'] / 3600)
            all_max_depths.extend(ts['max_depth'])
            all_mean_depths.extend(ts['mean_depth'])
            all_volumes.extend(ts['total_volume'])
    
    axes[0, 0].plot(all_times, all_max_depths, 'b-')
    axes[0, 0].set_title('Maximum Depth')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Depth (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(all_times, all_mean_depths, 'g-')
    axes[0, 1].set_title('Mean Depth')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Depth (m)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(all_times, np.array(all_volumes) / 1e6, 'r-')
    axes[1, 0].set_title('Total Volume')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Volume (million m³)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    info_text = f"Checkpoints: {len(checkpoints)}\n"
    info_text += f"Time range: {times_hours[0]:.1f}h - {times_hours[-1]:.1f}h\n"
    info_text += f"Duration: {(times_hours[-1] - times_hours[0])/24:.1f} days"
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'checkpoint_summary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Summary plot saved: {output_file}")
    sys.exit(0)

# ============================================================================
# STEP 4: Create Water Depth Animation
# ============================================================================

print(f"\n{'='*70}")
print("STEP 3: CREATING WATER DEPTH ANIMATION")
print(f"{'='*70}")

# Create triangulation for plotting
print("\n🔺 Creating triangulation...")
tri_obj = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
print(f"   ✓ Triangulation ready ({len(triangles):,} triangles)")

# Prepare depth data for all checkpoints
print(f"\n💧 Computing depth for all checkpoints...")
depth_data = []

for i, cp in enumerate(checkpoints):
    stage = cp['quantities']['stage']
    depth = np.maximum(stage - elevation, 0.0)
    depth_data.append(depth)
    
    max_depth = depth.max()
    wet_cells = (depth > 0.001).sum()
    print(f"   {i+1}. t={times_hours[i]:.1f}h: max_depth={max_depth:.3f}m, wet_cells={wet_cells:,}")

print(f"\n✅ Depth data ready for {len(depth_data)} checkpoints")

# Determine color scale
all_depths = np.concatenate(depth_data)
vmin = 0.0
vmax = np.percentile(all_depths[all_depths > 0], 99) if (all_depths > 0).any() else 0.1

print(f"\n🎨 Color scale: {vmin:.3f}m - {vmax:.3f}m")

# Create animation
print(f"\n🎬 Creating animation...")
print(f"   Checkpoints: {len(checkpoints)}")
print(f"   Frames per checkpoint: {FRAMES_PER_CHECKPOINT}")
print(f"   Total frames: {len(checkpoints) * FRAMES_PER_CHECKPOINT}")
print(f"   FPS: {FPS}")

# Prepare interpolated frames
print(f"\n⏳ Interpolating between checkpoints...")
interpolated_depths = []
interpolated_times = []

for i in range(len(checkpoints) - 1):
    depth_start = depth_data[i]
    depth_end = depth_data[i + 1]
    time_start = times_hours[i]
    time_end = times_hours[i + 1]
    
    for j in range(FRAMES_PER_CHECKPOINT):
        alpha = j / FRAMES_PER_CHECKPOINT
        interpolated_depth = (1 - alpha) * depth_start + alpha * depth_end
        interpolated_time = (1 - alpha) * time_start + alpha * time_end
        
        interpolated_depths.append(interpolated_depth)
        interpolated_times.append(interpolated_time)

# Add last checkpoint
interpolated_depths.append(depth_data[-1])
interpolated_times.append(times_hours[-1])

print(f"   ✓ Created {len(interpolated_depths)} interpolated frames")

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_aspect('equal')
ax.set_xlabel('Easting (m)', fontsize=14)
ax.set_ylabel('Northing (m)', fontsize=14)

# Initial plot (use tricontourf for smoother visualization)
tpc = ax.tripcolor(tri_obj, interpolated_depths[0], 
                   cmap=CMAP, vmin=vmin, vmax=vmax, 
                   shading='gouraud')  # Smooth shading

# Add colorbar (ONCE, outside animation loop to fix double colorbar issue)
cbar = plt.colorbar(tpc, ax=ax, label='Water Depth (m)', pad=0.02)

# Title with time info
title = ax.set_title(f'Water Depth | Time: {interpolated_times[0]:.1f}h ({interpolated_times[0]/24:.1f} days) | Max: {interpolated_depths[0].max():.3f}m',
                     fontsize=16, fontweight='bold', pad=20)

# Add info text
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def animate(frame):
    """Animation update function."""
    depth = interpolated_depths[frame]
    time_h = interpolated_times[frame]
    
    # Update data
    tpc.set_array(depth)
    
    # Update title
    max_depth = depth.max()
    wet_cells = (depth > 0.001).sum()
    title.set_text(f'Water Depth | Time: {time_h:.1f}h ({time_h/24:.1f} days) | Max: {max_depth:.3f}m')
    
    # Update info text
    info = f"Frame {frame+1}/{len(interpolated_depths)}\n"
    info += f"Wet cells: {wet_cells:,}\n"
    info += f"Mean depth: {depth[depth > 0.001].mean():.4f}m" if wet_cells > 0 else "Mean depth: 0.0000m"
    info_text.set_text(info)
    
    return tpc, title, info_text

# Create animation
print(f"\n🎥 Rendering animation (this will take several minutes)...")
anim = FuncAnimation(fig, animate, frames=len(interpolated_depths),
                    interval=1000/FPS, blit=False)

# Save as GIF
output_file = os.path.join(OUTPUT_DIR, 'water_depth_animation_from_checkpoints.gif')
print(f"\n💾 Saving animation to: {output_file}")
print(f"   This will take 5-15 minutes depending on mesh size...")

writer = PillowWriter(fps=FPS)
anim.save(output_file, writer=writer, dpi=DPI)

print(f"\n✅ Animation saved successfully!")
print(f"   File: {output_file}")
print(f"   Size: {os.path.getsize(output_file) / (1024**2):.1f} MB")
print(f"   Duration: {len(interpolated_depths)/FPS:.1f} seconds")
print(f"   Frames: {len(interpolated_depths)}")

plt.close()

# ============================================================================
# STEP 5: Create Summary Statistics Plot
# ============================================================================

print(f"\n{'='*70}")
print("STEP 4: CREATING SUMMARY STATISTICS")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Rainfall-Infiltration Simulation Summary', fontsize=16, fontweight='bold')

all_times = []
all_max_depths = []
all_mean_depths = []
all_volumes = []

for cp in checkpoints:
    if 'timeseries' in cp:
        ts = cp['timeseries']
        all_times.extend(ts['time'] / 3600)
        all_max_depths.extend(ts['max_depth'])
        all_mean_depths.extend(ts['mean_depth'])
        all_volumes.extend(ts['total_volume'])

# Max depth
axes[0, 0].plot(all_times, all_max_depths, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (hours)', fontsize=12)
axes[0, 0].set_ylabel('Maximum Depth (m)', fontsize=12)
axes[0, 0].set_title('Maximum Water Depth', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Mean depth
axes[0, 1].plot(all_times, all_mean_depths, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time (hours)', fontsize=12)
axes[0, 1].set_ylabel('Mean Depth (m)', fontsize=12)
axes[0, 1].set_title('Mean Water Depth', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Volume
axes[1, 0].plot(all_times, np.array(all_volumes) / 1e6, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time (hours)', fontsize=12)
axes[1, 0].set_ylabel('Volume (million m³)', fontsize=12)
axes[1, 0].set_title('Total Water Volume', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Info
axes[1, 1].axis('off')
info_text = f"Simulation Summary\n\n"
info_text += f"Checkpoints analyzed: {len(checkpoints)}\n"
info_text += f"Time range: {times_hours[0]:.1f}h - {times_hours[-1]:.1f}h\n"
info_text += f"Duration: {(times_hours[-1] - times_hours[0])/24:.1f} days\n\n"
info_text += f"Peak depth: {max(all_max_depths):.3f}m\n"
info_text += f"Peak volume: {max(all_volumes)/1e6:.1f} million m³\n"
info_text += f"Domain cells: {len(elevation):,}\n"
info_text += f"Mesh area: {domain.get_area()/1e6:.2f} km²"

axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
summary_file = os.path.join(OUTPUT_DIR, 'simulation_summary_from_checkpoints.png')
plt.savefig(summary_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Summary plot saved: {summary_file}")

# ============================================================================
# COMPLETE
# ============================================================================

print(f"\n{'='*70}")
print("✅ ANIMATION CREATION COMPLETE!")
print(f"{'='*70}")
print(f"\nOutput files:")
print(f"  1. {output_file}")
print(f"  2. {summary_file}")
print(f"\nTo view:")
print(f"  - Open GIF in browser or image viewer")
print(f"  - Use 'eog {output_file}' (Linux)")
print(f"  - Use 'open {output_file}' (Mac)")
print(f"\n{'='*70}\n")
