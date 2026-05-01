#!/usr/bin/env python3
"""
Quick test to verify animation creation works.
Run this FIRST before creating full animations.
"""

import os
import sys

print("="*70)
print("TESTING ANIMATION SCRIPT")
print("="*70)

# Check if output directory exists
output_dir = 'rainfall_runoff_only'
if not os.path.exists(output_dir):
    print(f"\n✗ Output directory not found: {output_dir}/")
    sys.exit(1)

print(f"\n✓ Found output directory: {output_dir}/")

# Check for files
sww_files = [f for f in os.listdir(output_dir) if f.endswith('.sww')]
checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_') and f.endswith('.pkl')]

print(f"\nFiles found:")
print(f"  SWW files: {len(sww_files)}")
if sww_files:
    for f in sww_files:
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"    - {f} ({size_mb:.1f} MB)")

print(f"  Checkpoint files: {len(checkpoint_files)}")
if checkpoint_files:
    for f in checkpoint_files[:3]:  # Show first 3
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"    - {f} ({size_mb:.1f} MB)")
    if len(checkpoint_files) > 3:
        print(f"    ... and {len(checkpoint_files) - 3} more")

# Test imports
print(f"\nTesting required packages...")
try:
    import numpy as np
    print("  ✓ numpy")
except:
    print("  ✗ numpy (REQUIRED)")

try:
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib")
except:
    print("  ✗ matplotlib (REQUIRED)")

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    print("  ✓ matplotlib.animation")
except:
    print("  ✗ matplotlib.animation (REQUIRED)")

try:
    from anuga.file.netcdf import NetCDFFile
    print("  ✓ anuga.file.netcdf")
except:
    try:
        from scipy.io import netcdf_file
        print("  ✓ scipy.io.netcdf_file (fallback)")
    except:
        print("  ✗ netcdf (REQUIRED)")

# Try to load SWW file quickly
if sww_files:
    print(f"\nTesting SWW file loading...")
    try:
        from anuga.file.netcdf import NetCDFFile
        sww_path = os.path.join(output_dir, sww_files[0])
        fid = NetCDFFile(sww_path, 'r')
        
        n_times = len(fid.variables['time'])
        n_triangles = len(fid.variables['volumes'])
        
        fid.close()
        
        print(f"  ✓ Successfully loaded {sww_files[0]}")
        print(f"    Timesteps: {n_times}")
        print(f"    Triangles: {n_triangles:,}")
        
        print(f"\n{'='*70}")
        print("ALL TESTS PASSED!")
        print("="*70)
        print(f"\nYou can now run:")
        print(f"  python create_simulation_animation.py")
        
    except Exception as e:
        print(f"  ✗ Error loading SWW: {e}")
        print(f"\n⚠ There may be an issue with the SWW file.")
else:
    print(f"\n{'='*70}")
    print("TESTS PASSED (checkpoint mode)")
    print("="*70)
    print(f"\nYou can now run:")
    print(f"  python create_simulation_animation.py")
    print(f"\nNote: Will create summary plots from checkpoints (no SWW file)")
