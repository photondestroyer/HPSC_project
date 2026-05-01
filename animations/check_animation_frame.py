#!/usr/bin/env python3
"""
Quick utility to check if animation has single colorbar
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to animation
animation_path = "rainfall_infiltration_runoff/animations/water_depth.gif"

if os.path.exists(animation_path):
    print(f"✓ Found animation: {animation_path}")
    
    # Open and display first frame
    gif = Image.open(animation_path)
    
    print(f"\n  Total frames: {gif.n_frames}")
    print(f"  Size: {gif.size}")
    print(f"  Mode: {gif.mode}")
    
    # Show first frame
    gif.seek(0)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(gif)
    plt.axis('off')
    plt.title('First Frame of Animation - Check for Single Colorbar', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    preview_path = "rainfall_infiltration_runoff/animations/preview_first_frame.png"
    plt.savefig(preview_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Preview saved: {preview_path}")
    
    # Check file size
    size_mb = os.path.getsize(animation_path) / 1e6
    print(f"\n  Animation file size: {size_mb:.1f} MB")
    
    if size_mb < 50:
        print("  ✓ File size looks good!")
    else:
        print("  ⚠ File size is quite large - consider reducing frames or DPI")
    
else:
    print(f"✗ Animation not found: {animation_path}")
    print("  Animation is still rendering...")
