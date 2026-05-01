#!/usr/bin/env python3
"""
Quick viewer to display animation progress and results.
"""

import os
import glob
from datetime import datetime

def format_size(bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def check_animation_status(output_dir='rainfall_runoff_only'):
    """Check status of animation files."""
    animation_dir = os.path.join(output_dir, 'animations')
    
    print("="*70)
    print("ANIMATION STATUS CHECK")
    print("="*70)
    
    if not os.path.exists(animation_dir):
        print(f"\n⏳ Animations directory not found: {animation_dir}/")
        print("  Animation script may still be running...")
        return
    
    print(f"\n✓ Found animations directory: {animation_dir}/")
    
    # List all files
    files = glob.glob(os.path.join(animation_dir, '*'))
    
    if not files:
        print("\n⏳ No animation files found yet.")
        print("  Animation script may still be rendering...")
        return
    
    print(f"\n📁 Found {len(files)} file(s):\n")
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    total_size = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        size = os.path.getsize(filepath)
        total_size += size
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        age_minutes = (datetime.now() - mod_time).total_seconds() / 60
        
        # File type indicator
        if filename.endswith('.gif'):
            icon = "🎬"
        elif filename.endswith('.png'):
            icon = "📷"
        else:
            icon = "📄"
        
        print(f"  {icon} {filename}")
        print(f"     Size: {format_size(size)}")
        print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.0f} min ago)")
        print()
    
    print(f"Total size: {format_size(total_size)}")
    
    # Check for expected files
    expected = [
        'water_depth_animation.gif',
        'flow_speed_animation.gif',
        'simulation_summary.png'
    ]
    
    found_files = [os.path.basename(f) for f in files]
    missing = [f for f in expected if f not in found_files]
    
    if missing:
        print(f"\n⏳ Still rendering:")
        for f in missing:
            print(f"  - {f}")
    else:
        print(f"\n✅ All animations complete!")
        print(f"\nTo view animations:")
        print(f"  1. Open file browser: nautilus {animation_dir}")
        print(f"  2. Or copy to current dir: cp {animation_dir}/*.gif .")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'rainfall_runoff_only'
    
    check_animation_status(output_dir)
