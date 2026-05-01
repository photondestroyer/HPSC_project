#!/usr/bin/env python3
"""
ANUGA Rainfall-Infiltration Simulation Animation Creator
=========================================================

Creates animations from ANUGA simulation results. Works with:
1. SWW files (if domain.set_store(True) was enabled)
2. Checkpoint files (if SWW not available)
3. Reconstructs spatial data from checkpoints

Designed for INFILTERED.py simulation results.

Author: AI Assistant
Date: 2025-11-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path
import os
import sys
import glob
import pickle
import time as pytime

print("="*70)
print("ANUGA SIMULATION ANIMATION CREATOR")
print("="*70)


class CheckpointAnimator:
    """
    Create animations from ANUGA checkpoint files when SWW is not available.
    """
    
    def __init__(self, output_dir, checkpoint_pattern='checkpoint_*.pkl'):
        """
        Initialize animator with checkpoint files.
        
        Parameters:
        -----------
        output_dir : str
            Directory containing checkpoint files
        checkpoint_pattern : str
            Glob pattern for checkpoint files
        """
        self.output_dir = output_dir
        self.animation_dir = os.path.join(output_dir, 'animations')
        os.makedirs(self.animation_dir, exist_ok=True)
        
        # Find checkpoint files
        checkpoint_files = glob.glob(os.path.join(output_dir, checkpoint_pattern))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {output_dir}/")
        
        # Sort by checkpoint number
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        self.checkpoint_files = checkpoint_files
        print(f"\n✓ Found {len(checkpoint_files)} checkpoint files")
        print(f"  Range: {os.path.basename(checkpoint_files[0])} to {os.path.basename(checkpoint_files[-1])}")
        
        # Load first checkpoint to get domain info
        self._load_domain_geometry()
    
    def _load_domain_geometry(self):
        """
        Load domain geometry from ANUGA domain reconstruction.
        This requires recreating the domain with the same parameters.
        """
        print("\n⚠ WARNING: To create animations from checkpoints, we need domain geometry.")
        print("  Please ensure you have the same configuration as the original simulation.")
        print("\nAttempting to load domain geometry...")
        
        # Try to import configuration
        try:
            # Attempt to load from INFILTERED.py's config
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # This is a simplified approach - you may need to adjust based on your setup
            print("\n  Option 1: Reconstruct domain using original configuration")
            print("  Option 2: Extract mesh from existing domain.sww file")
            print("\n  For now, showing you how to use this with SWW file fallback...")
            
            self.domain_available = False
            
        except Exception as e:
            print(f"  Could not reconstruct domain: {e}")
            self.domain_available = False
    
    def load_checkpoint_data(self, checkpoint_file):
        """Load data from a single checkpoint file."""
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def create_checkpoint_summary_plot(self):
        """
        Create a summary plot showing evolution across all checkpoints.
        """
        print(f"\n{'='*70}")
        print("CREATING CHECKPOINT SUMMARY PLOT")
        print(f"{'='*70}")
        
        times = []
        max_depths = []
        mean_depths = []
        volumes = []
        infiltration_depths = []
        infiltrated_volumes = []
        
        # Load timeseries from each checkpoint
        for cp_file in self.checkpoint_files:
            data = self.load_checkpoint_data(cp_file)
            
            if 'timeseries' in data:
                ts = data['timeseries']
                times.extend(ts['time'] / 3600)  # Convert to hours
                max_depths.extend(ts['max_depth'])
                mean_depths.extend(ts['mean_depth'])
                volumes.extend(ts['total_volume'])
                
                if 'infiltration_depth' in ts:
                    infiltration_depths.extend(ts['infiltration_depth'])
                if 'infiltrated_volume' in ts:
                    infiltrated_volumes.extend(ts['infiltrated_volume'])
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Rainfall-Infiltration Simulation Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Max Depth
        ax1 = axes[0, 0]
        ax1.plot(times, max_depths, 'b-', linewidth=2)
        ax1.set_xlabel('Time (hours)', fontsize=11)
        ax1.set_ylabel('Maximum Depth (m)', fontsize=11)
        ax1.set_title('Maximum Water Depth', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean Depth
        ax2 = axes[0, 1]
        ax2.plot(times, mean_depths, 'g-', linewidth=2)
        ax2.set_xlabel('Time (hours)', fontsize=11)
        ax2.set_ylabel('Mean Depth (m)', fontsize=11)
        ax2.set_title('Mean Water Depth', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total Volume
        ax3 = axes[0, 2]
        ax3.plot(times, np.array(volumes)/1e6, 'r-', linewidth=2)
        ax3.set_xlabel('Time (hours)', fontsize=11)
        ax3.set_ylabel('Volume (million m³)', fontsize=11)
        ax3.set_title('Total Water Volume', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Infiltration Depth
        if infiltration_depths:
            ax4 = axes[1, 0]
            ax4.plot(times, np.array(infiltration_depths)*1000, 'purple', linewidth=2)
            ax4.set_xlabel('Time (hours)', fontsize=11)
            ax4.set_ylabel('Infiltration Depth (mm)', fontsize=11)
            ax4.set_title('Mean Infiltration Depth', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Infiltration Data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Plot 5: Infiltrated Volume
        if infiltrated_volumes:
            ax5 = axes[1, 1]
            ax5.plot(times, np.array(infiltrated_volumes)/1e6, 'orange', linewidth=2)
            ax5.set_xlabel('Time (hours)', fontsize=11)
            ax5.set_ylabel('Infiltrated Volume (million m³)', fontsize=11)
            ax5.set_title('Total Infiltrated Volume', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Infiltration Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Plot 6: Water Balance
        if infiltrated_volumes:
            ax6 = axes[1, 2]
            ax6.plot(times, np.array(volumes)/1e6, 'b-', linewidth=2, label='Surface Water')
            ax6.plot(times, np.array(infiltrated_volumes)/1e6, 'brown', linewidth=2, label='Infiltrated')
            ax6.plot(times, (np.array(volumes) + np.array(infiltrated_volumes))/1e6, 
                    'k--', linewidth=2, label='Total')
            ax6.set_xlabel('Time (hours)', fontsize=11)
            ax6.set_ylabel('Volume (million m³)', fontsize=11)
            ax6.set_title('Water Balance', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Water Balance Data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.animation_dir, 'simulation_summary.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Summary plot saved: {output_file}")
        
        file_size_mb = os.path.getsize(output_file) / 1e6
        print(f"  File size: {file_size_mb:.1f} MB")
        
        plt.close()
        
        return output_file
    
    def create_spatial_animation_from_checkpoints(self, 
                                                   frames_per_checkpoint=10,
                                                   fps=10, dpi=100):
        """
        Create spatial animation by loading all checkpoints and reconstructing domain.
        """
        print(f"\n{'='*70}")
        print("CREATING SPATIAL ANIMATION FROM CHECKPOINTS")
        print(f"{'='*70}")
        
        # Import required libraries
        try:
            import anuga
            from shapely import wkt
            import pandas as pd
        except ImportError as e:
            print(f"❌ ERROR: Missing required library: {e}")
            print("   Install with: conda install -c conda-forge anuga shapely pandas")
            return None
        
        # Configuration (must match INFILTERED.py)
        DEM_FILE = 'filled_smoothed_topography_EPSG32643.tif'
        BOUNDING_BOX_CSV = 'bounder.csv'
        MAX_TRIANGLE_AREA = 25000
        BUFFER_DISTANCE = -5000.0
        
        print(f"\n📐 Reconstructing domain geometry...")
        
        # Parse boundary
        def parse_qgis_bounding_box(csv_path, buffer_distance=-5000.0):
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
        
        try:
            bounding_polygon = parse_qgis_bounding_box(BOUNDING_BOX_CSV, BUFFER_DISTANCE)
            print(f"   ✓ Boundary loaded ({len(bounding_polygon)} vertices)")
        except Exception as e:
            print(f"   ❌ Failed to load boundary: {e}")
            return None
        
        # Create mesh
        print(f"\n🔨 Creating mesh...")
        num_segments = len(bounding_polygon)
        tags = {'exterior': list(range(num_segments))}
        
        import time as pytime
        start_time = pytime.time()
        domain = anuga.create_domain_from_regions(
            bounding_polygon,
            boundary_tags=tags,
            maximum_triangle_area=MAX_TRIANGLE_AREA
        )
        mesh_time = pytime.time() - start_time
        
        print(f"   ✓ Mesh created in {mesh_time:.2f}s")
        print(f"      Triangles: {domain.get_number_of_triangles():,}")
        print(f"      Vertices: {domain.get_number_of_nodes():,}")
        
        # Set coordinate reference
        domain.geo_reference.set_zone(43)
        domain.geo_reference.set_hemisphere('northern')
        
        # Load topography
        print(f"\n📊 Loading topography...")
        domain.get_quantity('elevation').set_values_from_tif_file(DEM_FILE)
        elevation = domain.get_quantity('elevation').centroid_values
        print(f"   ✓ Elevation loaded (min={elevation.min():.1f}m, max={elevation.max():.1f}m)")
        
        # Get mesh for triangulation
        vertices = domain.get_nodes()
        triangles = domain.get_triangles()
        
        # Load all checkpoint data
        print(f"\n💧 Loading depth data from {len(self.checkpoint_files)} checkpoints...")
        depth_data = []
        times_hours = []
        
        for i, cp_file in enumerate(self.checkpoint_files):
            data = self.load_checkpoint_data(cp_file)
            stage = data['quantities']['stage']
            depth = np.maximum(stage - elevation, 0.0)
            depth_data.append(depth)
            
            time_h = data['time'] / 3600.0
            times_hours.append(time_h)
            
            max_depth = depth.max()
            wet_cells = (depth > 0.001).sum()
            print(f"   {i+1}. t={time_h:.1f}h: max={max_depth:.3f}m, wet={wet_cells:,}")
        
        # Interpolate between checkpoints
        print(f"\n⏳ Interpolating frames (FRAMES_PER_CHECKPOINT={frames_per_checkpoint})...")
        interpolated_depths = []
        interpolated_times = []
        
        for i in range(len(depth_data) - 1):
            depth_start = depth_data[i]
            depth_end = depth_data[i + 1]
            time_start = times_hours[i]
            time_end = times_hours[i + 1]
            
            for j in range(frames_per_checkpoint):
                alpha = j / frames_per_checkpoint
                interp_depth = (1 - alpha) * depth_start + alpha * depth_end
                interp_time = (1 - alpha) * time_start + alpha * time_end
                
                interpolated_depths.append(interp_depth)
                interpolated_times.append(interp_time)
        
        # Add last checkpoint
        interpolated_depths.append(depth_data[-1])
        interpolated_times.append(times_hours[-1])
        
        print(f"   ✓ Created {len(interpolated_depths)} frames total")
        
        # Determine color scale
        all_depths = np.concatenate(depth_data)
        vmin = 0.0
        vmax = np.percentile(all_depths[all_depths > 0], 99) if (all_depths > 0).any() else 0.1
        print(f"\n🎨 Color scale: {vmin:.3f}m - {vmax:.3f}m")
        
        # Create animation
        print(f"\n🎬 Rendering animation...")
        print(f"   Total frames: {len(interpolated_depths)}")
        print(f"   FPS: {fps}")
        print(f"   Duration: {len(interpolated_depths)/fps:.1f} seconds")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        
        # Initial plot (use 'flat' shading for centroid-based data)
        tpc = ax.tripcolor(triang, interpolated_depths[0], 
                          cmap='Blues', vmin=vmin, vmax=vmax, 
                          shading='flat')  # FIXED: use 'flat' not 'gouraud'
        
        # Colorbar (created once)
        cbar = plt.colorbar(tpc, ax=ax, label='Water Depth (m)', pad=0.02)
        
        # Title and info
        title = ax.set_title(f'Water Depth | Time: {interpolated_times[0]:.1f}h',
                            fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Easting (m)', fontsize=14)
        ax.set_ylabel('Northing (m)', fontsize=14)
        ax.set_aspect('equal')
        
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            depth = interpolated_depths[frame]
            time_h = interpolated_times[frame]
            
            # Update data
            tpc.set_array(depth)
            
            # Update title
            max_depth = depth.max()
            title.set_text(f'Water Depth | Time: {time_h:.1f}h ({time_h/24:.1f} days) | Max: {max_depth:.3f}m')
            
            # Update info
            wet_cells = (depth > 0.001).sum()
            mean_depth = depth[depth > 0.001].mean() if wet_cells > 0 else 0.0
            info = f"Frame {frame+1}/{len(interpolated_depths)}\n"
            info += f"Wet cells: {wet_cells:,}\n"
            info += f"Mean depth: {mean_depth:.4f}m"
            info_text.set_text(info)
            
            if frame % 10 == 0:
                print(f"   Progress: {frame+1}/{len(interpolated_depths)} frames ({(frame+1)/len(interpolated_depths)*100:.1f}%)")
            
            return tpc, title, info_text
        
        anim = FuncAnimation(fig, animate, frames=len(interpolated_depths),
                            interval=1000/fps, blit=False)
        
        # Save
        output_file = os.path.join(self.animation_dir, 'water_depth_from_checkpoints.gif')
        print(f"\n💾 Saving to: {output_file}")
        print(f"   This will take 5-15 minutes...")
        
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        
        plt.close()
        
        file_size_mb = os.path.getsize(output_file) / 1e6
        print(f"\n✅ Animation saved!")
        print(f"   File: {output_file}")
        print(f"   Size: {file_size_mb:.1f} MB")
        
        return output_file


class SWWAnimator:
    """
    Create animations from ANUGA SWW files.
    """
    
    def __init__(self, sww_file, output_dir=None):
        """
        Initialize animator with SWW file.
        
        Parameters:
        -----------
        sww_file : str
            Path to ANUGA .sww file
        output_dir : str
            Directory for output animations (defaults to same dir as SWW)
        """
        self.sww_file = sww_file
        
        if output_dir is None:
            output_dir = os.path.dirname(sww_file)
        
        self.animation_dir = os.path.join(output_dir, 'animations')
        os.makedirs(self.animation_dir, exist_ok=True)
        
        print(f"\n✓ Loading SWW file: {sww_file}")
        self._load_sww()
        
        print(f"  Loaded {len(self.time)} timesteps")
        print(f"  Domain: {self.n_triangles:,} triangles, {self.n_vertices:,} vertices")
        print(f"  Time range: 0 to {self.time[-1]/3600:.1f} hours ({self.time[-1]/86400:.1f} days)")
    
    def _load_sww(self):
        """Load data from SWW file."""
        try:
            from anuga.file.netcdf import NetCDFFile
        except:
            from scipy.io import netcdf_file as NetCDFFile
        
        fid = NetCDFFile(self.sww_file, 'r')
        
        # Mesh
        self.x = np.array(fid.variables['x'][:])
        self.y = np.array(fid.variables['y'][:])
        self.volumes = np.array(fid.variables['volumes'][:])
        
        # Time
        self.time = np.array(fid.variables['time'][:])
        
        # Quantities
        self.stage = np.array(fid.variables['stage'][:])
        self.elevation = np.array(fid.variables['elevation'][:])
        self.xmomentum = np.array(fid.variables['xmomentum'][:])
        self.ymomentum = np.array(fid.variables['ymomentum'][:])
        
        fid.close()
        
        # Compute depth
        self.depth = np.maximum(self.stage - self.elevation, 0)
        
        # Dimensions
        self.n_timesteps = len(self.time)
        self.n_vertices = len(self.x)
        self.n_triangles = len(self.volumes)
    
    def create_depth_animation(self, output_gif='water_depth.gif', 
                               frames=50, fps=5, 
                               depth_vmax=None,
                               figsize=(14, 10), dpi=100):
        """
        Create water depth animation.
        """
        print(f"\n{'='*70}")
        print("CREATING WATER DEPTH ANIMATION")
        print(f"{'='*70}")
        print(f"  Frames: {frames} at {fps} fps")
        
        # Select timesteps
        step = max(1, self.n_timesteps // frames)
        frame_indices = range(0, self.n_timesteps, step)
        
        # Determine color scale
        if depth_vmax is None:
            sample_depths = self.depth[::max(1, self.n_timesteps//10), :]
            depth_vmax = np.percentile(sample_depths[sample_depths > 0], 99) if (sample_depths > 0).any() else 1.0
        
        print(f"  Depth color scale: 0 to {depth_vmax:.3f} m")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        # Create initial plot for colorbar
        tcf_init = ax.tripcolor(triang, self.depth[0], 
                                vmin=0, vmax=depth_vmax, 
                                cmap='Blues', shading='flat')
        cbar = plt.colorbar(tcf_init, ax=ax, pad=0.02)
        cbar.set_label('Water Depth (m)', fontsize=11)
        
        def animate(frame_num):
            ax.clear()
            idx = frame_indices[frame_num]
            
            # Plot depth
            tcf = ax.tripcolor(triang, self.depth[idx], 
                              vmin=0, vmax=depth_vmax, 
                              cmap='Blues', shading='flat')
            
            # Time info
            sim_time_hours = self.time[idx] / 3600
            sim_time_days = sim_time_hours / 24
            max_depth = self.depth[idx].max()
            
            ax.set_title(f'Water Depth\n'
                        f'Time: {sim_time_hours:.1f}h ({sim_time_days:.2f} days) | '
                        f'Max Depth: {max_depth:.3f}m',
                        fontsize=13, fontweight='bold')
            
            ax.set_xlabel('Easting (m)', fontsize=11)
            ax.set_ylabel('Northing (m)', fontsize=11)
            ax.set_aspect('equal')
            ax.ticklabel_format(style='plain', useOffset=False)
            
            return tcf,
        
        print(f"  Rendering frames...")
        anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                           interval=1000/fps, blit=False, repeat=True)
        
        output_path = os.path.join(self.animation_dir, output_gif)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✓ Animation saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def create_speed_animation(self, output_gif='flow_speed.gif',
                              frames=50, fps=5,
                              speed_vmax=None,
                              figsize=(14, 10), dpi=100):
        """
        Create flow speed animation.
        """
        print(f"\n{'='*70}")
        print("CREATING FLOW SPEED ANIMATION")
        print(f"{'='*70}")
        print(f"  Frames: {frames} at {fps} fps")
        
        # Compute speed
        depth_safe = np.maximum(self.depth, 0.001)
        u = np.divide(self.xmomentum, depth_safe, where=depth_safe>0.001, out=np.zeros_like(self.xmomentum))
        v = np.divide(self.ymomentum, depth_safe, where=depth_safe>0.001, out=np.zeros_like(self.ymomentum))
        speed = np.hypot(u, v)
        
        # Mask speed where depth is too small
        speed[self.depth < 0.001] = 0
        
        # Select timesteps
        step = max(1, self.n_timesteps // frames)
        frame_indices = range(0, self.n_timesteps, step)
        
        # Determine color scale
        if speed_vmax is None:
            sample_speeds = speed[::max(1, self.n_timesteps//10), :]
            speed_vmax = np.percentile(sample_speeds[sample_speeds > 0], 99) if (sample_speeds > 0).any() else 1.0
        
        print(f"  Speed color scale: 0 to {speed_vmax:.3f} m/s")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        # Create initial plot for colorbar
        tcf_init = ax.tripcolor(triang, speed[0], 
                                vmin=0, vmax=speed_vmax,
                                cmap='plasma', shading='flat')
        cbar = plt.colorbar(tcf_init, ax=ax, pad=0.02)
        cbar.set_label('Flow Speed (m/s)', fontsize=11)
        
        def animate(frame_num):
            ax.clear()
            idx = frame_indices[frame_num]
            
            tcf = ax.tripcolor(triang, speed[idx], 
                              vmin=0, vmax=speed_vmax,
                              cmap='plasma', shading='flat')
            
            sim_time_hours = self.time[idx] / 3600
            sim_time_days = sim_time_hours / 24
            max_speed = speed[idx].max()
            
            ax.set_title(f'Flow Speed\n'
                        f'Time: {sim_time_hours:.1f}h ({sim_time_days:.2f} days) | '
                        f'Max Speed: {max_speed:.3f}m/s',
                        fontsize=13, fontweight='bold')
            
            ax.set_xlabel('Easting (m)', fontsize=11)
            ax.set_ylabel('Northing (m)', fontsize=11)
            ax.set_aspect('equal')
            ax.ticklabel_format(style='plain', useOffset=False)
            
            return tcf,
        
        print(f"  Rendering frames...")
        anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                           interval=1000/fps, blit=False, repeat=True)
        
        output_path = os.path.join(self.animation_dir, output_gif)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✓ Animation saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def create_static_snapshot(self, timestep_hours,
                              output_file='snapshot.png',
                              figsize=(16, 6), dpi=150):
        """
        Create static snapshot at specific time.
        """
        print(f"\n{'='*70}")
        print(f"CREATING STATIC SNAPSHOT at t={timestep_hours:.1f}h")
        print(f"{'='*70}")
        
        # Find closest timestep
        timestep_sec = timestep_hours * 3600
        idx = np.argmin(np.abs(self.time - timestep_sec))
        actual_time = self.time[idx] / 3600
        
        print(f"  Using timestep {idx} (t={actual_time:.2f}h)")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        # Left: Depth
        depth_vmax = np.percentile(self.depth[idx][self.depth[idx] > 0], 99) if (self.depth[idx] > 0).any() else 1.0
        tcf1 = ax1.tripcolor(triang, self.depth[idx], 
                            vmin=0, vmax=depth_vmax,
                            cmap='Blues', shading='flat')
        
        ax1.set_title(f'Water Depth at t={actual_time:.1f}h',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Easting (m)', fontsize=10)
        ax1.set_ylabel('Northing (m)', fontsize=10)
        ax1.set_aspect('equal')
        ax1.ticklabel_format(style='plain', useOffset=False)
        cbar1 = plt.colorbar(tcf1, ax=ax1, pad=0.02)
        cbar1.set_label('Depth (m)', fontsize=10)
        
        # Right: Speed
        depth_safe = np.maximum(self.depth[idx], 0.001)
        u = self.xmomentum[idx] / depth_safe
        v = self.ymomentum[idx] / depth_safe
        speed = np.hypot(u, v)
        speed[self.depth[idx] < 0.001] = 0
        
        speed_vmax = np.percentile(speed[speed > 0], 99) if (speed > 0).any() else 1.0
        tcf2 = ax2.tripcolor(triang, speed, 
                            vmin=0, vmax=speed_vmax,
                            cmap='plasma', shading='flat')
        
        ax2.set_title(f'Flow Speed at t={actual_time:.1f}h',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Easting (m)', fontsize=10)
        ax2.set_ylabel('Northing (m)', fontsize=10)
        ax2.set_aspect('equal')
        ax2.ticklabel_format(style='plain', useOffset=False)
        cbar2 = plt.colorbar(tcf2, ax=ax2, pad=0.02)
        cbar2.set_label('Speed (m/s)', fontsize=10)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.animation_dir, output_file)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"✓ Snapshot saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Max depth: {self.depth[idx].max():.3f}m")
        print(f"  Max speed: {speed.max():.3f}m/s")
        
        return output_path


def main():
    """
    Main function to create animations.
    """
    print("\n" + "="*70)
    print("SEARCHING FOR SIMULATION OUTPUT FILES")
    print("="*70)
    
    # Default output directory (adjust if needed)
    output_dirs = [
        # 'rainfall_runoff_only',  # INFILTERED.py output with checkpoints
        'rainfall_infiltration_runoff',
        '.'
    ]
    
    # Find output directory
    found_dir = None
    for dir_name in output_dirs:
        if os.path.exists(dir_name):
            found_dir = dir_name
            print(f"\n✓ Found output directory: {dir_name}/")
            break
    
    if not found_dir:
        print(f"\n✗ No output directory found. Tried: {output_dirs}")
        print("  Please specify the output directory manually.")
        return
    
    # Strategy 1: Check for checkpoint files FIRST (prioritize loading all checkpoints)
    checkpoint_files = glob.glob(os.path.join(found_dir, 'checkpoint_*.pkl'))
    
    if checkpoint_files and len(checkpoint_files) > 1:
        print(f"\n✓ Found {len(checkpoint_files)} checkpoint files - will load ALL for complete animation")
        print("\n" + "="*70)
        print("CREATING ANIMATIONS FROM ALL CHECKPOINTS")
        print("="*70)
        
        # Use CheckpointAnimator with enhanced functionality
        animator = CheckpointAnimator(found_dir)
        
        # Create summary plot
        animator.create_checkpoint_summary_plot()
        
        # Create spatial animation from all checkpoints
        animator.create_spatial_animation_from_checkpoints(
            frames_per_checkpoint=10,  # Interpolate 10 frames between each checkpoint
            fps=10,  # 10 frames per second
            dpi=100  # Resolution
        )
        
        print("\n✅ Checkpoint-based animations complete!")
        print(f"   Check {found_dir}/animations/ for results")
        return
    
    # Strategy 2: Look for SWW file (fallback)
    sww_files = list(Path(found_dir).glob('*.sww'))
    
    if sww_files:
        print(f"\n✓ Found SWW file: {sww_files[0]}")
        print("\n" + "="*70)
        print("CREATING ANIMATIONS FROM SWW FILE")
        print("="*70)
        
        animator = SWWAnimator(str(sww_files[0]), output_dir=found_dir)
        
        # Create animations
        animator.create_depth_animation(
            output_gif='water_depth_animation.gif',
            frames=50,
            fps=5
        )
        
        # animator.create_speed_animation(
        #     output_gif='flow_speed_animation.gif',
        #     frames=50,
        #     fps=5
        # )
        
        # Create snapshots at key times
        max_time = animator.time[-1] / 3600
        snapshot_times = [max_time * 0.25, max_time * 0.5, max_time * 0.75]
        
        for i, t in enumerate(snapshot_times):
            animator.create_static_snapshot(
                timestep_hours=t,
                output_file=f'snapshot_{i+1}_t{t:.0f}h.png'
            )
    
    else:
        print(f"\n⚠ No SWW file found (domain.set_store(False) was used)")
        print("  Falling back to checkpoint-based visualization...")
        
        # Strategy 2: Use checkpoint files
        checkpoint_files = glob.glob(os.path.join(found_dir, 'checkpoint_*.pkl'))
        
        if checkpoint_files:
            print(f"\n✓ Found {len(checkpoint_files)} checkpoint files")
            print("\n" + "="*70)
            print("CREATING SUMMARY FROM CHECKPOINTS")
            print("="*70)
            
            animator = CheckpointAnimator(found_dir)
            animator.create_checkpoint_summary_plot()
            
            print("\n" + "="*70)
            print("NOTE: Spatial animations require SWW file")
            print("="*70)
            print("To create spatial animations, re-run your simulation with:")
            print("  domain.set_store(True)")
            print("  domain.set_store_vertices_uniquely(False)")
            print("\nFor now, I've created a time-series summary plot.")
        
        else:
            print(f"\n✗ No checkpoint files found either!")
            print(f"  Searched in: {found_dir}/")
            print("\nPlease ensure your simulation has completed and saved output files.")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Check the '{found_dir}/animations/' directory for results")


if __name__ == '__main__':
    main()
