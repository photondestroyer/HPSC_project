#!/usr/bin/env python3
"""
Advanced ANUGA Flow Visualization with Directional Arrows
==========================================================

Creates animations and GIFs from ANUGA simulation results with:
- Water depth/stage visualization
- Flow direction arrows (velocity vectors)
- Flow speed magnitude
- Customizable styling and output

Designed for results from trial_gpu_accelerated.ipynb

Author: AI Assistant
Date: 2025-11-23
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

# Try to import ANUGA utilities
try:
    from anuga.file.netcdf import NetCDFFile
    from anuga import plot_utils as util
    ANUGA_AVAILABLE = True
except ImportError:
    print("Warning: ANUGA plot utilities not available. Using basic NetCDF.")
    from scipy.io import netcdf_file as NetCDFFile
    ANUGA_AVAILABLE = False


class FlowVisualizer:
    """
    Advanced flow visualization with directional arrows.
    """
    
    def __init__(self, sww_file, output_dir='outputs/animations'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        sww_file : str
            Path to ANUGA .sww simulation file
        output_dir : str
            Directory for output animations
        """
        self.sww_file = sww_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading simulation data from: {sww_file}")
        self._load_data()
        print(f"  ✓ Loaded {len(self.time)} timesteps")
        print(f"  ✓ Simulation domain: {self.n_triangles:,} triangles, {self.n_vertices:,} vertices")
        print(f"  ✓ Time range: 0 to {self.time[-1]/3600:.1f} hours")
    
    def _load_data(self):
        """Load data from SWW file."""
        fid = NetCDFFile(self.sww_file, 'r')
        
        # Mesh data
        self.x = np.array(fid.variables['x'][:])
        self.y = np.array(fid.variables['y'][:])
        self.volumes = np.array(fid.variables['volumes'][:])
        
        # Time data
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
        
        # Compute centroid coordinates
        self._compute_centroids()
    
    def _compute_centroids(self):
        """Compute triangle centroids."""
        x_verts = self.x[self.volumes]
        y_verts = self.y[self.volumes]
        
        self.xc = np.mean(x_verts, axis=1)
        self.yc = np.mean(y_verts, axis=1)
    
    def _compute_centroid_quantities(self, timestep):
        """
        Compute quantities at centroids for a given timestep.
        
        Returns:
        --------
        dict with keys: depth, stage, xvel, yvel, speed
        """
        # Get vertex values
        stage_verts = self.stage[timestep, self.volumes]
        elev_verts = self.elevation[self.volumes] if len(self.elevation.shape) == 1 else self.elevation[timestep, self.volumes]
        xmom_verts = self.xmomentum[timestep, self.volumes]
        ymom_verts = self.ymomentum[timestep, self.volumes]
        
        # Average to centroids
        stage_c = np.mean(stage_verts, axis=1)
        elev_c = np.mean(elev_verts, axis=1)
        xmom_c = np.mean(xmom_verts, axis=1)
        ymom_c = np.mean(ymom_verts, axis=1)
        
        # Compute depth
        depth_c = np.maximum(stage_c - elev_c, 0)
        
        # Compute velocity (momentum / depth)
        # Avoid division by zero
        depth_safe = np.maximum(depth_c, 0.001)
        xvel_c = xmom_c / depth_safe
        yvel_c = ymom_c / depth_safe
        
        # Mask velocities where depth is too small
        mask = depth_c < 0.001
        xvel_c[mask] = 0
        yvel_c[mask] = 0
        
        # Compute speed
        speed_c = np.sqrt(xvel_c**2 + yvel_c**2)
        
        return {
            'depth': depth_c,
            'stage': stage_c,
            'xvel': xvel_c,
            'yvel': yvel_c,
            'speed': speed_c
        }
    
    def create_depth_with_arrows_animation(self, 
                                          output_gif='water_depth_with_flow.gif',
                                          frames=50, 
                                          fps=5,
                                          arrow_density=0.05,
                                          arrow_scale=50,
                                          min_speed_threshold=0.01,
                                          depth_vmax=None,
                                          figsize=(14, 10),
                                          dpi=100):
        """
        Create animation showing water depth with flow direction arrows.
        
        Parameters:
        -----------
        output_gif : str
            Output filename
        frames : int
            Number of frames to include
        fps : int
            Frames per second
        arrow_density : float
            Fraction of triangles to show arrows (0-1). Lower = fewer arrows
        arrow_scale : float
            Arrow size scaling factor (higher = shorter arrows)
        min_speed_threshold : float
            Minimum flow speed (m/s) to show arrows
        depth_vmax : float
            Maximum depth for colorbar (None = auto)
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Resolution
        """
        print(f"\nCreating depth + flow direction animation...")
        print(f"  Frames: {frames} at {fps} fps")
        print(f"  Arrow density: {arrow_density*100:.1f}%")
        
        # Select timesteps
        step = max(1, self.n_timesteps // frames)
        frame_indices = range(0, self.n_timesteps, step)
        
        # Determine color scale
        if depth_vmax is None:
            all_depths = []
            for idx in frame_indices[:min(10, len(frame_indices))]:  # Sample first 10 frames
                q = self._compute_centroid_quantities(idx)
                all_depths.extend(q['depth'][q['depth'] > 0])
            depth_vmax = np.percentile(all_depths, 99) if all_depths else 1.0
        
        print(f"  Depth color scale: 0 to {depth_vmax:.3f} m")
        
        # Select subset of triangles for arrows (for performance)
        n_arrows = int(self.n_triangles * arrow_density)
        arrow_indices = np.random.choice(self.n_triangles, n_arrows, replace=False)
        print(f"  Drawing {n_arrows:,} arrows")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create triangulation
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        def animate(frame_num):
            ax.clear()
            idx = frame_indices[frame_num]
            
            # Compute quantities
            q = self._compute_centroid_quantities(idx)
            
            # Plot depth as filled contours
            tcf = ax.tripcolor(triang, q['depth'], vmin=0, vmax=depth_vmax, cmap='viridis', shading='flat')
            
            # Filter arrows by speed threshold
            arrow_mask = q['speed'][arrow_indices] > min_speed_threshold
            arrow_idx_filtered = arrow_indices[arrow_mask]
            
            if len(arrow_idx_filtered) > 0:
                # Plot velocity arrows
                ax.quiver(self.xc[arrow_idx_filtered], 
                         self.yc[arrow_idx_filtered],
                         q['xvel'][arrow_idx_filtered],
                         q['yvel'][arrow_idx_filtered],
                         color='white',
                         scale=arrow_scale,
                         scale_units='xy',
                         width=0.003,
                         alpha=0.7,
                         headwidth=4,
                         headlength=5)
            
            # Add time info
            sim_time_hours = self.time[idx] / 3600
            sim_time_days = sim_time_hours / 24
            ax.set_title(f'Water Depth & Flow Direction\n'
                        f'Time: {sim_time_hours:.1f}h ({sim_time_days:.2f} days) | '
                        f'Max Depth: {q["depth"].max():.3f}m | '
                        f'Max Speed: {q["speed"].max():.3f}m/s',
                        fontsize=13, fontweight='bold')
            
            ax.set_xlabel('Easting (m)', fontsize=11)
            ax.set_ylabel('Northing (m)', fontsize=11)
            ax.set_aspect('equal')
            ax.ticklabel_format(style='plain', useOffset=False)
            
            # Add colorbar on first frame
            if frame_num == 0:
                cbar = plt.colorbar(tcf, ax=ax, pad=0.02)
                cbar.set_label('Water Depth (m)', fontsize=11)
            
            return tcf,
        
        # Create animation
        print(f"  Rendering frames...")
        anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                           interval=1000/fps, blit=False, repeat=True)
        
        # Save
        output_path = os.path.join(self.output_dir, output_gif)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✓ Animation saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def create_speed_magnitude_animation(self,
                                        output_gif='flow_speed.gif',
                                        frames=50,
                                        fps=5,
                                        speed_vmax=None,
                                        figsize=(14, 10),
                                        dpi=100):
        """
        Create animation showing flow speed magnitude.
        """
        print(f"\nCreating flow speed animation...")
        
        # Select timesteps
        step = max(1, self.n_timesteps // frames)
        frame_indices = range(0, self.n_timesteps, step)
        
        # Determine color scale
        if speed_vmax is None:
            all_speeds = []
            for idx in frame_indices[:min(10, len(frame_indices))]:
                q = self._compute_centroid_quantities(idx)
                all_speeds.extend(q['speed'][q['speed'] > 0])
            speed_vmax = np.percentile(all_speeds, 99) if all_speeds else 1.0
        
        print(f"  Speed color scale: 0 to {speed_vmax:.3f} m/s")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        def animate(frame_num):
            ax.clear()
            idx = frame_indices[frame_num]
            
            q = self._compute_centroid_quantities(idx)
            
            # Plot speed
            tcf = ax.tripcolor(triang, q['speed'], vmin=0, vmax=speed_vmax, cmap='plasma', shading='flat')
            
            sim_time_hours = self.time[idx] / 3600
            sim_time_days = sim_time_hours / 24
            ax.set_title(f'Flow Speed Magnitude\n'
                        f'Time: {sim_time_hours:.1f}h ({sim_time_days:.2f} days) | '
                        f'Max Speed: {q["speed"].max():.3f}m/s',
                        fontsize=13, fontweight='bold')
            
            ax.set_xlabel('Easting (m)', fontsize=11)
            ax.set_ylabel('Northing (m)', fontsize=11)
            ax.set_aspect('equal')
            ax.ticklabel_format(style='plain', useOffset=False)
            
            if frame_num == 0:
                cbar = plt.colorbar(tcf, ax=ax, pad=0.02)
                cbar.set_label('Flow Speed (m/s)', fontsize=11)
            
            return tcf,
        
        print(f"  Rendering frames...")
        anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                           interval=1000/fps, blit=False, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_gif)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✓ Animation saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def create_static_snapshot(self, 
                              timestep_hours,
                              output_file='flow_snapshot.png',
                              show_arrows=True,
                              arrow_density=0.1,
                              arrow_scale=50,
                              min_speed_threshold=0.01,
                              figsize=(16, 12),
                              dpi=150):
        """
        Create a high-resolution static snapshot at a specific time.
        
        Parameters:
        -----------
        timestep_hours : float
            Time in hours to visualize
        """
        print(f"\nCreating static snapshot at t={timestep_hours:.1f}h...")
        
        # Find closest timestep
        timestep_sec = timestep_hours * 3600
        idx = np.argmin(np.abs(self.time - timestep_sec))
        actual_time = self.time[idx] / 3600
        
        print(f"  Using timestep {idx} (t={actual_time:.2f}h)")
        
        # Compute quantities
        q = self._compute_centroid_quantities(idx)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        triang = tri.Triangulation(self.x, self.y, self.volumes)
        
        # Left: Depth with arrows
        depth_vmax = np.percentile(q['depth'][q['depth'] > 0], 99)
        tcf1 = ax1.tripcolor(triang, q['depth'], vmin=0, vmax=depth_vmax,
                           cmap='viridis', shading='flat')
        
        if show_arrows:
            n_arrows = int(self.n_triangles * arrow_density)
            arrow_indices = np.random.choice(self.n_triangles, n_arrows, replace=False)
            arrow_mask = q['speed'][arrow_indices] > min_speed_threshold
            arrow_idx_filtered = arrow_indices[arrow_mask]
            
            if len(arrow_idx_filtered) > 0:
                ax1.quiver(self.xc[arrow_idx_filtered], 
                         self.yc[arrow_idx_filtered],
                         q['xvel'][arrow_idx_filtered],
                         q['yvel'][arrow_idx_filtered],
                         color='white',
                         scale=arrow_scale,
                         scale_units='xy',
                         width=0.003,
                         alpha=0.7,
                         headwidth=4,
                         headlength=5)
        
        ax1.set_title(f'Water Depth & Flow Direction at t={actual_time:.1f}h',
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Easting (m)', fontsize=11)
        ax1.set_ylabel('Northing (m)', fontsize=11)
        ax1.set_aspect('equal')
        ax1.ticklabel_format(style='plain', useOffset=False)
        cbar1 = plt.colorbar(tcf1, ax=ax1, pad=0.02)
        cbar1.set_label('Water Depth (m)', fontsize=11)
        
        # Right: Speed magnitude
        speed_vmax = np.percentile(q['speed'][q['speed'] > 0], 99) if (q['speed'] > 0).any() else 1.0
        tcf2 = ax2.tripcolor(triang, q['speed'], vmin=0, vmax=speed_vmax,
                           cmap='plasma', shading='flat')
        
        ax2.set_title(f'Flow Speed Magnitude at t={actual_time:.1f}h',
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('Easting (m)', fontsize=11)
        ax2.set_ylabel('Northing (m)', fontsize=11)
        ax2.set_aspect('equal')
        ax2.ticklabel_format(style='plain', useOffset=False)
        cbar2 = plt.colorbar(tcf2, ax=ax2, pad=0.02)
        cbar2.set_label('Flow Speed (m/s)', fontsize=11)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"✓ Snapshot saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Max depth: {q['depth'].max():.3f}m")
        print(f"  Max speed: {q['speed'].max():.3f}m/s")
        
        return output_path


def main():
    """
    Main function with example usage.
    """
    print("="*70)
    print("ANUGA FLOW VISUALIZATION WITH DIRECTIONAL ARROWS")
    print("="*70)
    
    # Find SWW file
    sww_files = list(Path('outputs').glob('*.sww'))
    
    if not sww_files:
        print("\n✗ No .sww files found in outputs/")
        print("  Make sure you've run the simulation with domain.set_store(True)")
        return
    
    sww_file = str(sww_files[0])
    print(f"\nUsing: {sww_file}\n")
    
    # Create visualizer
    viz = FlowVisualizer(sww_file)
    
    # Create animations
    print("\n" + "="*70)
    print("CREATING ANIMATIONS")
    print("="*70)
    
    # 1. Depth with flow arrows
    viz.create_depth_with_arrows_animation(
        output_gif='depth_with_flow_arrows.gif',
        frames=50,
        fps=5,
        arrow_density=0.05,  # 5% of triangles
        arrow_scale=50,
        min_speed_threshold=0.01
    )
    
    # 2. Flow speed magnitude
    viz.create_speed_magnitude_animation(
        output_gif='flow_speed_magnitude.gif',
        frames=50,
        fps=5
    )
    
    # 3. High-resolution snapshots at key times
    print("\n" + "="*70)
    print("CREATING SNAPSHOTS")
    print("="*70)
    
    # Snapshot during peak rainfall
    viz.create_static_snapshot(
        timestep_hours=5.0,  # During rainfall
        output_file='snapshot_during_rain.png',
        show_arrows=True,
        arrow_density=0.1
    )
    
    # Snapshot after rainfall
    viz.create_static_snapshot(
        timestep_hours=24.0,  # After 1 day
        output_file='snapshot_day1.png',
        show_arrows=True,
        arrow_density=0.1
    )
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nCheck the 'outputs/animations/' directory for results")


if __name__ == '__main__':
    main()
