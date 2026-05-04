# ANUGA-ing: GPU-Accelerated Flood & Infiltration Simulation Workflow

https://github.com/user-attachments/assets/a2144ac3-dcad-4405-a082-706699169132

**⚠️ Disclaimer on Reproducibility:**
This exact workflow is **not strictly reproducible** out-of-the-box via this repository because the required raw input data files (e.g., high-resolution DEM GeoTIFFs, large IMERG rainfall CSVs, and spatial boundary files) cannot be hosted on GitHub due to file size limits and licensing restrictions.

## Overview
This repository contains an end-to-end HPC-optimized workflow for running GPU-accelerated 2D overland flow and infiltration simulations using a modified [ANUGA](https://github.com/GeoscienceAustralia/anuga_core) engine. The primary "production" script driving this is `INFILTERED.py`, which efficiently couples dynamic rainfall data and a Green-Ampt infiltration operator with ANUGA's GPU flux solver.

## HPC Optimizations
To achieve high performance on modern CPU/GPU systems and handle realistically large geographic domains, several architectural optimizations have been implemented in `INFILTERED.py`. These focus heavily on **temporal locality**, **spatial locality**, and **minimizing memory allocations**:

1. **Disk-Backed, Time-Major Rainfall Cache:**
   Rainfall rates for every individual triangle centroid are pre-computed into a disk-backed `float32` memmapped array mapped as `(n_timesteps, n_triangles)`. Because the simulation evolves chronologically, this time-major layout guarantees predictable, sequential row access and extremely fast reads heavily assisted by the OS page cache.

2. **Temporal & Spatial Locality in Data Alignment:**
   - **Temporal:** Bypasses $O(n_{timesteps} \cdot n_{records})$ lookups by pre-sorting timestamps and using bounds tracking so rainfall records map cleanly to a contiguous slice of time.
   - **Spatial:** Pre-computes the 4 surrounding spatial grid cell indices for each mesh triangle centroid exactly once, allowing zero-allocation vectorized gathers/scatters instead of dynamic spatial lookups.

3. **Allocation-Free Hot Loops (Operators & Evolve):**
   Inside the tight simulation `evolve()` loop, Python/NumPy heap allocations are strictly minimized to prevent bandwidth bottlenecks and GC pressure:
   - Reuses persistent arrays (`_rate_buffer`, `depth`, `wet_mask`, `infiltrating`) initialized once.
   - Uses in-place operations aggressively (e.g. `np.maximum(out=...)`, `np.subtract(out=...)`, `np.copyto()`).
   - Uses scalar dot products `np.dot(depth, areas)` to calculate total volume rather than temporarily instantiating a massive `depth * areas` full-sized domain array.

4. **Sparse vs Dense Operator Execution:**
   The `GreenAmptInfiltrationOperator` conditionally routes execution. When only a small fraction of the geographic domain is wet (sparse), it isolates and updates only those specific memory indices. It switches to dense full-array operations only when infiltration is widespread, preventing unnecessary memory churn on dry land.

5. **Robust Checkpointing & Recovery:**
   Full state, infiltration progression, and diagnostic variables are aggressively checkpointed to `.pkl` and `.npz` arrays. Since large mesh + long time-scale iterations can easily crash traditional storage arrays (like SWW), this custom scheme allows both immediate restart upon interruption and independent, headless animation generation without relying on massive mesh files.

## Workflow Pipeline
While the original DEM and rainfall data are deliberately omitted, the intended pipeline execution order is:
1. **Install:** Deploy the custom GPU-enabled ANUGA engine (see `INSTALL_MODIFIED_ANUGA.md`).
2. **Preprocess:** Clean, fill, and smooth the DEM (`preprocess_dem.ipynb`) and generate the geographic modeling boundary (`GDALing.py`).
3. **Simulate:** Run `python INFILTERED.py` (which transforms local geographic data, caches rainfall iteratively, and executes the simulation loop).
4. **Animate:** Run `python create_simulation_animation.py` to compile the visual video renders directly from the checkpoint outputs.




