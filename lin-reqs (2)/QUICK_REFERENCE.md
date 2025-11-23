# Quick Reference Card - ANUGA GPU Simulations

## GPU Setup (One-Time)
```bash
# Check CUDA version
nvidia-smi

# Install CuPy (choose based on your CUDA version)
pip install cupy-cuda11x   # For CUDA 11.x
pip install cupy-cuda12x   # For CUDA 12.x
```

## Run Notebook Cells in Order

### Cell 1: Import & Check GPU
```python
import anuga, numpy, matplotlib
# Shows: ✓ CuPy found - GPU acceleration ENABLED
```

### Cell 2: Check System Info
```python
# Shows GPU details, memory, CUDA version
```

### Cell 3: Setup Configuration Classes
```python
# Defines helper functions and SimulationConfig class
```

### Cell 4: Adjust Configuration ⚙️
```python
# MODIFY THIS CELL FOR YOUR SCENARIO

# Quick test (15 min)
config.total_simulation_hours = 0.25

# Standard (24 hours)  
config.total_simulation_hours = 24.0

# Long run (1 week)
config.total_simulation_hours = 168.0

# Very long (1 month)
config.total_simulation_hours = 720.0
```

### Cell 5: Create Domain
```python
# Loads DEM, creates mesh, sets up boundaries
# Enables GPU automatically if available
```

### Cell 6: Run Simulation 🚀
```python
# This is the LONG-RUNNING cell
# Watch progress in output
# Saves checkpoints automatically
```

### Cell 7: Plot Results
```python
# Creates time series plots
```

### Cell 8: Create Animation
```python
# Generates animation from .sww file
```

### Cell 9: Resume (if needed)
```python
# Load last checkpoint to continue interrupted simulation
```

## Key Configuration Parameters

| Parameter | Fast | Balanced | Detailed |
|-----------|------|----------|----------|
| `maximum_triangle_area` | 100000 | 50000 | 25000 |
| `output_interval_minutes` | 60 | 30 | 10 |
| `checkpoint_interval_hours` | 24 | 6 | 2 |

## Speed Expectations

### GPU Performance (vs CPU baseline)
- Small domain (<10k triangles): ~1x (no benefit)
- Medium (50k triangles): ~4x faster
- Large (250k triangles): ~7x faster  
- Very large (1M+ triangles): ~10x faster

### Real-Time Estimates (with GPU)
- 1 hour sim ≈ 10-20 min real time
- 24 hour sim ≈ 4-8 hours real time
- 1 week sim ≈ 1-2 days real time
- 1 month sim ≈ 4-7 days real time

*(Varies by mesh resolution)*

## Monitoring During Simulation

### Progress Output
```
[RAIN ] t=00:30 (0.50h) | Max:0.015m | Mean:0.003m | Vol:2.45M m³ | dt:2.15s | Speed:45.2x realtime
```

- **[RAIN/DRAIN]**: Rainfall active or draining phase
- **Max/Mean**: Water depth in meters
- **Vol**: Total water volume (million m³)
- **dt**: Adaptive timestep (seconds)
- **Speed**: Simulation speed (higher = faster)

### GPU Monitor (separate terminal)
```bash
nvidia-smi -l 1  # Update every 1 second
```
Watch for:
- GPU utilization: Should be 70-100%
- Memory usage: Should be stable
- Temperature: <85°C is safe

## Common Issues & Fixes

### ❌ "No module named 'cupy'"
**Fix:** `pip install cupy-cuda11x` (or cuda12x)

### ❌ GPU not being used (slow simulation)
**Check:**
1. Cell 1 shows "GPU acceleration ENABLED"?
2. Cell 5 shows "multiprocessor_mode=2"?
3. `nvidia-smi` shows Python process?

**Fix:** Restart kernel, verify CuPy installed

### ❌ Out of GPU memory
**Fix:** Reduce mesh resolution
```python
config.maximum_triangle_area = 100000  # Larger = less memory
```

### ❌ Simulation too slow
**Speed up:**
1. Increase triangle area (coarser mesh)
2. Increase output interval
3. Use GPU (if not already)
4. Close other GPU applications

### ❌ Checkpoint files not saving
**Check:**
```python
config.save_checkpoints = True  # Must be True
os.path.exists('outputs')  # Directory must exist
```

## Output Files Location

```
E:\4th year\ANUGA-ing\outputs\
├── rainfall_simulation.sww       # Main simulation output
├── timeseries_data.npz           # Statistics over time
├── timeseries_plots.png          # Summary plots
└── checkpoint_XXX.pkl            # Recovery checkpoints
```

## Resume Interrupted Simulation

1. Run all cells up to Cell 5 (domain creation)
2. Run Cell 9 (Resume from checkpoint)
3. Modify Cell 6: adjust `finaltime` to continue
4. Run Cell 6 again to continue simulation

## Performance Checklist

✅ **Before starting long simulation:**
- [ ] GPU detected (Cell 1 shows "ENABLED")
- [ ] Configuration set (Cell 4)
- [ ] Test run completed (15 min test)
- [ ] Checkpoints enabled
- [ ] Sufficient disk space (~1GB per day of sim time)
- [ ] Output directory exists

✅ **During simulation:**
- [ ] Progress updates showing regularly
- [ ] Speed >10x realtime (with GPU)
- [ ] Checkpoint files being created
- [ ] GPU utilization >70% (`nvidia-smi`)

✅ **After simulation:**
- [ ] .sww file created
- [ ] Time series plots generated
- [ ] Animation created (optional)
- [ ] Results make physical sense

## Contact & Support

**Documentation:** `SIMULATION_GUIDE.md` (detailed)
**Examples:** `anuga_core-main/examples/cuda/`
**ANUGA Docs:** https://anuga.anu.edu.au/

## Quick Commands

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Test CuPy
python -c "import cupy; print(cupy.cuda.Device().name)"

# Monitor GPU (real-time)
watch -n 1 nvidia-smi

# Disk space
du -sh outputs/

# Kill if stuck
# Ctrl+C in notebook or restart kernel
```

## Memory Estimates

| Mesh Size | GPU Memory | Output Size (per day sim) |
|-----------|------------|---------------------------|
| 10k tri   | 0.5 GB     | 50 MB                     |
| 50k tri   | 2 GB       | 250 MB                    |
| 250k tri  | 6 GB       | 1.2 GB                    |
| 1M tri    | 16+ GB     | 5+ GB                     |

*Your domain: ~50k triangles = 2GB GPU, ~250MB output per day*
