#!/usr/bin/env python
"""
Check if ANUGA has GPU/CUDA support compiled in
"""

import sys

print("="*70)
print("ANUGA GPU SUPPORT CHECK")
print("="*70)

# Check 1: ANUGA import
try:
    import anuga
    print(f"\n✅ ANUGA imported successfully")
    print(f"   Version: {anuga.__version__ if hasattr(anuga, '__version__') else 'Unknown'}")
    print(f"   Location: {anuga.__file__}")
except ImportError as e:
    print(f"\n❌ Cannot import ANUGA: {e}")
    sys.exit(1)

# Check 2: CuPy availability
try:
    import cupy as cp
    print(f"\n✅ CuPy available")
    print(f"   Version: {cp.__version__}")
    
    # Test GPU
    device = cp.cuda.Device()
    print(f"   GPU Device: {device.id}")
    mem = device.mem_info
    print(f"   GPU Memory: {mem[0]/1e9:.2f} GB free / {mem[1]/1e9:.2f} GB total")
except ImportError:
    print(f"\n❌ CuPy not available - GPU acceleration impossible")
    sys.exit(1)

# Check 3: Domain has GPU methods
try:
    domain = anuga.rectangular_cross_domain(10, 10, len1=100, len2=100)
    
    print(f"\n✅ Test domain created")
    print(f"   Triangles: {domain.get_number_of_triangles()}")
    
    # Check for multiprocessor mode method
    if hasattr(domain, 'set_multiprocessor_mode'):
        print(f"   ✅ Has set_multiprocessor_mode() method")
    else:
        print(f"   ❌ Missing set_multiprocessor_mode() method")
        print(f"      ANUGA not compiled with GPU support!")
        sys.exit(1)
    
    # Try to enable GPU mode
    print(f"\n🔧 Attempting to enable GPU mode...")
    domain.set_multiprocessor_mode(2)
    
    mode = domain.get_multiprocessor_mode()
    print(f"   Multiprocessor mode set to: {mode}")
    
    if mode == 2:
        print(f"   ✅ GPU mode activated (mode=2)")
    else:
        print(f"   ⚠️  GPU mode NOT activated (mode={mode})")
    
    # Check GPU interface
    print(f"\n🔍 Checking GPU interface...")
    if hasattr(domain, 'gpu_interface'):
        gpu_int = domain.gpu_interface
        print(f"   gpu_interface exists: {gpu_int is not None}")
        
        if gpu_int is not None:
            print(f"   ✅ GPU interface created!")
            print(f"      Type: {type(gpu_int).__name__}")
            print(f"      Module: {gpu_int.__class__.__module__}")
            
            # Check for critical kernels
            kernels = [
                'protect_against_infinitesimal_and_negative_heights_kernel',
                'compute_fluxes_ext_short_kernel',
                'saxpy_centroid_values_kernel',
                'update_conserved_quantities_kernel'
            ]
            
            print(f"\n   GPU Kernels:")
            all_present = True
            for kernel in kernels:
                has_it = hasattr(gpu_int, kernel)
                status = "✅" if has_it else "❌"
                print(f"      {status} {kernel}")
                if not has_it:
                    all_present = False
            
            if all_present:
                print(f"\n   ✅ All required GPU kernels present!")
            else:
                print(f"\n   ❌ Some GPU kernels missing - GPU mode may not work")
        else:
            print(f"   ❌ GPU interface is None")
            print(f"      ANUGA GPU mode failed to initialize")
            print(f"\n💡 Possible reasons:")
            print(f"   1. ANUGA not compiled with CUDA support")
            print(f"   2. CuPy/CUDA version mismatch")
            print(f"   3. GPU kernels not found")
    else:
        print(f"   ❌ Domain has no 'gpu_interface' attribute")
        print(f"      ANUGA not compiled with GPU support")
    
except Exception as e:
    print(f"\n❌ Error during GPU check: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Look for CUDA-related files
print(f"\n🔍 Checking for CUDA libraries in ANUGA installation...")
import os
from pathlib import Path

anuga_dir = Path(anuga.__file__).parent
cuda_files = list(anuga_dir.rglob("*.cu")) + list(anuga_dir.rglob("*cuda*")) + list(anuga_dir.rglob("*gpu*"))

if cuda_files:
    print(f"   ✅ Found {len(cuda_files)} CUDA/GPU related files:")
    for f in cuda_files[:10]:  # Show first 10
        print(f"      - {f.name}")
    if len(cuda_files) > 10:
        print(f"      ... and {len(cuda_files)-10} more")
else:
    print(f"   ❌ No CUDA/GPU files found in ANUGA installation")
    print(f"      Your ANUGA was likely installed without GPU support")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

# Final verdict
gpu_ready = (
    'cupy' in sys.modules and
    hasattr(domain, 'set_multiprocessor_mode') and
    domain.gpu_interface is not None
)

if gpu_ready:
    print(f"✅ ANUGA GPU SUPPORT: FULLY FUNCTIONAL")
    print(f"   You can run simulations with GPU acceleration")
    print(f"\n   In your script:")
    print(f"      domain.set_multiprocessor_mode(2)")
    print(f"      # GPU will be used automatically")
else:
    print(f"❌ ANUGA GPU SUPPORT: NOT AVAILABLE")
    print(f"   Your ANUGA installation does not support GPU")
    print(f"\n   To fix:")
    print(f"   1. Uninstall current ANUGA:")
    print(f"      pip uninstall anuga")
    print(f"   2. Install ANUGA with CUDA support:")
    print(f"      pip install anuga[cuda]")
    print(f"   OR compile from source with CUDA enabled")

print(f"{'='*70}\n")
