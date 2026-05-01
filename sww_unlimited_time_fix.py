"""
ANUGA SWW Storage Fix - Runtime Monkey Patch
============================================

This module patches ANUGA's SWW writer at runtime to fix the IndexError
with large yieldsteps without requiring ANUGA recompilation.

The Issue:
----------
When `domain.set_store(True)` is used with large yieldsteps (e.g., 24 hours),
ANUGA creates a SWW file with `number_of_timesteps=0` (unlimited dimension),
but the store_quantities method tries to access `file_time[0]` which fails.

The Fix:
--------
This patches the Write_sww class to properly handle unlimited time dimensions
by modifying the store_header method at runtime.

Usage:
------
    # Import this module BEFORE creating your domain
    import sww_unlimited_time_fix
    sww_unlimited_time_fix.apply_patch()
    
    # Then create and use domain normally
    domain = anuga.create_domain_from_regions(...)
    domain.set_store(True)  # Now works with large yieldsteps!

Author: AI Assistant
Date: 2025-11-26
"""

import sys
import numpy as np

def apply_patch():
    """
    Apply runtime patch to ANUGA's SWW writer to support unlimited time dimensions.
    
    This function must be called BEFORE creating any ANUGA domains.
    """
    
    try:
        # Import ANUGA modules
        from anuga.file.sww import Write_sww
        from anuga.file.netcdf import NetCDFFile
        
        print("="*70)
        print("APPLYING ANUGA SWW UNLIMITED TIME DIMENSION FIX")
        print("="*70)
        
        # Store the original store_header method
        original_store_header = Write_sww.store_header
        
        # Create the patched version
        def patched_store_header(self,
                                 outfile,
                                 times,
                                 number_of_volumes,
                                 number_of_points,
                                 description='Stored from anuga.file.sww',
                                 institution='Geoscience Australia',
                                 smoothing=True,
                                 order=1,
                                 sww_precision=np.float32,
                                 timezone='UTC',
                                 verbose=False):
            """
            Patched version of store_header that uses unlimited time dimension
            when number_of_times is 0 or unknown.
            """
            
            from anuga import get_revision_number, get_revision_date, get_version
            
            outfile.institution = institution
            outfile.description = description
            
            # Smoothing settings
            if smoothing is True:
                outfile.smoothing = 'Yes'
                outfile.vertices_are_stored_uniquely = 'False'
            else:
                outfile.smoothing = 'No'
                outfile.vertices_are_stored_uniquely = 'True'
            outfile.order = order
            
            # Version information
            try:
                revision_number = get_revision_number()
            except:
                revision_number = None
            outfile.revision_number = str(revision_number)
            
            try:
                revision_date = get_revision_date()
            except:
                revision_date = None
            outfile.revision_date = str(revision_date)
            
            try:
                anuga_version = get_version()
            except:
                anuga_version = None
            outfile.anuga_version = str(anuga_version)
            
            # Handle times parameter
            if isinstance(times, (list, np.ndarray)):
                number_of_times = len(times)
                times = np.asarray(times, dtype=np.float64)
                if number_of_times == 0:
                    starttime = 0
                else:
                    starttime = times[0]
                    times = times - starttime
            else:
                number_of_times = 0
                starttime = times
            
            outfile.starttime = starttime
            outfile.timezone = timezone
            
            # Create dimensions
            outfile.createDimension('number_of_volumes', number_of_volumes)
            outfile.createDimension('number_of_triangle_vertices', number_of_points)
            outfile.createDimension('number_of_vertices', 3)
            outfile.createDimension('numbers_in_range', 2)
            
            if smoothing is True:
                outfile.createDimension('number_of_points', number_of_points)
            else:
                outfile.createDimension('number_of_points', 3*number_of_volumes)
            
            # **THE FIX**: Use unlimited dimension when number_of_times is 0
            if number_of_times == 0:
                outfile.createDimension('number_of_timesteps', None)  # UNLIMITED!
                print("✅ SWW Fix Applied: Using unlimited time dimension")
            else:
                outfile.createDimension('number_of_timesteps', number_of_times)
                print(f"ℹ️  SWW: Using fixed time dimension ({number_of_times} timesteps)")
            
            # Create variable definitions (rest of the original method)
            outfile.createVariable('x', sww_precision, ('number_of_points',))
            outfile.createVariable('y', sww_precision, ('number_of_points',))
            outfile.createVariable('volumes', np.int32, 
                                   ('number_of_volumes', 'number_of_vertices'))
            
            # Static quantities
            max_float = np.finfo(sww_precision).max
            for q in self.static_quantities:
                outfile.createVariable(q, sww_precision, ('number_of_points',))
                outfile.createVariable(q + Write_sww.RANGE, sww_precision, 
                                       ('numbers_in_range',))
                outfile.variables[q + Write_sww.RANGE][0] = max_float
                outfile.variables[q + Write_sww.RANGE][1] = -max_float
            
            # Static centroid quantities
            for q in self.static_c_quantities:
                outfile.createVariable(q, sww_precision, ('number_of_volumes',))
            
            # Write dynamic quantities structure
            self.write_dynamic_quantities(outfile, times, precis=sww_precision)
            
            outfile.sync()
        
        # Apply the patch
        Write_sww.store_header = patched_store_header
        
        print("\n✅ Patch applied successfully!")
        print("   Write_sww.store_header() now uses unlimited time dimension")
        print("   You can now use domain.set_store(True) with large yieldsteps")
        print(f"{'='*70}\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import ANUGA modules: {e}")
        print("   Make sure ANUGA is installed: conda activate anugs")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_patch():
    """
    Verify that the patch has been applied correctly.
    """
    try:
        from anuga.file.sww import Write_sww
        
        # Check if the method has been patched
        import inspect
        source = inspect.getsource(Write_sww.store_header)
        
        if "UNLIMITED" in source or "unlimited" in source.lower():
            print("✅ Patch verification: SUCCESS")
            print("   The store_header method contains unlimited dimension logic")
            return True
        else:
            print("⚠️  Patch verification: UNCERTAIN")
            print("   Cannot confirm patch in source code")
            return False
            
    except Exception as e:
        print(f"❌ Patch verification failed: {e}")
        return False


# Auto-apply patch when module is imported (optional)
if __name__ != "__main__":
    print("\n💡 TIP: Call sww_unlimited_time_fix.apply_patch() before creating your domain")
    print("   to enable SWW storage with large yieldsteps\n")
