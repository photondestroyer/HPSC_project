#!/usr/bin/env python
"""Test the SWW unlimited time dimension fix"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/intern1/ishan/ANUGA-ing')

# Import and apply the patch
import sww_unlimited_time_fix
success = sww_unlimited_time_fix.apply_patch()

if success:
    print("\n✅ SUCCESS! The patch is working correctly.")
    print("\nYou can now run INFILTERED.py with domain.set_store(True) enabled!")
    print("The IndexError with large yieldsteps should be fixed.\n")
else:
    print("\n❌ FAILED! The patch could not be applied.")
    print("Check the error messages above.\n")
    sys.exit(1)
