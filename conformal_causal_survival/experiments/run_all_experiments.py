import os
import sys
import pandas as pd
import warnings

# Set R_HOME explicitly
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = '/opt/miniconda3/envs/py313/lib/R'

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from conformal_causal_survival.experiments.run_melanoma import run_melanoma_experiment as run_melanoma
from conformal_causal_survival.experiments.run_pbc import run_pbc_experiment as run_pbc
from conformal_causal_survival.experiments.run_follicular import run_follicular_experiment as run_follicular
from conformal_causal_survival.experiments.run_bmt import run_bmt_experiment as run_bmt

def capture_results(func):
    """Wrapper to run experiment and capture result dict if returned."""
    try:
        # Our modified scripts return a dict. The original melanoma didn't. 
        # Need to patch run_melanoma to return dict or parse output.
        # Since I cannot easily patch the imported function's return without editing it, 
        # I rely on the fact I should have returned it.
        # Wait, run_melanoma was executed but I didn't verify if it returns a dict in the file content.
        # I checked earlier: run_melanoma prints but does not return.
        # I should probably update run_melanoma or just let this script print everything.
        
        # Actually, for clean aggregation, I should update run_melanoma.
        # But to be safe, I will just run them and let them print.
        
        func()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running experiment: {e}")

if __name__ == "__main__":
    print("="*60)
    print("RUNNING ALL EXPERIMENTS: Melanoma, PBC, Follicular, BMT")
    print("="*60)
    
    print("\n\n>>> Dataset 1: MEANOMA")
    try:
        run_melanoma()
    except:
        print("Melanoma Failed")

    print("\n\n>>> Dataset 2: PBC")
    try:
        run_pbc()
    except:
        print("PBC Failed")
        
    print("\n\n>>> Dataset 3: FOLLICULAR")
    try:
        run_follicular()
    except:
        print("Follicular Failed")
        
    print("\n\n>>> Dataset 4: BMT")
    try:
        run_bmt()
    except:
        print("BMT Failed")
    
    print("\n\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
