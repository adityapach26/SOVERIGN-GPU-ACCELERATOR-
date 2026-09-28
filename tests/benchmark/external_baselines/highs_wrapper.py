import subprocess
import time
import json
import os

def solve_mps(mps_path):
    try:
        start_time = time.perf_counter()
        # Ensure we run highs without interactive prompt
        result = subprocess.run(["highs", "--model_file", mps_path], capture_output=True, text=True, check=False)
        end_time = time.perf_counter()
        
        if result.returncode != 0 and "Model status" not in result.stdout:
            return {"solver": "HiGHS", "status": "Failed", "solve_time_ms": (end_time - start_time)*1000}
            
        status = "Optimal" if "Optimal" in result.stdout else "Unknown"
        
        # parse objective
        obj = 0.0
        iters = 0
        for line in result.stdout.split("\n"):
            if "Objective value" in line:
                try:
                    obj = float(line.split(":")[1].strip())
                except:
                    pass
            if "Iteration count" in line:
                try:
                    iters = int(line.split(":")[1].strip())
                except:
                    pass
                    
        return {
            "solver": "HiGHS",
            "status": status,
            "solve_time_ms": (end_time - start_time) * 1000,
            "iterations": iters,
            "objective": obj
        }
    except FileNotFoundError:
        return {"solver": "HiGHS", "status": "BASELINE UNAVAILABLE"}

