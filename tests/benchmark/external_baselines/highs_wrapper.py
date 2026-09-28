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
        
        solve_time_ms = round((end_time - start_time) * 1000, 3)
        
        if result.returncode != 0 and "Model status" not in result.stdout and "Model   status" not in result.stdout:
            return {"solver": "HiGHS", "status": "Failed", "solve_time_ms": solve_time_ms}
            
        status = "Optimal" if "Optimal" in result.stdout else "Unknown"
        
        # parse objective
        obj = 0.0
        iters = 0
        vars_count = ""
        cons_count = ""
        nnz_count = ""
        version = ""
        
        for line in result.stdout.split("\n"):
            line_lower = line.lower()
            if "objective value" in line_lower:
                try:
                    obj = float(line.split(":")[1].strip())
                except:
                    pass
            elif "iteration count" in line_lower or "iterations" in line_lower:
                if ":" in line:
                    try:
                        iters = int(line.split(":")[1].strip())
                    except:
                        pass
            elif "number of variables" in line_lower or "number of columns" in line_lower:
                if ":" in line:
                    try:
                        vars_count = int(line.split(":")[1].strip())
                    except:
                        pass
            elif "number of constraints" in line_lower or "number of rows" in line_lower:
                if ":" in line:
                    try:
                        cons_count = int(line.split(":")[1].strip())
                    except:
                        pass
            elif "number of nonzeros" in line_lower:
                if ":" in line:
                    try:
                        nnz_count = int(line.split(":")[1].strip())
                    except:
                        pass
            elif "highs version" in line_lower:
                # e.g. "Running HiGHS 1.5.0 [date: 2023...]"
                version = line.strip()
                    
        return {
            "solver": "HiGHS",
            "status": status,
            "solve_time_ms": solve_time_ms,
            "iterations": iters,
            "objective": obj,
            "variables": vars_count,
            "constraints": cons_count,
            "nnz": nnz_count,
            "version": version
        }
    except FileNotFoundError:
        return {"solver": "HiGHS", "status": "BASELINE UNAVAILABLE"}

