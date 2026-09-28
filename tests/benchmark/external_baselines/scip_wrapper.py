import subprocess
import time

def solve_mps(mps_path):
    try:
        start_time = time.perf_counter()
        result = subprocess.run(["scip", "-c", f"read {mps_path} optimize display statistics quit"], capture_output=True, text=True, check=False)
        end_time = time.perf_counter()
        
        if result.returncode != 0 and "SCIP Status" not in result.stdout:
            return {"solver": "SCIP", "status": "Failed", "solve_time_ms": (end_time - start_time)*1000}
            
        status = "Optimal" if "optimal" in result.stdout.lower() else "Unknown"
        
        obj = 0.0
        iters = 0
        for line in result.stdout.split("\n"):
            if "Primal Bound       :" in line:
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = parts[1].split()[0]
                        if val != "-":
                            obj = float(val)
                except:
                    pass
            if "primal LP          :" in line or "dual LP            :" in line:
                try:
                    iters += int(line.split()[3])
                except:
                    pass
                    
        return {
            "solver": "SCIP",
            "status": status,
            "solve_time_ms": (end_time - start_time) * 1000,
            "iterations": iters,
            "objective": obj
        }
    except FileNotFoundError:
        return {"solver": "SCIP", "status": "BASELINE UNAVAILABLE"}

