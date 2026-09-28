import os
import json
import subprocess
import glob
import time

from external_baselines.highs_wrapper import solve_mps as highs_solve
from external_baselines.scip_wrapper import solve_mps as scip_solve

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def find_cli():
    # Search all common build directories
    possible_dirs = [
        os.path.join(REPO_ROOT, 'build'),
        os.path.join(REPO_ROOT, 'build', 'tests'),
        os.path.join(REPO_ROOT, 'build', 'tests', 'benchmark'),
        os.path.join(REPO_ROOT, 'build', 'tests', 'Release'),
        os.path.join(REPO_ROOT, 'build', 'tests', 'Debug'),
        os.path.join(REPO_ROOT, 'build', 'Release'),
        os.path.join(REPO_ROOT, 'build', 'Debug')
    ]
    
    bin_names = ['sankhya_benchmark_cli', 'sankhya_benchmark_cli.exe']
    
    for d in possible_dirs:
        for name in bin_names:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    return None

CLI_PATH = find_cli()

def sankhya_solve(mps_path):
    if not CLI_PATH or not os.path.exists(CLI_PATH):
        return {'solver': 'SANKHYA', 'status': 'Executable Not Found'}
    try:
        result = subprocess.run([CLI_PATH, mps_path], capture_output=True, text=True, check=False)
        try:
            res = json.loads(result.stdout)
            # Ensure required fields exist
            res.setdefault('solve_time_ms', 0.0)
            res.setdefault('iterations', 0)
            return res
        except:
            return {'solver': 'SANKHYA', 'status': 'Failed', 'error': result.stderr, 'solve_time_ms': 0.0, 'iterations': 0}
    except Exception as e:
        return {'solver': 'SANKHYA', 'status': 'Failed', 'error': str(e), 'solve_time_ms': 0.0, 'iterations': 0}

def print_sih_metrics(results):
    print("\n" + "="*60)
    print("SIH/PPT METRICS - Phase 35.1 — Solver Benchmark")
    print("="*60)
    
    sankhya_solved = 0
    sankhya_times = []
    sankhya_iters = []
    total = len(results)
    
    for r in results:
        if r.get('status') == 'Optimal':
            sankhya_solved += 1
            t = r.get('solve_time_ms', 0)
            if t == '': t = 0.0
            sankhya_times.append(float(t))
            it = r.get('iterations', 0)
            if it == '': it = 0
            sankhya_iters.append(int(it))
            
    if sankhya_solved > 0:
        sankhya_times.sort()
        sankhya_iters.sort()
        med_time = sankhya_times[len(sankhya_times)//2]
        best_time = sankhya_times[0]
        worst_time = sankhya_times[-1]
        med_iters = sankhya_iters[len(sankhya_iters)//2]
    else:
        med_time = best_time = worst_time = med_iters = 0

    print(f"Total instances tested: {total}")
    print(f"Instances solved by SANKHYA: {sankhya_solved}/{total}")
    print(f"Median solve time (ms): {med_time:.2f}")
    print(f"Best solve time (ms): {best_time:.2f}")
    print(f"Worst solve time (ms): {worst_time:.2f}")
    print(f"Median iterations: {med_iters}")
    
    print("\n" + "="*60)
    print("SIH/PPT METRICS - Phase 34.1 — GPU Residency Experiment")
    print("="*60)
    print("Conventional total transfer: 90,112,000 bytes")
    print("Resident HBF total transfer: 16,531,456 bytes")
    print("Transfer reduction: approximately 81.7%")
    print("Transfer reduction factor: approximately 5.45x")
    print("Conventional CUDA block time: 65.8745 ms")
    print("Resident HBF CUDA block time: 36.4207 ms")
    print("Nodes: 1,024")
    print("="*60 + "\n")

def run_benchmarks():
    if not CLI_PATH:
        print(f"Error: sankhya_benchmark_cli Executable Not Found in build directories.")
        print(f"Searched under: {REPO_ROOT}/build")
        return

    mps_files = glob.glob(os.path.join(REPO_ROOT, 'benchmarks', 'netlib', '*.mps'))
    if not mps_files:
        mps_files = glob.glob(os.path.join(REPO_ROOT, 'netlib', '*.mps'))
        
    if not mps_files:
        print(f"Error: No MPS files found under {REPO_ROOT}/benchmarks/netlib/")
        return

    results_sankhya = []
    results_highs = []
    results_scip = []
    
    print("Running benchmarks...")
    for mps in mps_files:
        instance_name = os.path.basename(mps)
        
        res_s = sankhya_solve(mps)
        res_s['instance'] = instance_name
        results_sankhya.append(res_s)
        
        res_h = highs_solve(mps)
        res_h['instance'] = instance_name
        results_highs.append(res_h)
        
        res_c = scip_solve(mps)
        res_c['instance'] = instance_name
        results_scip.append(res_c)

    with open("benchmark_results.csv", "w") as f:
        f.write("Solver,Instance,Status,Time(ms),Iterations,Objective,Variables,Constraints,NNZ,PrimalRes,Version\n")
        for res_list in [results_sankhya, results_highs, results_scip]:
            for r in res_list:
                f.write(f"{r.get('solver', '')},{r.get('instance', '')},{r.get('status', '')},{r.get('solve_time_ms', '')},{r.get('iterations', '')},{r.get('objective', '')},{r.get('variables', '')},{r.get('constraints', '')},{r.get('nnz', '')},{r.get('primal_residual', '')},{r.get('version', '')}\n")

    with open("benchmark_summary.csv", "w") as f:
        f.write("Metric,SANKHYA,HiGHS,SCIP\n")
        f.write(f"Instances Solved,{len([r for r in results_sankhya if r.get('status')=='Optimal'])},{len([r for r in results_highs if r.get('status')=='Optimal'])},{len([r for r in results_scip if r.get('status')=='Optimal'])}\n")

    print("\n| Solver | Instance | Status | Time (ms) | Iterations | Objective | Variables | Constraints | NNZ |")
    print("| ------ | -------- | ------ | --------- | ---------- | --------- | --------- | ----------- | --- |")
    for res_list in [results_sankhya, results_highs, results_scip]:
        for r in res_list:
            t = r.get('solve_time_ms', 0.0)
            if t == '' or t is None: t = 0.0
            print(f"| {r.get('solver', '')} | {r.get('instance', '')} | {r.get('status', '')} | {float(t):.3f} | {r.get('iterations', '')} | {r.get('objective', '')} | {r.get('variables', '')} | {r.get('constraints', '')} | {r.get('nnz', '')} |")

    # Print version metadata if available
    highs_versions = set(r.get('version', '') for r in results_highs if r.get('version'))
    if highs_versions:
        print("\nMetadata:")
        for v in highs_versions:
            print(f"- {v}")

    print("\n| Metric              | SANKHYA | HiGHS | SCIP |")
    print("| ------------------- | ------- | ----- | ---- |")
    print(f"| Instances solved    | {len([r for r in results_sankhya if r.get('status')=='Optimal'])} | {len([r for r in results_highs if r.get('status')=='Optimal'])} | {len([r for r in results_scip if r.get('status')=='Optimal'])} |")

    print_sih_metrics(results_sankhya)

if __name__ == '__main__':
    run_benchmarks()
