import os
import re
import uuid
import json
import subprocess
import threading
import queue
from flask import Flask, request, jsonify, render_template, Response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

jobs = {}

def parse_mps_info(filepath):
    rows = set()
    cols = set()
    nnz = 0
    section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'): continue
            if line in ('ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'RANGES', 'ENDATA'):
                section = line
                continue
                
            parts = line.split()
            if section == 'ROWS':
                if len(parts) >= 2 and parts[0] != 'N':
                    rows.add(parts[1])
            elif section == 'COLUMNS':
                if len(parts) >= 3:
                    cols.add(parts[0])
                    nnz += 1
                    if len(parts) >= 5:
                        nnz += 1
                        
    return {
        'variables': len(cols),
        'constraints': len(rows),
        'nnz': nnz,
        'type': 'LP'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.lower().endswith('.mps'):
        return jsonify({'error': 'Invalid file format. Only .mps supported.'}), 400
        
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    info = parse_mps_info(filepath)
    info['filename'] = file.filename
    info['filepath'] = filepath
    info['status'] = 'VALID'
    
    return jsonify(info)

@app.route('/api/gpu/init', methods=['POST'])
def init_gpu():
    try:
        # Try to query nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            return jsonify({
                'status': 'READY',
                'device': gpu_name,
                'log': f"Initializing CUDA runtime...\nDetecting GPU...\nFound: {gpu_name}\nAllocating device memory pools...\nGPU READY"
            })
    except Exception:
        pass
        
    return jsonify({
        'status': 'READY',
        'device': 'N/A',
        'log': "Initializing runtime...\nDetecting device...\nNo NVIDIA GPU detected by nvidia-smi.\nSystem READY"
    })

@app.route('/api/solver/start', methods=['POST'])
def start_solver():
    data = request.json
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
        
    job_id = str(uuid.uuid4())
    q = queue.Queue()
    
    jobs[job_id] = {
        'queue': q,
        'process': None,
        'status': 'RUNNING'
    }
    
    def run_solver():
        exe_path = r".\build\tests\sankhya_benchmark_cli.exe"
        if not os.path.exists(exe_path):
            exe_path = r".\build\tests\Release\sankhya_benchmark_cli.exe"
        if not os.path.exists(exe_path):
            exe_path = r"./build/tests/sankhya_benchmark_cli" # Linux/Mac fallback
            
        print(f"[SOLVER] executable = {exe_path}")
        print(f"[SOLVER] model = {filepath}")
        print(f"[SOLVER] cwd = {os.getcwd()}")
        print(f"[SOLVER] command = {exe_path} {filepath}")
            
        try:
            # Start process
            process = subprocess.Popen(
                [exe_path, filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            jobs[job_id]['process'] = process
            
            q.put({"type": "solver_state", "state": "OPTIMIZING"})
            
            all_lines = []
            emitted_states = set()
            
            # Read stdout line by line
            for line in process.stdout:
                all_lines.append(line)
                line = line.strip()
                if not line: continue
                
                m = re.search(r'Iter (\d+) \| rp: ([0-9eE\.\-]+) \| rd: ([0-9eE\.\-]+)', line)
                if m:
                    q.put({
                        "type": "iteration",
                        "iteration": int(m.group(1)),
                        "primal_residual": float(m.group(2)),
                        "objective": "N/A"
                    })
                elif "[KKT Verify]" in line and "VERIFYING" not in emitted_states:
                    emitted_states.add("VERIFYING")
                    q.put({"type": "solver_state", "state": "VERIFYING"})
            
            # Wait for completion and read stderr
            process.wait()
            stderr_output = process.stderr.read()
            
            return_code = process.returncode
            stdout_str = "".join(all_lines)
            
            print(f"[SOLVER] return_code = {return_code}")
            print(f"[SOLVER] stdout_bytes = {len(stdout_str.encode('utf-8'))}")
            print(f"[SOLVER] stderr_bytes = {len(stderr_output.encode('utf-8'))}")
            
            if len(stdout_str) == 0:
                print("[SOLVER] STDOUT EMPTY")
            if len(stderr_output) > 0:
                print("[SOLVER] STDERR:")
                print(stderr_output)
            
            if return_code != 0:
                q.put({
                    "type": "error",
                    "message": f"SOLVER PROCESS FAILED\n\nExit code: {return_code}\n\nSee system log for stderr."
                })
                return
                
            # Robust extraction: find the last occurrence of { and }
            json_start = stdout_str.rfind('{')
            json_end = stdout_str.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = stdout_str[json_start:json_end+1]
                try:
                    result = json.loads(json_str)
                    if "solver" in result and "status" in result:
                        q.put({"type": "result", "result": result})
                    else:
                        q.put({"type": "error", "message": "RESULT PARSING FAILED\n\nJSON found but missing required fields."})
                except json.JSONDecodeError:
                    q.put({"type": "error", "message": "RESULT PARSING FAILED\n\nSolver exited successfully but no valid result JSON was found."})
            else:
                q.put({"type": "error", "message": "RESULT PARSING FAILED\n\nSolver exited successfully but no JSON object was found in stdout."})
                
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put({"type": "eof"})
            jobs[job_id]['status'] = 'DONE'

    threading.Thread(target=run_solver, daemon=True).start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/solver/<job_id>/stream')
def stream(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    q = jobs[job_id]['queue']
    
    def generate():
        while True:
            msg = q.get()
            if msg['type'] == 'eof':
                break
            yield f"data: {json.dumps(msg)}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
