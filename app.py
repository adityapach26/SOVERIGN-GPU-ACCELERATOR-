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
            
        try:
            # Start process
            process = subprocess.Popen(
                [exe_path, filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            jobs[job_id]['process'] = process
            
            q.put({"type": "solver_state", "state": "OPTIMIZING"})
            
            all_lines = []
            emitted_states = set()
            
            for line in process.stdout:
                line = line.strip()
                if not line: continue
                all_lines.append(line)
                
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
                        
            process.wait()
            
            # Extract final JSON robustly
            json_str = None
            for i in range(len(all_lines)-1, -1, -1):
                if all_lines[i] == '{':
                    json_str = "\n".join(all_lines[i:])
                    break
                    
            if json_str:
                try:
                    result = json.loads(json_str)
                    q.put({"type": "result", "result": result})
                except json.JSONDecodeError:
                    q.put({"type": "error", "message": "Failed to parse solver result"})
            else:
                q.put({"type": "error", "message": "Solver produced no JSON output. It may have crashed."})
                
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
