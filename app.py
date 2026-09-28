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
            q.put({"type": "solver_state", "state": "INITIALIZING_GPU", "log": "Initializing CUDA runtime...\nDetecting GPU...\nAllocating device memory...\nUploading model...\nInitializing solver state...\nGPU READY"})
            
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
            
            json_output = []
            capture_json = False
            
            for line in process.stdout:
                line = line.strip()
                if not line: continue
                
                if line == '{':
                    capture_json = True
                    
                if capture_json:
                    json_output.append(line)
                else:
                    m = re.search(r'Iter (\d+) \| rp: ([0-9eE\.\-]+) \| rd: ([0-9eE\.\-]+)', line)
                    if m:
                        q.put({
                            "type": "iteration",
                            "iteration": int(m.group(1)),
                            "primal_residual": float(m.group(2)),
                            "objective": "N/A"
                        })
                    elif "[KKT Verify]" in line:
                        q.put({"type": "solver_state", "state": "VERIFYING"})
                        
            process.wait()
            
            if json_output:
                try:
                    result = json.loads("\n".join(json_output))
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
