from flask import Flask, render_template, request, send_file, jsonify
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
import shutil
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Detected: {GPU_NAME}")
    print(f"GPU Memory: {GPU_MEMORY:.2f} GB")
    print(f"Using GPU acceleration for faster processing!")
else:
    print("No GPU detected. Running on CPU (slower)")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    gpu_info = {
        'available': GPU_AVAILABLE,
        'name': GPU_NAME if GPU_AVAILABLE else 'CPU',
        'memory': f"{GPU_MEMORY:.2f} GB" if GPU_AVAILABLE else 'N/A'
    }
    return render_template('index.html', gpu_info=gpu_info)

@app.route('/gpu-status')
def gpu_status():
    if GPU_AVAILABLE:
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return jsonify({
            'available': True,
            'name': GPU_NAME,
            'total_memory': f"{GPU_MEMORY:.2f} GB",
            'allocated': f"{memory_allocated:.2f} GB",
            'reserved': f"{memory_reserved:.2f} GB",
            'utilization': f"{(memory_allocated/GPU_MEMORY)*100:.1f}%"
        })
    return jsonify({'available': False, 'name': 'CPU'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        unique_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        fidelity = float(request.form.get('fidelity', 0.7))
        upscale = request.form.get('upscale', 'true') == 'true'
        face_upsample = request.form.get('face_upsample', 'true') == 'true'
        
        result_dir = os.path.join(app.config['RESULT_FOLDER'], unique_id)
        
        cmd = [
            'python', 'inference_codeformer.py',
            '--input_path', upload_dir,
            '--output_path', result_dir,
            '-w', str(fidelity)
        ]
        
        if upscale:
            cmd.extend(['--bg_upsampler', 'realesrgan'])
        if face_upsample:
            cmd.append('--face_upsample')
        
        if GPU_AVAILABLE:
            if GPU_MEMORY >= 8:
                cmd.extend(['--bg_tile', '800'])
            elif GPU_MEMORY >= 6:
                cmd.extend(['--bg_tile', '600'])
            else:
                cmd.extend(['--bg_tile', '400'])
        else:
            cmd.extend(['--bg_tile', '200'])
        
        try:
            env = os.environ.copy()
            if GPU_AVAILABLE:
                env['CUDA_VISIBLE_DEVICES'] = '0'
            
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            
            final_results = os.path.join(result_dir, 'final_results')
            if os.path.exists(final_results):
                result_files = [f for f in os.listdir(final_results) if allowed_file(f)]
                if result_files:
                    return jsonify({
                        'success': True,
                        'result_id': unique_id,
                        'filename': result_files[0],
                        'gpu_used': GPU_AVAILABLE
                    })
            
            return jsonify({'error': 'Processing failed'}), 500
            
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'Processing error: {e.stderr}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/result/<result_id>/<filename>')
def get_result(result_id, filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_id, 'final_results', filename)
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    return jsonify({'error': 'File not found'}), 404

@app.route('/cleanup', methods=['POST'])
def cleanup():
    data = request.json
    result_id = data.get('result_id')
    if result_id:
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], result_id)
        result_dir = os.path.join(app.config['RESULT_FOLDER'], result_id)
        
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI QUEUE Image Enhancement Server")
    print("="*60)
    if GPU_AVAILABLE:
        print(f"GPU Acceleration: ENABLED")
        print(f"Device: {GPU_NAME}")
        print(f"VRAM: {GPU_MEMORY:.2f} GB")
    else:
        print("GPU Acceleration: DISABLED (CPU mode)")
    print("="*60)
    print("Server starting at http://localhost:5000")
    print("="*60 + "\n")
    
    # app.run(debug=True, host='0.0.0.0', port=5000)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
