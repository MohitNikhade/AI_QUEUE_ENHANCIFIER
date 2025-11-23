from flask import Flask, render_template, request, send_file, jsonify
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

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
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            final_results = os.path.join(result_dir, 'final_results')
            if os.path.exists(final_results):
                result_files = [f for f in os.listdir(final_results) if allowed_file(f)]
                if result_files:
                    return jsonify({
                        'success': True,
                        'result_id': unique_id,
                        'filename': result_files[0]
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
    app.run(debug=True, host='0.0.0.0', port=5000)
