import os
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import *
import tempfile
import h5py

from cad_server_utils import CadServerUtils

app = Flask(__name__)
CORS(app)

# Directory containing web frontend files (index.html, main.js)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_front")
# Directory containing the JSON file
STATIC_DIR = "/home/admin123456/Desktop/work/flask_server/"

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    if os.path.exists(os.path.join(FRONTEND_DIR, filename)):
        return send_from_directory(FRONTEND_DIR, filename)
    return send_from_directory(STATIC_DIR, filename)

@app.route('/cad/read_step', methods=['POST'])
def read_step():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    data = request.form.to_dict()
    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
        file.save(temp_file.name)
        step_file = temp_file.name

        shape = CadServerUtils.read_step(step_file)
        try:
            shape_json = CadServerUtils.convert_shape_to_json(
                name=data.get('name'),
                shape=shape,
                export_edges=data.get('export_edges', False),
                mesh_quality=float(data.get('mesh_quality', 1.0))
            )
            return jsonify(shape_json)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/cad/export_step', methods=['POST'])
def export_step():
    data = request.json
    shape = data.get('shape')  # todo need to adjust
    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
        temp_path = temp_file.name
        CadServerUtils.write_step(shape, temp_path)
        response = send_file(
            temp_path, as_attachment=True,
            download_name=data.get('name'), 
            mimetype='application/step'
        )
        # Delete temp file after sending
        os.unlink(temp_path)
        return response

@app.route('/cad/read_h5', methods=['POST'])
def read_h5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        file.save(temp_file.name)
        h5_path = temp_file.name

        try:
            with h5py.File(h5_path, 'r') as f:
                result = CadServerUtils.convert_h5_to_json(f)
            os.unlink(h5_path)
            return jsonify(result)
        except Exception as e:
            os.unlink(h5_path)
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='192.168.110.54', port=9998, debug=True)

