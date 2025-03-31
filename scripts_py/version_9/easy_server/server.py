import os
import shutil
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import *
import tempfile
import h5py
import json
import time
import subprocess
import pyvista

from cad_server_utils import CadServerUtils
from cad_server_utils import GmshStepReader
from scripts_py.version_9.simulator.su2_simulator import Su2Simulator
# from ..simulator.su2_simulator import Su2Simulator


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


@app.route('/cad/make_circle_pipe_shell', methods=['POST'])
def make_circle_pipe_shell():
    data = request.json
    path_info = []
    for info in data.get('skeleton_cfgs', []):
        if info['style'] == 'torus':
            info.update({'style': 'circle'})
        path_info.append(info)

    if len(path_info) == 0:
        return jsonify({'error': 'No valid skeleton configurations found'}), 400

    shell = CadServerUtils.make_occ_thru_sections(path_info, ruled=data.get('ruled', True))
    result = CadServerUtils.convert_shape_to_json(
        name=data.get('name', 'circle_pipe_shell'),
        shape=shell,
        export_edges=data.get('export_edges', False),
        mesh_quality=1.0
    )
    return jsonify(result)


@app.route('/cad/make_bspline_surface', methods=['POST'])
def make_bspline_surface():
    data = request.json
    boundary_infos = []
    guide_infos = []

    # Process boundary objects
    for info in data.get('body', {}).get('boundary_objs', []):
        if info['style'] == 'bspline':
            info.update({
                'left_clamp': True,
                'right_clamp': True,
            })
        boundary_infos.append(info)

    # Process guide objects
    for info in data.get('body', {}).get('guide_objs', []):
        info.update({
            'left_clamp': True,
            'right_clamp': True,
        })
        guide_infos.append(info)

    if not boundary_infos:
        return jsonify({'error': 'No valid boundary configurations found'}), 400
    if not guide_infos:
        return jsonify({'error': 'No valid guide configurations found'}), 400

    shell = CadServerUtils.make_occ_shell(boundary_infos, guide_infos)
    result = CadServerUtils.convert_shape_to_json(
        name=data.get('name', 'bspline_surface'),
        shape=shell,
        export_edges=data.get('export_edges', False),
        mesh_quality=1.0
    )
    return jsonify(result)


@app.route('/cad/extract_step_face', methods=['POST'])
def extract_step_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    data = request.form.to_dict()
    shape_jsons = {}

    # Create temp directory
    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded file
    temp_path = os.path.join(temp_dir, f"{file.filename.split('.')[0]}.step")
    file.save(temp_path)

    gmsh_reader = GmshStepReader(temp_path)
    total_faces = len(gmsh_reader.face_cfgs)

    def generate():
        try:
            for i, (face_name, face_cfg) in enumerate(gmsh_reader.face_cfgs.items(), 1):
                if i < total_faces:
                    progress = int((i / total_faces) * 100)
                    progress_data = {
                        'progress': progress,
                        'is_finished': False,
                        'message': f'Processing face {i}/{total_faces}: {face_name}'
                    }
                    time.sleep(0.05)
                    yield json.dumps(progress_data)

                temp_face_path = os.path.join(temp_dir, f"{face_name}.step")
                res = gmsh_reader.extract_face_to_step(face_name, temp_face_path)
                if not res['success']:
                    continue

                shape = CadServerUtils.read_step(temp_face_path)
                shape_json = CadServerUtils.convert_shape_to_json(
                    name=face_name,
                    shape=shape,
                    export_edges=False,
                    mesh_quality=1.0
                )
                shape_jsons[face_name] = shape_json

                # Delete temp face file
                os.remove(temp_face_path)

            gmsh_reader.close()
            # Delete original temp file
            os.remove(temp_path)
            # Remove temp directory if empty
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)

            time.sleep(1)
            final_data = {'progress': 100, 'is_finished': True, 'shape_jsons': shape_jsons}
            yield json.dumps(final_data)

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            raise e

    return Response(generate(), mimetype='text/event-stream')


@app.route('/cad/make_su2_mesh', methods=['POST'])
def make_su2_mesh():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Parse form data
    data = request.form.to_dict()
    
    # Parse numeric values
    max_size = float(data.get('max_size', 0.1))
    min_size = float(data.get('min_size', 0.01))

    # Parse group configurations
    try:
        group_cfgs = json.loads(data.get('group_cfgs', '{}'))
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid group configurations format'}), 400

    try:
        # Create temp directory
        temp_dir = os.path.join(os.getcwd(), 'temp_files')
        os.makedirs(temp_dir, exist_ok=True)
    
        # Save uploaded file
        temp_path = os.path.join(temp_dir, f"{file.filename.split('.')[0]}.step")
        file.save(temp_path)
    
        gmsh_reader = GmshStepReader(temp_path)
    
        defined_idxs = set()
        for name, tag_names in group_cfgs.items():
            tags = [gmsh_reader.face_cfgs[tag_name]['id'] for tag_name in tag_names]
            gmsh_reader.define_group(dim=2, tags=tags, group_name=name)
            defined_idxs.update(tags)

        if len(defined_idxs) != len(gmsh_reader.face_cfgs):
            print('Defined face indices do not match the number of faces')
            return jsonify({'error': 'Defined face indices do not match the number of faces'}), 400
    
        gmsh_reader.define_volume_group()
        gmsh_reader.generate_mesh(dim=3, max_size=max_size, min_size=min_size)
        gmsh_reader.export_su2(os.path.join(temp_dir, 'temp.su2'))
        gmsh_reader.close()
    
        su2_simulator = Su2Simulator(mesh_path=os.path.join(temp_dir, 'temp.su2'))

        # Return mesh information to frontend
        mesh_info = {
            'n_dim': su2_simulator.n_dim,
            'n_elem': su2_simulator.n_elem,
            'n_point': su2_simulator.n_point,
            'n_mark': su2_simulator.n_mark,
            'cells': su2_simulator.cells.tolist(),
            'points': su2_simulator.points.tolist(),
            'marker_dict': su2_simulator.marker_dict
        }
        return jsonify(mesh_info)
    
    except Exception as e:
        print('error:', e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)


@app.route('/simulate/simulate', methods=['POST'])
def simulate():
    data = request.json

    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)    
    
    su2_simulator = Su2Simulator()
    su2_simulator.set_mesh_data(
        n_dim=data['n_dim'],
        n_mark=data['n_mark'],
        cells=np.array(data['cells']),
        points=np.array(data['points']),
        marker_dict=data['marker_dict']
    )
    su2_simulator.prepare_sim_proj(
        proj_dir=os.path.join(temp_dir, 'temp'),
        mesh_filename='temp.su2',
        inlets_info=data['condition']['inlet_cfg'],
        outlets_info=data['condition']['outlet_cfg'],
        walls_info=data['condition']['wall_cfg']
    )

    def generate():
        try:
            cfg_file = 'simulate.cfg'
            run_bash = f"cd {os.path.join(temp_dir, 'temp')}; SU2_CFD {cfg_file}"

            process = subprocess.Popen(
                run_bash,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            output_buffer = []
            last_yield_time = time.time()

            while True:
                output = process.stdout.readline()

                if output == '' and process.poll() is not None:
                    break

                if output:
                    output_buffer.append(output.strip())
                    current_time = time.time()

                    # Send buffered output every second
                    if current_time - last_yield_time >= 1.0:
                        progress_data = {
                            'name': data['name'],
                            'condition_name': data['condition_name'],
                            'is_finished': False, 'simulate_status': 'running',
                            'msg': output_buffer[-1]
                            }
                        yield json.dumps(progress_data)
                        output_buffer = []
                        last_yield_time = current_time

            if process.returncode != 0:
                progress_data = {
                    'name': data['name'],
                    'condition_name': data['condition_name'],
                    'is_finished': True, 'simulate_status': 'failed',
                    'msg': f'Process failed with return code {process.returncode}'
                }
                yield json.dumps(progress_data)

            else:
                # First send completion message
                progress_data = {
                    'name': data['name'],
                    'condition_name': data['condition_name'],
                    'is_finished': False, 'simulate_status': 'completed',
                    'msg': 'Process completed'
                }
                # print(progress_data)
                yield json.dumps(progress_data)

                # Then stream the VTK data in chunks
                res_vtk: pyvista.UnstructuredGrid = pyvista.read(os.path.join(temp_dir, 'temp/flow.vtu'))
                
                # Stream velocity data
                velocity_data = res_vtk.point_data['Velocity']
                chunk_size = 100  # Adjust chunk size as needed
                for i in range(0, len(velocity_data), chunk_size):
                    chunk = velocity_data[i:i + chunk_size]
                    time.sleep(0.1)
                    yield json.dumps({
                        'name': data['name'],
                        'condition_name': data['condition_name'],
                        'is_finished': False, 
                        'simulate_status': 'velocity',
                        'chunk_index': i // chunk_size,
                        'chunk': chunk.tolist(),
                        'msg': f'Velocity data transfer {i // chunk_size}/{len(velocity_data) // chunk_size}'
                    })

                # Stream pressure data 
                pressure_data = res_vtk.point_data['Pressure']
                for i in range(0, len(pressure_data), chunk_size):
                    chunk = pressure_data[i:i + chunk_size]
                    time.sleep(0.1)
                    yield json.dumps({
                        'name': data['name'],
                        'condition_name': data['condition_name'],
                        'is_finished': False, 
                        'simulate_status': 'pressure',
                        'chunk_index': i // chunk_size,
                        'chunk': chunk.tolist(),
                        'msg': f'Pressure data transfer {i // chunk_size}/{len(pressure_data) // chunk_size}'
                    })

                time.sleep(1)
                # Send end marker
                yield json.dumps({
                    'name': data['name'],
                    'condition_name': data['condition_name'],
                    'is_finished': True, 'simulate_status': 'completed',
                    'msg': 'Data transfer complete'
                })

        except Exception as e:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
        finally:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                shutil.rmtree(temp_dir)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='192.168.110.54', port=9998, debug=True)
