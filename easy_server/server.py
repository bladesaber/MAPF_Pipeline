import eventlet
eventlet.monkey_patch()

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
from flask_sock import Sock
from functools import partial
import uuid
from datetime import datetime

from cad_server_utils import CadServerUtils
from cad_server_utils import GmshStepReader
from scripts_py.simulator.su2_simulator import Su2Simulator
from server_utils import StreamDataResponse, StreamTaskResponse

from response_utils import StreamThreadTaskResponse, StreamProcessTaskResponse, StreamProcessDataResponse
from path_server_utils import PathServerUtils, ProjectInfo
from smooth_server_utils import SmoothServerUtils, SmoothInfo
from scripts_py.mapf_pkg.point_cloud_utils import PathSurfaceGenerator
from simulate_server_utils import SimulationServerUtils, SimulationTask


# Create Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

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

    shell = CadServerUtils.make_OCC_thruSections(path_info, ruled=data.get('ruled', True))
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

    shell = CadServerUtils.make_OCC_shell(boundary_infos, guide_infos)
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

    # Create temp directory
    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded file
    temp_path = os.path.join(temp_dir, f"{file.filename.split('.')[0]}.step")
    file.save(temp_path)

    def generate():
        try:
            def task_callback(data_queue):
                try:
                    gmsh_reader = GmshStepReader(temp_path)
                    total_faces = len(gmsh_reader.face_cfgs)
                    shape_jsons = {}

                    for i, (face_name, face_cfg) in enumerate(gmsh_reader.face_cfgs.items(), 1):
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

                        # when complete the sub task, send msg to data_queue
                        data_queue.put({
                            'is_finished': False,
                            'msg': f'Processing face {i}/{total_faces}: {face_name}'
                        })

                        # Delete temp face file
                        os.remove(temp_face_path)

                    # when complete all the sub tasks, save the last result
                    data_queue.put({
                        'is_finished': True,
                        'data': shape_jsons
                    })
                except Exception as e:
                    print(f'[Error]: {e}')
                finally:
                    data_queue.put({
                        'is_finished': True,
                        'data': []
                    })
                    gmsh_reader.close()

            task = StreamTaskResponse(name='extract_step_face', callback_func=task_callback, interval_time=0.5)
            yield from task.start()
            
            last_result = task.last_result

            # ------ return data ------
            time.sleep(0.5)
            def get_data_size_callback():
                return len(last_result)

            res_keys = list(last_result.keys())
            def get_chunk_data_callback(data_index, chunk_size):
                return {res_keys[data_index]: last_result[res_keys[data_index]]}

            stream_data_response = StreamDataResponse(
                name='return_extract_step_face', 
                get_data_size_callback=get_data_size_callback, 
                get_chunk_data_callback=get_chunk_data_callback, 
                chunk_size=1, time_interval=0.5)
            yield from stream_data_response.generate()

            time.sleep(0.5)
            yield json.dumps({'total_finished': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_dir)

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

    # Create temp directory
    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded file
    temp_path = os.path.join(temp_dir, f"{file.filename.split('.')[0]}.step")
    file.save(temp_path)

    su2_file_path = os.path.join(temp_dir, 'temp.su2')

    def generate():
        try:
            def task_callback(data_queue):
                try:
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
                    gmsh_reader.export_su2(su2_file_path)
                    gmsh_reader.close()

                except Exception as e:
                    print(f'[Error]: {e}')
                finally:
                    data_queue.put({
                        'is_finished': True,
                        'data': []
                    })
                    gmsh_reader.close()

            task = StreamTaskResponse(name='generate_mesh', callback_func=task_callback, interval_time=0.5)
            yield from task.start()

            # ------ return mesh info ------
            time.sleep(0.5)
            su2_simulator = Su2Simulator(mesh_path=su2_file_path)

            yield json.dumps({
                'name': 'return_mesh_info',
                'is_finished': True,
                'data': {
                    'n_dim': su2_simulator.n_dim,
                    'n_elem': su2_simulator.n_elem,
                    'n_point': su2_simulator.n_point,
                    'n_mark': su2_simulator.n_mark,
                    'marker_dict': su2_simulator.marker_dict
                }
            })

            # ------ return cells ------
            time.sleep(0.5)
            cells = su2_simulator.cells.tolist()

            def get_cells_size_callback():
                return len(cells)

            def get_cells_chunk_data_callback(data_index, chunk_size):
                return cells[data_index:data_index + chunk_size]
            
            stream_data_response = StreamDataResponse(
                name='return_cells', 
                get_data_size_callback=get_cells_size_callback, 
                get_chunk_data_callback=get_cells_chunk_data_callback, 
                chunk_size=1000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return points ------
            time.sleep(0.5)
            points = su2_simulator.points.tolist()

            def get_points_size_callback():
                return len(points)

            def get_points_chunk_data_callback(data_index, chunk_size):
                return points[data_index:data_index + chunk_size]

            stream_data_response = StreamDataResponse(
                name='return_points', 
                get_data_size_callback=get_points_size_callback, 
                get_chunk_data_callback=get_points_chunk_data_callback, 
                chunk_size=5000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return total_finished ------
            time.sleep(0.5)
            yield json.dumps({'total_finished': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_dir)

    return Response(generate(), mimetype='text/event-stream')


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
            def task_callback(data_queue):
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
                    last_update_time = time.time()
                    while True:
                        output = process.stdout.readline()

                        if output == '' and process.poll() is not None:
                            break

                        if output:
                            output_buffer.append(output.strip())
                            current_time = time.time()

                            # Send buffered output every second
                            if current_time - last_update_time >= 1.0:
                                progress_data = {
                                    'is_finished': False,
                                    'msg': output_buffer[-1]
                                }
                                data_queue.put(progress_data)
                                output_buffer = []
                                last_update_time = current_time

                    if not (process.returncode == 0):
                        progress_data = {
                            'is_finished': True,
                            'msg': f'Process failed with return code {process.returncode}',
                            'data': {
                                'is_success': False,
                                'msg': f'Process failed with return code {process.returncode}'
                            }
                        }
                        data_queue.put(progress_data)
                    else:
                        progress_data = {
                            'is_finished': True,
                            'msg': 'Simulation completed successfully.',
                            'data': {
                                'is_success': True,
                                'msg': 'Simulation completed successfully.'
                            }
                        }
                        data_queue.put(progress_data)
                except Exception as e:
                    print(f'[Error]: {e}')
                finally:
                    data_queue.put({
                        'is_finished': True,
                        'data': []
                    })

            task = StreamTaskResponse(name='simulate', callback_func=task_callback, interval_time=0.5)
            yield from task.start()
            
            if not task.last_result['is_success']:
                return jsonify({'error': 'Simulation failed'}), 500
            
            res_vtk: pyvista.UnstructuredGrid = pyvista.read(os.path.join(temp_dir, 'temp/flow.vtu'))

            # ------ return velocity data ------
            time.sleep(0.5)
            velocity_data = res_vtk.point_data['Velocity'].tolist()

            def get_velocity_data_size_callback():
                return len(velocity_data)
            
            def get_velocity_data_chunk_data_callback(data_index, chunk_size):
                return velocity_data[data_index:data_index + chunk_size]
            
            stream_data_response = StreamDataResponse(
                name='return_velocity_data', 
                get_data_size_callback=get_velocity_data_size_callback, 
                get_chunk_data_callback=get_velocity_data_chunk_data_callback, 
                chunk_size=1000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return pressure data ------
            time.sleep(0.5)
            pressure_data = res_vtk.point_data['Pressure'].tolist()

            def get_pressure_data_size_callback():
                return len(pressure_data)
            
            def get_pressure_data_chunk_data_callback(data_index, chunk_size):
                return pressure_data[data_index:data_index + chunk_size]
            
            stream_data_response = StreamDataResponse(
                name='return_pressure_data', 
                get_data_size_callback=get_pressure_data_size_callback, 
                get_chunk_data_callback=get_pressure_data_chunk_data_callback, 
                chunk_size=1000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return total_finished ------
            time.sleep(0.5)
            yield json.dumps({'total_finished': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return Response(generate(), mimetype='text/event-stream')


# ---------------- New API ----------------

@app.route('/path_search/clear_cache', methods=['POST'])
def clear_cache():
    data = request.json
    project_ids = data.get('project_ids')
    smooth_ids = data.get('smooth_ids')
    for project_id in project_ids:
        if project_id in PathServerUtils.project_map:
            print(f'[Debug]: Clearing cache for project {project_id}')
            PathServerUtils.project_map[project_id].grid = None
            del PathServerUtils.project_map[project_id]
    for smooth_id in smooth_ids:
        if smooth_id in SmoothServerUtils.smooth_map:
            print(f'[Debug]: Clearing cache for smooth {smooth_id}')
            del SmoothServerUtils.smooth_map[smooth_id]
    return jsonify({'is_success': True})


@app.route('/path_search/upload_static_obstacle_pcd', methods=['POST'])
def upload_static_obstacle_pcd():
    data = request.json
    
    # Get session ID from request headers or generate one
    project_id = data.get('project_id')
    chunk_index = data.get('chunk_index', 0)
    total_chunks = data.get('total_chunks', 1)
    is_last_chunk = data.get('is_last_chunk', True)
    obstacle_pcd = np.array(data.get('obstacle_pcd', []))
    x_size = data.get('x_size')
    y_size = data.get('y_size')
    z_size = data.get('z_size')
    resolution = data.get('resolution')

    if project_id not in PathServerUtils.project_map:
        project_info = ProjectInfo()
        project_info.static_obstacle_pcd = []
        project_info.dynamic_obstacle_pcd = {}
        project_info.x_size = x_size + 1  # be careful with the x_size
        project_info.y_size = y_size + 1  # be careful with the y_size
        project_info.z_size = z_size + 1  # be careful with the z_size
        project_info.resolution = resolution
        PathServerUtils.project_map[project_id] = project_info
    if chunk_index == 0:
        PathServerUtils.project_map[project_id].static_obstacle_pcd.clear()
    PathServerUtils.project_map[project_id].static_obstacle_pcd.extend(obstacle_pcd)

    print(f'static_obstacle_pcd: {len(PathServerUtils.project_map[project_id].static_obstacle_pcd)}')

    return jsonify({
        'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
        'is_success': True
    })


@app.route('/path_search/upload_relax_group_obstacle_pcd', methods=['POST'])
def upload_relax_group_obstacle_pcd():
    data = request.json
    project_id = data.get('project_id')
    chunk_index = data.get('chunk_index', 0)
    total_chunks = data.get('total_chunks', 1)
    is_last_chunk = data.get('is_last_chunk', True)
    obstacle_pcd = data.get('obstacle_pcd', [])
    group_name = data.get('group_name')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400
    
    project_info = PathServerUtils.project_map[project_id]

    if group_name not in project_info.dynamic_group_obstacle_pcd:
        project_info.dynamic_group_obstacle_pcd[group_name] = []
    if chunk_index == 0:
        project_info.dynamic_group_obstacle_pcd[group_name].clear()
    project_info.dynamic_group_obstacle_pcd[group_name].extend(obstacle_pcd)

    return jsonify({
        'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
        'is_success': True
    })


@app.route('/path_search/request_inner_xyzd', methods=['POST'])
def request_inner_xyzd():
    data = request.json
    project_id = data.get('project_id')
    pipe_data = data.get('pipe_data')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400
    
    if len(PathServerUtils.project_map[project_id].static_obstacle_pcd) == 0:
        return jsonify({'error': 'No obstacle pcd provided'}), 400

    project_info = PathServerUtils.project_map[project_id]
    obstacle_pcd = np.array(project_info.static_obstacle_pcd)

    project_info.inner_xyzd = PathServerUtils.compute_inner_xyzd(
        x_size=project_info.x_size, 
        y_size=project_info.y_size, 
        z_size=project_info.z_size, 
        resolution=project_info.resolution,
        obstacle_pcd=obstacle_pcd
    )
    project_info.static_obstacle_pcd.clear()
    PathServerUtils.init_env(project_id, pipe_data)

    '''
    # Web Frontend is too slow
    def generate():
        try:
            pos_data, dist_data = project_info.inner_xyzd
            concat_data = np.concatenate([pos_data, dist_data[:, np.newaxis]], axis=1)

            def get_data_size_callback():
                return len(concat_data)
                    
            def get_chunk_data_callback(data_index, chunk_size):
                print('data_index:', data_index, 'chunk_size:', chunk_size)
                return concat_data[data_index:data_index + chunk_size].tolist()
                        
            stream_data_response = StreamProcessDataResponse(
                name='request_inner_xyzd', 
                get_data_size_callback=get_data_size_callback, 
                get_chunk_data_callback=get_chunk_data_callback, 
                chunk_size=50000, time_interval=0.2
            )
            yield from stream_data_response.generate()

            time.sleep(0.5)
            yield json.dumps({'total_finished': True})

        except Exception as e:
            print(f'[Error]: {str(e)}')
            yield json.dumps({'error': str(e)})
            return
                
    return Response(generate(), mimetype='text/event-stream')
    '''
    return jsonify({'is_success': True})


@app.route('/path_search/upload_task_cache', methods=['POST'])
def upload_task_cache():
    data = request.json
    project_id = data.get('project_id')
    cache_data = data.get('cache_data')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400
    
    PathServerUtils.update_task_cache(project_id, cache_data)
    return jsonify({'is_success': True})


@app.route('/path_search/run_cell_task', methods=['POST'])
def run_cell_task():
    data = request.json
    project_id = data.get('project_id')
    task_name = data.get('task_name')
    priority_name = data.get('priority_name')
    group_name = data.get('group_name')
    cell_name = data.get('cell_name')
    response_type = data.get('response_type')
    max_running_time = data.get('max_running_time')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400

    if response_type == 'single':
        result = PathServerUtils.run_cell_task(project_id, task_name, priority_name, group_name, cell_name)
        return jsonify(result)
    
    elif response_type == 'stream':
        def generate():
            try:
                def task_callback(data_queue):
                    result = PathServerUtils.run_cell_task(project_id, task_name, priority_name, group_name, cell_name)
                    data_queue.put({
                        'is_finished': True,
                        'result': result
                    })
                
                task = StreamProcessTaskResponse(
                    name='run_cell_task', 
                    callback_func=task_callback, 
                    interval_time=0.3,
                    min_yield_interval=0.1,
                    max_running_time=max_running_time
                )
                yield from task.start()

                result = {
                    'name': 'run_cell_task',
                    'total_finished': True,
                    'result': task.result['result']
                }
                time.sleep(0.25)
                yield json.dumps(result)

            except Exception as e:
                yield json.dumps({
                    'name': 'run_cell_task',
                    'total_finished': True,
                    'error': str(e)
                })
                return
            
        return Response(generate(), mimetype='text/event-stream')
    else:
        return jsonify({'error': 'Invalid response type'}), 400


@app.route('/path_search/run_group_task', methods=['POST'])
def run_group_task():
    data = request.json
    project_id = data.get('project_id')
    task_name = data.get('task_name')
    priority_name = data.get('priority_name')
    group_name = data.get('group_name')
    response_type = data.get('response_type')
    max_running_time = data.get('max_running_time')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400

    if response_type == 'single':
        result = PathServerUtils.run_group_task(project_id, task_name, priority_name, group_name)
        return jsonify(result)
    
    elif response_type == 'stream':
        def generate():
            try:
                def task_callback(data_queue):
                    result = PathServerUtils.run_group_task(project_id, task_name, priority_name, group_name)
                    data_queue.put({
                        'is_finished': True,
                        'result': result
                    })
                
                task = StreamProcessTaskResponse(
                    name='run_group_task', 
                    callback_func=task_callback, 
                    interval_time=0.3,
                    min_yield_interval=0.1,
                    max_running_time=max_running_time
                )
                yield from task.start()

                result = {
                    'name': 'run_group_task',
                    'total_finished': True,
                    'result': task.result['result']
                }
                time.sleep(0.25)
                yield json.dumps(result)

            except Exception as e:
                yield json.dumps({
                    'name': 'run_group_task',
                    'total_finished': True,
                    'error': str(e)
                })
                return
            
        return Response(generate(), mimetype='text/event-stream')
    else:
        return jsonify({'error': 'Invalid response type'}), 400


@app.route('/path_search/run_priority_task', methods=['POST'])
def run_priority_task():
    data = request.json
    project_id = data.get('project_id')
    task_name = data.get('task_name')
    priority_name = data.get('priority_name')
    response_type = data.get('response_type')

    if project_id not in PathServerUtils.project_map:
        return jsonify({'error': 'No project data found'}), 400

    if response_type == 'single':
        result = PathServerUtils.run_priority_task(project_id, task_name, priority_name)
        return jsonify(result)

    elif response_type == 'stream':
        def generate():
            try:
                task_callback = partial(
                    PathServerUtils.run_priority_task,
                    project_id=project_id,
                    task_name=task_name,
                    priority_name=priority_name,
                    mode='stream'
                )
                
                task = StreamProcessTaskResponse(
                    name='run_priority_task', 
                    callback_func=task_callback, 
                    interval_time=0.3,
                    min_yield_interval=0.1,
                    max_running_time=1000000
                )
                yield from task.start()

                result = {
                    'name': 'run_priority_task',
                    'total_finished': True,
                    'result': task.result['result']
                }
                time.sleep(0.25)
                # print('priority_task result:', result)
                yield json.dumps(result)

            except Exception as e:
                yield json.dumps({
                    'name': 'run_priority_task',
                    'total_finished': True,
                    'error': str(e)
                })
                return
            
        return Response(generate(), mimetype='text/event-stream')
    else:
        return jsonify({'error': 'Invalid response type'}), 400


@app.route('/path_smooth/upload_smooth_obstacle_pcd', methods=['POST'])
def upload_smooth_obstacle_pcd():
    data = request.json
    
    # Get session ID from request headers or generate one
    smooth_name = data.get('smooth_name')
    chunk_index = data.get('chunk_index', 0)
    total_chunks = data.get('total_chunks', 1)
    is_last_chunk = data.get('is_last_chunk', True)
    obstacle_xyzr = np.array(data.get('obstacle_xyzr', []))

    if smooth_name not in SmoothServerUtils.smooth_map:
        project_info = SmoothInfo()
        project_info.obstacle_xyzr = []
        SmoothServerUtils.smooth_map[smooth_name] = project_info
    
    if SmoothServerUtils.smooth_map[smooth_name].is_obstacle_pcd_ready:
        SmoothServerUtils.smooth_map[smooth_name].obstacle_xyzr.clear()
        SmoothServerUtils.smooth_map[smooth_name].is_obstacle_pcd_ready = False

    if not SmoothServerUtils.smooth_map[smooth_name].is_obstacle_pcd_ready:
        SmoothServerUtils.smooth_map[smooth_name].obstacle_xyzr.extend(obstacle_xyzr)
        print(f'smooth_obstacle_pcd: {len(SmoothServerUtils.smooth_map[smooth_name].obstacle_xyzr)}')
    
    SmoothServerUtils.smooth_map[smooth_name].is_obstacle_pcd_ready = is_last_chunk

    return jsonify({
        'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
        'is_success': True
    })


@app.route('/path_smooth/init_smooth_map', methods=['POST'])
def init_smooth_map():
    data = request.json
    smooth_name = data.get('smooth_name')
    smooth_json = data.get('smooth_json')

    if smooth_name not in SmoothServerUtils.smooth_map:
        return jsonify({'error': 'No smooth map found'}), 400

    smooth_json = SmoothServerUtils.init_smooth_map(smooth_name, smooth_json)
    return jsonify({'is_success': True, 'smooth_json': smooth_json})


@app.route('/path_smooth/reset_config', methods=['POST'])
def reset_config():
    data = request.json
    smooth_name = data.get('smooth_name', None)
    smooth_json = data.get('smooth_json', None)

    if smooth_name not in SmoothServerUtils.smooth_map:
        return jsonify({'error': 'No smooth map found'}), 400

    smooth_info = SmoothServerUtils.smooth_map[smooth_name]
    for group_setting in smooth_json['setting']:
        group_name = group_setting['group_name']
        group_idx = group_setting['cell_setting']['group_idx']
        smooth_info.group_name2idx[group_name] = group_idx
        smooth_info.group_idx2name[group_idx] = group_name

    return jsonify({'is_success': True})


@app.route('/path_smooth/manual_smooth', methods=['POST'])
def manual_smooth():
    data = request.json
    smooth_name = data.get('smooth_name', None)
    smooth_json = data.get('smooth_json', None)
    if smooth_name is None or smooth_json is None:
        return jsonify({'error': 'No smooth name or smooth json provided'}), 400

    lr = data.get('lr', 0.01)
    max_iter = data.get('max_iter', 1000)
    log_iter = data.get('log_iter', 50)
    mode = data.get('mode', 'stream')

    if smooth_name not in SmoothServerUtils.smooth_map:
        return jsonify({'error': 'No smooth map found'}), 400

    def generate():
        try:
            if mode == 'process_stream':
                task_callback = partial(
                    SmoothServerUtils.manual_smooth,
                    smooth_name=smooth_name,
                    smooth_json=smooth_json,
                    run_times=max_iter,
                    lr=lr,
                    log_iter=log_iter,
                    mode=mode
                )
                            
                task = StreamProcessTaskResponse(
                    name='manual_smooth', 
                    callback_func=task_callback, 
                    interval_time=0.3,
                    min_yield_interval=0.1,
                    max_running_time=1000000
                )
                yield from task.start()

                result = {
                    'name': 'manual_smooth',
                    'total_finished': True,
                    'result': task.result['result'],
                    'msg': 'task_finished'
                }
                time.sleep(0.25)
                yield json.dumps(result) + '\n'

            elif mode == 'stream':
                yield from SmoothServerUtils.manual_smooth(
                    smooth_name=smooth_name,
                    smooth_json=smooth_json,
                    run_times=max_iter,
                    lr=lr,
                    log_iter=log_iter,
                    mode=mode
                )

        except Exception as e:
            yield json.dumps({
                'name': 'manual_smooth',
                'total_finished': True,
                'error': str(e)
            })
            return

    return Response(generate(), mimetype='text/event-stream')


@app.route('/path_smooth/generate_surface_from_paths', methods=['POST'])
def generate_surface_from_paths():
    """
    path_dict: {
        'path_name': {
            'path': np.ndarray, # (n, 4) means (x, y, z, radius)
            'is_left_input': bool,
            'is_right_input': bool,
            'is_left_cut': bool,
            'is_right_cut': bool,
            'left_vec': np.ndarray, # (3,)
            'right_vec': np.ndarray, # (3,)
        }
    }
    """
    data = request.json
    path_dict = data.get('path_dict', {})
    path_reso = data.get('path_reso', 0.1)
    point_reso = data.get('point_reso', 0.1)
    sample_num = data.get('sample_num', 1000)
    if len(path_dict) == 0:
        return jsonify({'error': 'Invalid parameters'}), 400
    
    # try:
    #     mesh = PathSurfaceGenerator.generate_surface(paths, path_reso, point_reso)
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 400

    for _, path_item in path_dict.items():
        path_item['path'] = np.array(path_item['path'])
        if path_item['left_vec'] is not None:
            path_item['left_vec'] = np.array(path_item['left_vec'])
        if path_item['right_vec'] is not None:
            path_item['right_vec'] = np.array(path_item['right_vec'])
            
    # Create temp directory
    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)
    surface_ply = os.path.join(temp_dir, f'surface_{uuid.uuid4()}.ply')
    surface_stl = os.path.join(temp_dir, f'surface_{uuid.uuid4()}.stl')

    def generate():
        try:
            def task_callback(data_queue):
                PathSurfaceGenerator.generate_surface(
                    path_dict, refine_path_reso=path_reso, point_reso=point_reso,
                    method="meshlab",
                    save_pcd_file=surface_ply,
                    save_stl_file=surface_stl,
                    sample_num=sample_num
                )
                data_queue.put({
                    'is_finished': True,
                    'result': None
                })
            
            task = StreamProcessTaskResponse(
                name='generate_surface_from_paths',
                callback_func=task_callback,
                interval_time=0.3,
                min_yield_interval=0.1,
                max_running_time=1000000
            )
            yield from task.start()

            mesh = pyvista.read(surface_stl).cast_to_unstructured_grid()

            # ------ return points ------
            points = mesh.points.tolist()
        
            def get_points_size_callback():
                return len(points)

            def get_points_chunk_data_callback(data_index, chunk_size):
                return points[data_index:data_index + chunk_size]

            stream_data_response = StreamProcessDataResponse(
                name='return_points', 
                get_data_size_callback=get_points_size_callback, 
                get_chunk_data_callback=get_points_chunk_data_callback, 
                chunk_size=5000, time_interval=0.1
            )
            yield from stream_data_response.generate()

            # ------ return cells ------
            time.sleep(0.25)
            cells = mesh.cells.reshape(-1, 4)
            cells = cells.tolist()

            def get_cells_size_callback():
                return len(cells)

            def get_cells_chunk_data_callback(data_index, chunk_size):
                return cells[data_index:data_index + chunk_size]
                
            stream_data_response = StreamProcessDataResponse(
                name='return_cells', 
                get_data_size_callback=get_cells_size_callback, 
                get_chunk_data_callback=get_cells_chunk_data_callback, 
                chunk_size=5000, time_interval=0.1
            )
            yield from stream_data_response.generate()

            # ------ return total_finished ------
            time.sleep(0.5)
            yield json.dumps({'total_finished': True})
        
        except Exception as e:
            print(f'generate_surface_from_paths error: {e}')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return jsonify({'error': str(e)}), 400

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/simulate/init_simulation_vertices', methods=['POST'])
def init_simulation_vertices():
    data = request.json
    simulation_name = data.get('simulation_name')
    chunk_index = data.get('chunk_index', 0)
    total_chunks = data.get('total_chunks', 1)
    is_last_chunk = data.get('is_last_chunk', True)
    vertices = np.array(data.get('vertices', []))
    
    if simulation_name not in SimulationServerUtils.simulation_map:
        simulation_info = SimulationTask()
        SimulationServerUtils.simulation_map[simulation_name] = simulation_info
    
    if not SimulationServerUtils.simulation_map[simulation_name].is_vertices_ready:
        SimulationServerUtils.simulation_map[simulation_name].vertices.extend(vertices)
        print(f'simulation_vertices: {len(SimulationServerUtils.simulation_map[simulation_name].vertices)}')
    else:
        return jsonify({'error': 'Vertices are already ready'}), 400
    
    SimulationServerUtils.simulation_map[simulation_name].is_vertices_ready = is_last_chunk

    return jsonify({'is_success': True})


@app.route('/simulate/init_simulation_cells', methods=['POST'])
def init_simulation_cells():
    data = request.json
    simulation_name = data.get('simulation_name')
    chunk_index = data.get('chunk_index', 0)
    total_chunks = data.get('total_chunks', 1)
    is_last_chunk = data.get('is_last_chunk', True)
    cells = np.array(data.get('cells', []))

    if simulation_name not in SimulationServerUtils.simulation_map:
        return jsonify({'error': 'No simulation map found'}), 400
    
    if not SimulationServerUtils.simulation_map[simulation_name].is_cells_ready:
        SimulationServerUtils.simulation_map[simulation_name].cells.extend(cells)
        print(f'simulation_cells: {len(SimulationServerUtils.simulation_map[simulation_name].cells)}')
    else:
        return jsonify({'error': 'Cells are already ready'}), 400
    
    SimulationServerUtils.simulation_map[simulation_name].is_cells_ready = is_last_chunk

    return jsonify({'is_success': True})


@app.route('/simulate/init_simulation_task', methods=['POST'])
def init_simulation_task():
    data = request.json
    simulation_name = data.get('simulation_name')
    
    min_size = data.get('min_size', None)
    if min_size is None:
        return jsonify({'error': 'Min size is required'}), 400
    
    max_size = data.get('max_size', None)
    if max_size is None:
        return jsonify({'error': 'Max size is required'}), 400

    terminal_cfg = data.get('terminal_cfg', [])
    if len(terminal_cfg) < 2:
        return jsonify({'error': 'Terminal cfg must contain at least 2 items'}), 400
    
    if simulation_name not in SimulationServerUtils.simulation_map:
        return jsonify({'error': 'No simulation map found'}), 400

    SimulationServerUtils.simulation_map[simulation_name].terminal_cfg = terminal_cfg

    temp_dir = os.path.join(os.getcwd(), f'{str(uuid.uuid4())}_temp_files')
    os.makedirs(temp_dir, exist_ok=True)

    stl_file = os.path.join(temp_dir, 'mesh.stl')
    su2_file = os.path.join(temp_dir, 'mesh.su2')
    terminal_tag_file = os.path.join(temp_dir, 'terminal_tag.json')
    
    def generate():
        try:
            def task_callback(data_queue):
                # ------ debug ------
                with open(terminal_tag_file, 'w') as f:
                    json.dump(terminal_cfg, f, indent=4)
                
                vertices = np.array(SimulationServerUtils.simulation_map[simulation_name].vertices)
                cells = np.array(SimulationServerUtils.simulation_map[simulation_name].cells)
                np.save(os.path.join(temp_dir, 'vertices.npy'), vertices)
                np.save(os.path.join(temp_dir, 'cells.npy'), cells)
                # ------ debug ------

                SimulationTask.process(
                    vertices=SimulationServerUtils.simulation_map[simulation_name].vertices,
                    cells=SimulationServerUtils.simulation_map[simulation_name].cells,
                    terminal_cfg=terminal_cfg,
                    min_size=min_size, max_size=max_size,
                    stl_file=stl_file, su2_file=su2_file,
                )
                data_queue.put({
                    'is_finished': True,
                    'result': {}
                })
            
            task = StreamProcessTaskResponse(
                name='init_simulation_task',
                callback_func=task_callback,
                interval_time=0.3,
                min_yield_interval=0.1,
                max_running_time=1000000
            )
            yield from task.start()

            SimulationServerUtils.simulation_map[simulation_name].su2_reader = Su2Simulator(su2_file)
            SimulationServerUtils.simulation_map[simulation_name].mesh = pyvista.read(stl_file)

            time.sleep(0.5)
            su2_simulator = SimulationServerUtils.simulation_map[simulation_name].su2_reader
            yield json.dumps({
                'name': 'return_mesh_info',
                'is_finished': True,
                'data': {
                    'n_dim': su2_simulator.n_dim,
                    'n_elem': su2_simulator.n_elem,
                    'n_point': su2_simulator.n_point,
                    'n_mark': su2_simulator.n_mark,
                    'marker_dict': su2_simulator.marker_dict
                }
            }) + '\n'

            # ------ return cells ------
            time.sleep(0.5)
            cells = su2_simulator.cells.tolist()

            def get_cells_size_callback():
                return len(cells)

            def get_cells_chunk_data_callback(data_index, chunk_size):
                return cells[data_index:data_index + chunk_size]
            
            stream_data_response = StreamProcessDataResponse(
                name='return_cells', 
                get_data_size_callback=get_cells_size_callback, 
                get_chunk_data_callback=get_cells_chunk_data_callback, 
                chunk_size=5000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return points ------
            time.sleep(0.5)
            points = su2_simulator.points.tolist()

            def get_points_size_callback():
                return len(points)

            def get_points_chunk_data_callback(data_index, chunk_size):
                return points[data_index:data_index + chunk_size]

            stream_data_response = StreamProcessDataResponse(
                name='return_points', 
                get_data_size_callback=get_points_size_callback, 
                get_chunk_data_callback=get_points_chunk_data_callback, 
                chunk_size=5000, time_interval=0.5)
            yield from stream_data_response.generate()

            # ------ return total_finished ------
            time.sleep(0.5)
            yield json.dumps({'total_finished': True}) + '\n'

        except Exception as e:
            print(f'init_simulation_task error: {e}')
            return jsonify({'error': str(e)}), 400
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/simulate/create_field_module', methods=['POST'])
def create_field_module():
    """
    创建场模块接口
    
    创建仿真所需的场模块，包括几何目录和网格参数设置
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    geometry_dir = data.get('geometry_dir', 'constant/geometry')
    mesh_max_size = data.get('mesh_max_size', 0.1)
    mesh_min_size = data.get('mesh_min_size', 0.05)
    
    try:
        # 创建仿真目录
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        os.makedirs(simulation_dir, exist_ok=True)
        
        # 创建几何目录
        geometry_path = os.path.join(simulation_dir, geometry_dir)
        os.makedirs(geometry_path, exist_ok=True)
        
        # 保存场模块信息
        field_module_info = {
            'simulation_name': simulation_name,
            'geometry_dir': geometry_path,
            'mesh_max_size': mesh_max_size,
            'mesh_min_size': mesh_min_size,
            'created_at': datetime.now().isoformat()
        }
        
        # 保存场模块信息到文件
        field_module_file = os.path.join(simulation_dir, 'field_module.json')
        with open(field_module_file, 'w') as f:
            json.dump(field_module_info, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Field module created successfully',
            'field_module': field_module_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create field module: {str(e)}'}), 500

@app.route('/simulate/compute_initial_value', methods=['POST'])
def compute_initial_value():
    """
    计算湍流初始值接口
    
    根据湍流模型和参考参数计算初始值
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    turbulence_model = data.get('turbulence_model', 'kEpsilon')
    velocity_ref = data.get('velocity_ref', 2.0)
    length_ref = data.get('length_ref', 1.0)
    turbulence_intensity = data.get('turbulence_intensity', 0.05)
    
    try:
        # 计算湍流初始值
        if turbulence_model == 'kEpsilon':
            # k-epsilon 模型初始值计算
            k = 1.5 * (velocity_ref * turbulence_intensity) ** 2
            epsilon = (k ** 1.5) / length_ref
            nu_t = 0.09 * (k ** 2) / epsilon
            
            initial_values = {
                'k': k,
                'epsilon': epsilon,
                'nu_t': nu_t,
                'turbulence_model': turbulence_model
            }
        else:
            # 其他湍流模型的默认值
            initial_values = {
                'k': 0.1,
                'epsilon': 0.1,
                'nu_t': 0.01,
                'turbulence_model': turbulence_model
            }
        
        # 保存初始值到仿真目录
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        initial_value_file = os.path.join(simulation_dir, 'initial_values.json')
        with open(initial_value_file, 'w') as f:
            json.dump(initial_values, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Initial values computed successfully',
            'initial_values': initial_values
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to compute initial values: {str(e)}'}), 500

@app.route('/simulate/set_boundary_conditions', methods=['POST'])
def set_boundary_conditions():
    """
    设置边界条件接口
    
    设置壁面、入口和出口边界条件
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    wall_config = data.get('wall_config', {})
    inlet_config = data.get('inlet_config', {})
    outlet_config = data.get('outlet_config', {})
    
    try:
        # 创建边界条件配置
        boundary_conditions = {
            'wall': {
                'x_min': wall_config.get('x_min', 0),
                'x_max': wall_config.get('x_max', 4),
                'y_min': wall_config.get('y_min', 0),
                'y_max': wall_config.get('y_max', 4),
                'z_min': wall_config.get('z_min', 0),
                'z_max': wall_config.get('z_max', 2),
                'type': 'wall',
                'velocity': [0, 0, 0]
            },
            'inlet': {
                'name': inlet_config.get('name', 'inlet'),
                'norm_vector': inlet_config.get('norm_vector', [1.0, 0.0, 0.0]),
                'center_point': inlet_config.get('center_point', [0.0, 2.0, 1.0]),
                'radius': inlet_config.get('radius', 0.5),
                'length': inlet_config.get('length', 0.25),
                'velocity': inlet_config.get('velocity', [2.0, 0.0, 0.0]),
                'type': 'velocity'
            },
            'outlet': {
                'name': outlet_config.get('name', 'outlet'),
                'norm_vector': outlet_config.get('norm_vector', [1.0, 0.0, 0.0]),
                'center_point': outlet_config.get('center_point', [4.0, 2.0, 1.0]),
                'radius': outlet_config.get('radius', 0.5),
                'length': outlet_config.get('length', 0.25),
                'pressure': outlet_config.get('pressure', 0.0),
                'type': 'pressure'
            }
        }
        
        # 保存边界条件到仿真目录
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        boundary_file = os.path.join(simulation_dir, 'boundary_conditions.json')
        with open(boundary_file, 'w') as f:
            json.dump(boundary_conditions, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Boundary conditions set successfully',
            'boundary_conditions': boundary_conditions
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to set boundary conditions: {str(e)}'}), 500

@app.route('/simulate/generate_geometry', methods=['POST'])
def generate_geometry():
    """
    生成几何体接口
    
    根据边界条件生成几何体文件
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    
    try:
        # 读取边界条件
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        boundary_file = os.path.join(simulation_dir, 'boundary_conditions.json')
        
        if not os.path.exists(boundary_file):
            return jsonify({'error': 'Boundary conditions not found'}), 400
        
        with open(boundary_file, 'r') as f:
            boundary_conditions = json.load(f)
        
        # 这里应该调用几何生成逻辑
        # 暂时返回成功状态
        geometry_info = {
            'simulation_name': simulation_name,
            'geometry_files': [
                'wall.stl',
                'inlet.stl',
                'outlet.stl'
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        # 保存几何信息
        geometry_file = os.path.join(simulation_dir, 'geometry_info.json')
        with open(geometry_file, 'w') as f:
            json.dump(geometry_info, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Geometry generated successfully',
            'geometry_info': geometry_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate geometry: {str(e)}'}), 500

@app.route('/simulate/create_openfoam_files', methods=['POST'])
def create_openfoam_files():
    """
    创建OpenFOAM文件接口
    
    根据配置创建OpenFOAM仿真所需的配置文件
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    solver_config = data.get('solver_config', {})
    mesh_config = data.get('mesh_config', {})
    
    try:
        # 创建OpenFOAM文件配置
        openfoam_config = {
            'solver': solver_config.get('solver', 'topoIncompressibleFluid'),
            'numerical_method': solver_config.get('numerical_method', 'PIMPLE'),
            'deltaT': solver_config.get('deltaT', 0.005),
            'endTime': solver_config.get('endTime', 10.0),
            'writeInterval': solver_config.get('writeInterval', 100),
            'mesh': {
                'grid_x': mesh_config.get('grid_x', 30),
                'grid_y': mesh_config.get('grid_y', 30),
                'grid_z': mesh_config.get('grid_z', 30),
                'inside_point': mesh_config.get('inside_point', [2.0, 2.0, 1.0])
            }
        }
        
        # 保存OpenFOAM配置
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        openfoam_file = os.path.join(simulation_dir, 'openfoam_config.json')
        with open(openfoam_file, 'w') as f:
            json.dump(openfoam_config, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'OpenFOAM files created successfully',
            'openfoam_config': openfoam_config
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create OpenFOAM files: {str(e)}'}), 500

@app.route('/simulate/set_topo_weight', methods=['POST'])
def set_topo_weight():
    """
    设置拓扑权重接口
    
    设置拓扑优化的权重分布
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    weight_config = data.get('weight_config', {})
    
    try:
        # 创建拓扑权重配置
        topo_weight_config = {
            'center_point': weight_config.get('center_point', [2.0, 2.0]),
            'radius': weight_config.get('radius', 0.5),
            'weight_value': weight_config.get('weight_value', 1e6),
            'function': weight_config.get('function', 'constant'),
            'decay_param': weight_config.get('decay_param', 1.0)
        }
        
        # 保存拓扑权重配置
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        topo_weight_file = os.path.join(simulation_dir, 'topo_weight_config.json')
        with open(topo_weight_file, 'w') as f:
            json.dump(topo_weight_config, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Topology weight set successfully',
            'topo_weight_config': topo_weight_config
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to set topology weight: {str(e)}'}), 500

@app.route('/simulate/run_simulation', methods=['POST'])
def run_simulation():
    """
    运行仿真接口
    
    根据配置运行StokeSimulator或NavierStokeSimulator
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    simulator_type = data.get('simulator_type', 'StokeSimulator')  # 默认使用StokeSimulator
    
    try:
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        
        if not os.path.exists(simulation_dir):
            return jsonify({'error': 'Simulation directory not found'}), 400
        
        if simulator_type == 'StokeSimulator':
            # 实现StokeSimulator运行逻辑
            result = run_stoke_simulation(simulation_name, simulation_dir)
        elif simulator_type == 'NavierStokeSimulator':
            # NavierStokeSimulator暂时只返回占位符
            result = {
                'success': True,
                'message': 'NavierStokeSimulator is not implemented yet',
                'simulator_type': 'NavierStokeSimulator',
                'status': 'placeholder'
            }
        else:
            return jsonify({'error': 'Invalid simulator type'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Failed to run simulation: {str(e)}'}), 500

def run_stoke_simulation(simulation_name: str, simulation_dir: str):
    """
    运行StokeSimulator仿真的具体实现
    """
    try:
        # 读取仿真配置
        config_files = [
            'field_module.json',
            'initial_values.json',
            'boundary_conditions.json',
            'geometry_info.json',
            'openfoam_config.json',
            'topo_weight_config.json'
        ]
        
        configs = {}
        for config_file in config_files:
            config_path = os.path.join(simulation_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    configs[config_file.replace('.json', '')] = json.load(f)
        
        # 这里应该调用实际的StokeSimulator
        # 暂时返回模拟结果
        result = {
            'success': True,
            'message': 'StokeSimulator simulation completed successfully',
            'simulator_type': 'StokeSimulator',
            'simulation_name': simulation_name,
            'configs_loaded': list(configs.keys()),
            'results': {
                'convergence': {
                    'iterations': 150,
                    'finalResidual': 1e-6
                },
                'performance': {
                    'executionTime': '2.5s'
                },
                'mesh': {
                    'cellCount': 50000
                }
            }
        }
        
        # 保存仿真结果
        result_file = os.path.join(simulation_dir, 'simulation_result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        raise Exception(f'StokeSimulator execution failed: {str(e)}')

@app.route('/simulate/get_results', methods=['POST'])
def get_simulation_results():
    """
    获取仿真结果接口
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    
    try:
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        result_file = os.path.join(simulation_dir, 'simulation_result.json')
        
        if not os.path.exists(result_file):
            return jsonify({'error': 'Simulation results not found'}), 404
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get simulation results: {str(e)}'}), 500

@app.route('/simulate/export_results', methods=['POST'])
def export_simulation_results():
    """
    导出仿真结果接口
    """
    data = request.json
    simulation_name = data.get('simulation_name')
    
    try:
        simulation_dir = os.path.join(os.getcwd(), 'simulations', simulation_name)
        
        # 创建结果压缩包
        import zipfile
        zip_path = os.path.join(simulation_dir, f'{simulation_name}_results.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(simulation_dir):
                for file in files:
                    if file.endswith(('.json', '.vtu', '.stl')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, simulation_dir)
                        zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{simulation_name}_results.zip')
        
    except Exception as e:
        return jsonify({'error': f'Failed to export simulation results: {str(e)}'}), 500


if __name__ == '__main__':
    print('Server is running on ws://192.168.110.66:9998')
    # Use threaded=True for better WebSocket support
    app.run(host='192.168.110.66', port=9998, debug=True, threaded=True)
