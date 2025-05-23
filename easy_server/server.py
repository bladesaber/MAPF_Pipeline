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

from cad_server_utils import CadServerUtils
from cad_server_utils import GmshStepReader
from scripts_py.version_9.simulator.su2_simulator import Su2Simulator
from server_utils import StreamDataResponse, StreamTaskResponse

from response_utils import StreamThreadTaskResponse, StreamProcessTaskResponse
from path_server_utils import PathServerUtils, ProjectInfo

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
                chunk_size=1000, time_interval=0.5)
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


@app.route('/path_search/clear_cache', methods=['POST'])
def clear_cache():
    data = request.json
    project_ids = data.get('project_ids')
    for project_id in project_ids:
        if project_id in PathServerUtils.project_map:
            del PathServerUtils.project_map[project_id]
    return jsonify({'is_success': True})


@app.route('/path_search/upload_obstacle_pcd', methods=['POST'])
def upload_obstacle_pcd():
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
        project_info.obstacle_pcd = []
        project_info.x_size = x_size + 1  # be careful with the x_size
        project_info.y_size = y_size + 1  # be careful with the y_size
        project_info.z_size = z_size + 1  # be careful with the z_size
        project_info.resolution = resolution
        PathServerUtils.project_map[project_id] = project_info
    
    PathServerUtils.project_map[project_id].obstacle_pcd.extend(obstacle_pcd)

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
    
    if len(PathServerUtils.project_map[project_id].obstacle_pcd) == 0:
        return jsonify({'error': 'No obstacle pcd provided'}), 400

    project_info = PathServerUtils.project_map[project_id]
    obstacle_pcd = np.array(project_info.obstacle_pcd)

    project_info.inner_xyzd = PathServerUtils.compute_inner_xyzd(
        x_size=project_info.x_size, 
        y_size=project_info.y_size, 
        z_size=project_info.z_size, 
        resolution=project_info.resolution,
        obstacle_pcd=obstacle_pcd
    )
    project_info.obstacle_pcd.clear()
    PathServerUtils.init_env(project_id, pipe_data)

    def generate():
        try:
            data = project_info.inner_xyzd

            def get_data_size_callback():
                return len(data)
                    
            def get_chunk_data_callback(data_index, chunk_size):
                return data[data_index:data_index + chunk_size].tolist()
                        
            stream_data_response = StreamDataResponse(
                name='request_inner_xyzd', 
                get_data_size_callback=get_data_size_callback, 
                get_chunk_data_callback=get_chunk_data_callback, 
                chunk_size=5000, time_interval=0.2
            )
            yield from stream_data_response.generate()

            time.sleep(0.5)
            yield json.dumps({'total_finished': True})

        except Exception as e:
            print(f'[Error]: {str(e)}')
            yield json.dumps({'error': str(e)})
            return
                
    return Response(generate(), mimetype='text/event-stream')


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
                    time.sleep(5.0)
                    result = PathServerUtils.run_cell_task(project_id, task_name, priority_name, group_name, cell_name)
                    data_queue.put({
                        'is_finished': True,
                        'result': result
                    })
                
                task = StreamThreadTaskResponse(
                    name='run_cell_task', 
                    callback_func=task_callback, 
                    interval_time=0.2,
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
                
                task = StreamThreadTaskResponse(
                    name='run_group_task', 
                    callback_func=task_callback, 
                    interval_time=0.2,
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
                    interval_time=0.2,
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
                print('result:', result)
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

'''
@sock.route('/path_search/run')
def path_search_run(ws):
    socket_manager = SocketManager(ws)
    print(f'[Debug]: New WebSocket connection {socket_manager.id} established')
    runner = PathSearchRunner()
    is_loading = False
    
    try:
        while True:
            msg_str = ws.receive()
            if msg_str is None:
                break
            msg: dict = json.loads(msg_str)

            if msg['run_type'] == 'HandShake':
                return_msg = {
                    'is_finished': True,
                    'msg': 'HandShake success.'
                }
            
            elif msg['run_type'] == 'prepareGrid':
                obstacle_points = np.array(msg['obstacle_points_data'])

                callback_func = partial(
                    runner.init_grid, 
                    x_size=msg['x_size'], y_size=msg['y_size'], z_size=msg['z_size'], 
                    resolution=msg['resolution'], obstacle_points=obstacle_points
                )
                task0 = SocketTaskResponse(
                    manager=socket_manager, 
                    run_type=msg['run_type'], 
                    name='init_grid',
                    callback_func=callback_func, 
                    interval_time=0.5
                )
                task0.start()
                runner.grid = task0.last_result['grid']
                validate_data = task0.last_result['validate_data']
                
                def get_data_size_callback():
                    return len(validate_data)
                
                def get_chunk_data_callback(data_index, chunk_size):
                    return validate_data[data_index:data_index + chunk_size]
                
                data0 = SocketDataResponse(
                    manager=socket_manager, 
                    run_type=msg['run_type'], 
                    name='return_validate_data', 
                    get_data_size_callback=get_data_size_callback, 
                    get_chunk_data_callback=get_chunk_data_callback, 
                    chunk_size=5000, time_interval=0.5)
                data0.generate()
                
                return_msg = data0.get_progress(
                    data=[], chunk_index=0, is_finished=True, progress=100, msg='Grid prepared successfully.'
                )
                return_msg.update({
                    'total_finished': True
                })

            elif msg['run_type'] == 'searchPreprocess':
                data = msg['data']
                runner.preprocess(data)
                return_msg = {
                    'run_type': msg['run_type'],
                    'is_finished': True,
                    'msg': 'Preprocess success.'
                }
            
            elif msg['run_type'] == 'runPrioritySearch':
                task_set = msg['data']['task_set']
                group_pipes = msg['data']['group_pipes']
                
                callback_func = partial(
                    runner.run_priority_task,
                    grid=runner.grid,
                    task_set=task_set,
                    group_pipes=group_pipes,
                    running_cache=runner.running_cache,
                    running_graph=runner.running_graph
                )

                task1 = SocketTaskResponse(
                    manager=socket_manager, 
                    run_type=msg['run_type'], name='run_priority_search', callback_func=callback_func, interval_time=0.5)
                task1.start()

                time.sleep(0.5)
                last_result = task1.last_result
                return_msg = task1.get_progress(
                    is_finished=True,
                    progress=100,
                    msg='Run priority search success.' if last_result['is_success']
                        else 'Run priority search failed.',
                )
                return_msg.update({
                    'total_finished': True,
                    'data': {
                        'is_success': last_result['is_success'],
                        'path_set_group': last_result['path_set_group'],
                    }
                })

            else:
                return_msg = {
                    'is_finished': True,
                    'msg': 'Non known run type.'
                }
                
            # Send response
            ws.send(json.dumps(return_msg))

    except json.JSONDecodeError:
        ws.send(json.dumps({'error': 'Invalid JSON format'}))

    except Exception as e:
        print(f"[Error]: Error processing message: {e}")
        ws.send(json.dumps({'error': str(e)}))
                
    finally:
        SocketManager.remove_socket(socket_manager)
        print(f"[Debug]: Connection {socket_manager.id} closed")
'''

if __name__ == '__main__':
    print('Server is running on ws://192.168.110.54:9998')
    # Use threaded=True for better WebSocket support
    app.run(host='192.168.110.54', port=9998, debug=True, threaded=True)
