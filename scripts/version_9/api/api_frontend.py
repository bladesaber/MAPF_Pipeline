"""

author qianyuzhang (@SinoDynamics)
23/10/13
Communicate with html to generate requests and call api_backend to handle the corresponding request
"""

from flask import Flask, request, jsonify, send_file
import api_backend
import json

app = Flask(__name__)
# backend = api_backend.MAPF_Pipeline_Backend()
# Test the api
@app.route('/')
def Hello_SinoDynamics():
    return 'Hello_SinoDynamics'

@app.route('/save_cfg_files', methods =['POST'])
def save_files():
    if 'file' not in request.files['file']:
        return jsonify({'error':'No files provoided'})

    # file = request.files['file']

    try:
        data = request.files.getlists('json_files')

        for json_file in data:
            # json_data = json_file.read().decode('utf-8')
            json_obj = json.loads(json_file.read().decode('utf-8'))
            project_id = json_obj['proj_id']
            json_type = json_obj['json_obj_type']
            if json_type == 'Save_Env_Config':
                api_backend.saveEnvJson(project_id, json_obj['data'])
                return jsonify({'message': 'Environment Configuration saved successfully.'})
            elif json_type == 'Save_PipeLinks_Config':
                api_backend.generatePipesLinkConfigJson(project_id, json_obj['data'])
                return jsonify({'message': 'Pipes Link Configuration saved successfully.'})
            elif json_type == 'Save_OptSettings':
                api_backend.generateOptSettingsJson(project_id, json_obj['data'])
                return jsonify({'message': 'Optimization Settings saved successfully.'})
    except:
        return jsonify({'error':'files saving error'})

# @app.route('/generate_pipeslink_config', methods =['POST'])
# def generate_pipeslink_config():
#     api_backend.generatePipesLinkConfigJson()
#     return jsonify({'message':'Pipes Link Configuration generated successfully.'})
#
# @app.route('/generate_OptSetting', methods =['POST'])
# def generate_OptSetting():
#     api_backend.generateOptSettingJson()
#     return jsonify({'message':'Optimization Settings generated successfully.'})

@app.route('/load_files', methods =['GET'])
def load_files():
    try:
        data = request.files.getlists('json_files') # get the loadfile names

        for json_file in data:
            # json_data = json_file.read().decode('utf-8')
            json_obj = json.loads(json_file.read().decode('utf-8'))
            project_id = json_obj['proj_id']
            json_type = json_obj['json_obj_type']
            if json_type == 'load_Env_Config':
                env_config = api_backend.loadJson(project_id, json_type)
                info = {'project_id':project_id, 'env_cfg': env_config}
                return jsonify(info)
            elif json_type == 'load_PipeLinks_Config':
                pipeLinks_setting = api_backend.loadJson(project_id, json_type)
                info = {'project_id': project_id, 'PipeLinks_Config': pipeLinks_setting}
                return jsonify(info)
            elif json_type == 'load_OptSettings':
                Optimization_Settings = api_backend.loadJson(project_id, json_type)
                info = {'project_id': project_id, 'OptSettings': Optimization_Settings}
                return jsonify(info)
            elif json_type == 'load_Grid_Env_Config':
                grid_env_config = api_backend.loadJson(project_id, json_type)
                info = {'project_id': project_id, 'Grid_Env_Config': grid_env_config}
                return jsonify(info)
    except:
        return jsonify({'error':'files loading error'})

@app.route('/generate_grid_env_cfg', methods=['GET'])
def generate_grid_env_cfg():
    project_id = request.form.get('project_id')
    grid_env_config = api_backend.discretize_env(project_id)
    return jsonify({'grid_env_config': grid_env_config})

@app.route('/search_path', methods=['GET'])
def search_path():
    try:
        project_id = request.form.get('project_id')
        search_path = api_backend.search_path(project_id)
        info = {'project_id': project_id, 'discrete_path': search_path}
        return jsonify(info)
    except:
        return jsonify({'error':'paths searched failed'})

@app.route('/optimize_path', methods=['GET'])
def optimize_path():
    try:
        project_id = request.form.get('project_id')
        optimized_path = api_backend.optimize_path(project_id)
        info = {'project_id': project_id, 'smooth_path': optimized_path}
        return jsonify(info)
    except:
        return jsonify({'error': 'paths smoothed failed'})

if __name__ == '__main__':
    app.run(debug=True)