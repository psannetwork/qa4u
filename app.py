# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask backend for drone deployment and supply distribution optimization
All endpoints use GET requests and JSON format.
Client data is stored per client_id to avoid mixing.
Includes error handling to prevent runtime errors.
Serves HTML client (index.html) and static templates directory.
Supports replacing and appending supplies, shelters, and drones.
"""
# Monkey-patch werkzeug for Flask compatibility
import werkzeug
import werkzeug.urls
if not hasattr(werkzeug.urls, 'url_quote'):
    werkzeug.urls.url_quote = werkzeug.urls.quote

from flask import Flask, request, jsonify, send_from_directory
import uuid
import json
import logging
import numpy as np
import pandas as pd
import jijmodeling as jm
from jijmodeling_transpiler.core import compile_model
from jijmodeling_transpiler.core.pubo import transpile_to_pubo
from openjij import SASampler

# Flask app setup
app = Flask(__name__, static_folder=None, template_folder='templates')
logging.basicConfig(level=logging.INFO)

# In-memory storage: client_id -> {'supplies': list, 'shelters': list, 'drones': list}
client_data = {}

# Constants
DRONE_CAPACITY = 10  # kg per drone
MAX_TRIPS = 3
CAP_PER_SHELTER = DRONE_CAPACITY * MAX_TRIPS
W_SCALE = 1.0
P_SCALE = 1e3
INTRANSIT_PEN = 1e3

# Utility: validate JSON list or object
def _parse_json_list(param_str, param_name):
    try:
        data = json.loads(param_str)
        if isinstance(data, dict):
            return [data]
        if not isinstance(data, list):
            raise ValueError(f"{param_name} should be a JSON object or list of objects")
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for {param_name}")

# Serve index and static templates
@app.route('/', methods=['GET'])
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/templates/<path:filename>', methods=['GET'])
def serve_template_file(filename):
    try:
        return send_from_directory('templates', filename)
    except FileNotFoundError:
        return jsonify({'error': f'{filename} not found'}), 404

# Endpoint: Register a new client_id
@app.route('/register_client', methods=['GET'])
def register_client():
    client_id = str(uuid.uuid4())
    client_data[client_id] = {'supplies': [], 'shelters': [], 'drones': []}
    return jsonify({'client_id': client_id}), 200

# Endpoint: Replace supplies list
@app.route('/supplies', methods=['GET'])
def update_supplies():
    cid = request.args.get('client_id')
    data_str = request.args.get('supplies')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing supplies parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'supplies')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['supplies'] = items
    return jsonify({'status': 'supplies replaced', 'count': len(items)}), 200

# Endpoint: Append supplies
@app.route('/add_supplies', methods=['GET'])
def add_supplies():
    cid = request.args.get('client_id')
    data_str = request.args.get('supplies')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing supplies parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'supplies')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['supplies'].extend(items)
    return jsonify({'status': 'supplies appended', 'new_count': len(client_data[cid]['supplies'])}), 200

# Endpoint: Replace shelters list
@app.route('/shelters', methods=['GET'])
def update_shelters():
    cid = request.args.get('client_id')
    data_str = request.args.get('shelters')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing shelters parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'shelters')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['shelters'] = items
    return jsonify({'status': 'shelters replaced', 'count': len(items)}), 200

# Endpoint: Append shelters list
@app.route('/add_shelters', methods=['GET'])
def add_shelters():
    cid = request.args.get('client_id')
    data_str = request.args.get('shelters')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing shelters parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'shelters')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['shelters'].extend(items)
    return jsonify({'status': 'shelters appended', 'new_count': len(client_data[cid]['shelters'])}), 200

# Endpoint: Replace drones list
@app.route('/drones', methods=['GET'])
def update_drones():
    cid = request.args.get('client_id')
    data_str = request.args.get('drones')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing drones parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'drones')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['drones'] = items
    return jsonify({'status': 'drones replaced', 'count': len(items)}), 200

# Endpoint: Append drones
@app.route('/add_drones', methods=['GET'])
def add_drones():
    cid = request.args.get('client_id')
    data_str = request.args.get('drones')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    if not data_str:
        return jsonify({'error': 'Missing drones parameter'}), 400
    try:
        items = _parse_json_list(data_str, 'drones')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    client_data[cid]['drones'].extend(items)
    return jsonify({'status': 'drones appended', 'new_count': len(client_data[cid]['drones'])}), 200

# Retrieval endpoints
@app.route('/get_supplies', methods=['GET'])
def get_supplies():
    cid = request.args.get('client_id')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    return jsonify({'supplies': client_data[cid]['supplies']}), 200

@app.route('/get_shelters', methods=['GET'])
def get_shelters():
    cid = request.args.get('client_id')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    return jsonify({'shelters': client_data[cid]['shelters']}), 200

@app.route('/get_drones', methods=['GET'])
def get_drones():
    cid = request.args.get('client_id')
    if not cid or cid not in client_data:
        return jsonify({'error': 'Invalid or missing client_id'}), 400
    return jsonify({'drones': client_data[cid]['drones']}), 200

# Helper: QUBO-based shelter selection
def select_shelters(supplies, shelters, max_drones):
    J = len(shelters)
    shortages = np.zeros(J, int)
    for j, sh in enumerate(shelters):
        shortages[j] = sum(
            max(sh['demand'].get(sup['name'], 0) - sh['stock'].get(sup['name'], 0), 0)
            for sup in supplies
        )
    need = np.ceil(shortages / DRONE_CAPACITY).astype(int)
    W = np.array([shelters[j]['urgency'] * shortages[j] for j in range(J)], float)
    in_trans = {j for j, sh in enumerate(shelters) if sh.get('in_transit')}

    model = jm.Problem("drone_alloc", sense=jm.ProblemSense.MINIMIZE)
    W_ph = jm.Placeholder("W", ndim=1)
    c_ph = jm.Placeholder("c", ndim=1)
    x = jm.BinaryVar("x", shape=[J])
    i = jm.Element("i", belong_to=(0, J))

    usage = jm.sum(i, c_ph[i] * x[i])
    importance = jm.sum(i, W_ph[i] * x[i])
    penalty = P_SCALE * (usage - max_drones) ** 2
    for j in in_trans:
        penalty += INTRANSIT_PEN * x[j]

    model += -W_SCALE * importance + penalty
    inst = {"W": W.tolist(), "c": need.tolist()}
    compiled = compile_model(model, inst)
    pubo = transpile_to_pubo(compiled)
    qubo, _ = pubo.get_qubo_dict()
    result = SASampler().sample_qubo(qubo, num_reads=200).first.sample

    selected = [j for j in range(J) if result.get(f"x[{j}]", 0) == 1]
    total_need = sum(int(need[j]) for j in selected)
    if total_need != max_drones:
        selected = []
        order = np.argsort(-W)
        cum = 0
        for j in order:
            if j in in_trans:
                continue
            if cum + need[j] <= max_drones:
                selected.append(j)
                cum += need[j]
            if cum == max_drones:
                break
    return selected, need, W

# Helper: Greedy supply assignment
def assign_supplies(supplies, shelters, selected):
    rows = []
    caps = {j: CAP_PER_SHELTER for j in selected}
    leftovers = {}
    for sup in supplies:
        name, weight, cost = sup.get('name'), sup.get('weight'), sup.get('cost')
        rem = sup.get('count', 0)
        order = sorted(selected, key=lambda j: shelters[j]['urgency'], reverse=True)
        # First satisfy demand
        for j in order:
            if rem <= 0:
                break
            shortage = max(
                shelters[j]['demand'].get(name, 0) - shelters[j]['stock'].get(name, 0), 0
            )
            can = min(shortage, rem, caps[j] // weight)
            if can > 0:
                rows.append({
                    '避難所ID': shelters[j]['id'],
                    '物資名': name,
                    '配送個数': int(can),
                    '重量合計': int(can * weight),
                    'コスト': int(can * cost)
                })
                rem -= can
                caps[j] -= can * weight
        # Then use leftover capacity
        for j in order:
            if rem <= 0:
                break
            space = caps[j] // weight
            if space > 0:
                take = min(rem, space)
                rows.append({
                    '避難所ID': shelters[j]['id'],
                    '物資名': name,
                    '配送個数': int(take),
                    '重量合計': int(take * weight),
                    'コスト': int(take * cost)
                })
                rem -= take
                caps[j] -= take * weight
        leftovers[name] = int(rem)
    df = pd.DataFrame(rows)
    total_w = int(df['重量合計'].sum()) if not df.empty else 0
    total_cost = int(df['コスト'].sum()) if not df.empty else 0
    return df, total_w, total_cost, leftovers

# Endpoint: Optimize and return results
@app.route('/optimize', methods=['GET'])
def optimize():
    try:
        cid = request.args.get('client_id')
        if not cid or cid not in client_data:
            return jsonify({'error': 'Invalid or missing client_id'}), 400
        supplies = client_data[cid]['supplies']
        shelters = client_data[cid]['shelters']
        drones = client_data[cid]['drones']
        if not (supplies and shelters and drones):
            return jsonify({'error': 'Missing supplies, shelters, or drones data'}), 400
        available = [d for d in drones if isinstance(d, dict) and d.get('available')]
        max_drones = len(available)
        sel_idx, need_arr, W_arr = select_shelters(supplies, shelters, max_drones)
        sel_ids = [shelters[j]['id'] for j in sel_idx]
        need_map = {shelters[j]['id']: int(need_arr[j]) for j in sel_idx}
        df, total_w, total_cost, leftovers = assign_supplies(supplies, shelters, sel_idx)
        return jsonify({
            'selected_shelters': sel_ids,
            'need_drones': need_map,
            'total_available_drones': max_drones,
            'used_drones': sum(need_map.values()),
            'assignments': df.to_dict(orient='records'),
            'total_weight': total_w,
            'total_cost': total_cost,
            'leftovers': leftovers
        }), 200
    except Exception as e:
        logging.exception("Optimization failed")
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
