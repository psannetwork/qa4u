#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask backend for drone deployment and supply distribution optimization.
Data persisted to .temp.txt to survive restarts.
All endpoints use GET requests and JSON format.
Client data is stored per client_id to avoid mixing.
Includes error handling to prevent runtime errors.
Serves HTML client (index.html) and static templates directory.
Supports replacing and appending supplies, shelters, and drones.
"""
import uuid
import json
import logging
import os
from typing import List, Dict, Tuple

from flask import Flask, request, jsonify, send_from_directory, abort
import numpy as np
import pandas as pd
import jijmodeling as jm
from jijmodeling_transpiler.core import compile_model
from jijmodeling_transpiler.core.pubo import transpile_to_pubo
from openjij import SASampler

# -- Configuration -----------------------------------------------------------
PERSIST_FILE = '.temp.txt'
DRONE_CAPACITY_KG = 10
MAX_TRIPS_PER_SHELTER = 3
CAPACITY_PER_SHELTER = DRONE_CAPACITY_KG * MAX_TRIPS_PER_SHELTER
WEIGHT_SCALE = 1.0
PENALTY_SCALE = 1e3
IN_TRANSIT_PENALTY = 1e3

# -- Flask App Setup ----------------------------------------------------------
app = Flask(__name__, static_folder=None, template_folder='templates')
logging.basicConfig(level=logging.INFO)

# In-memory storage: client_id -> data dict
client_data: Dict[str, Dict[str, List[Dict]]] = {}

# -- Persistence -------------------------------------------------------------
def load_data():
    global client_data
    if os.path.exists(PERSIST_FILE):
        try:
            with open(PERSIST_FILE, 'r', encoding='utf-8') as f:
                client_data = json.load(f)
        except Exception:
            logging.exception('Failed to load persistence file')

def save_data():
    try:
        with open(PERSIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(client_data, f)
    except Exception:
        logging.exception('Failed to save persistence file')

load_data()

# -- Utility Functions -------------------------------------------------------
def parse_json_list(param_str: str, param_name: str) -> List[Dict]:
    try:
        data = json.loads(param_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for {param_name}")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"{param_name} should be an object or list of objects")

def require_client(func):
    def wrapper(*args, **kwargs):
        cid = request.args.get('client_id', '').strip()
        if not cid or cid not in client_data:
            return jsonify({'error': 'Invalid or missing client_id'}), 400
        return func(cid, *args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# -- Routes ------------------------------------------------------------------
@app.route('/', methods=['GET'])
@app.route('/index.html', methods=['GET'])
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/templates/<path:filename>', methods=['GET'])
def serve_static(filename: str):
    try:
        return send_from_directory('templates', filename)
    except FileNotFoundError:
        abort(404, description=f"{filename} not found")

@app.route('/register_client', methods=['GET'])
def register_client():
    cid = str(uuid.uuid4())
    client_data[cid] = {'supplies': [], 'shelters': [], 'drones': []}
    save_data()
    return jsonify({'client_id': cid}), 200

@app.route('/supplies', methods=['GET'])
@require_client
def update_supplies(cid: str):
    data_str = request.args.get('supplies', '')
    if not data_str:
        return jsonify({'error': 'Missing supplies parameter'}), 400
    try:
        items = parse_json_list(data_str, 'supplies')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['supplies'] = items
    save_data()
    return jsonify({'status': 'supplies replaced', 'count': len(items)}), 200

@app.route('/add_supplies', methods=['GET'])
@require_client
def add_supplies(cid: str):
    data_str = request.args.get('supplies', '')
    if not data_str:
        return jsonify({'error': 'Missing supplies parameter'}), 400
    try:
        items = parse_json_list(data_str, 'supplies')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['supplies'].extend(items)
    save_data()
    return jsonify({'status': 'supplies appended', 'new_count': len(client_data[cid]['supplies'])}), 200

@app.route('/get_supplies', methods=['GET'])
@require_client
def get_supplies(cid: str):
    return jsonify({'supplies': client_data[cid]['supplies']}), 200

@app.route('/shelters', methods=['GET'])
@require_client
def update_shelters(cid: str):
    data_str = request.args.get('shelters', '')
    if not data_str:
        return jsonify({'error': 'Missing shelters parameter'}), 400
    try:
        items = parse_json_list(data_str, 'shelters')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['shelters'] = items
    save_data()
    return jsonify({'status': 'shelters replaced', 'count': len(items)}), 200

@app.route('/add_shelters', methods=['GET'])
@require_client
def add_shelters(cid: str):
    data_str = request.args.get('shelters', '')
    if not data_str:
        return jsonify({'error': 'Missing shelters parameter'}), 400
    try:
        items = parse_json_list(data_str, 'shelters')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['shelters'].extend(items)
    save_data()
    return jsonify({'status': 'shelters appended', 'new_count': len(client_data[cid]['shelters'])}), 200

@app.route('/get_shelters', methods=['GET'])
@require_client
def get_shelters(cid: str):
    return jsonify({'shelters': client_data[cid]['shelters']}), 200

@app.route('/drones', methods=['GET'])
@require_client
def update_drones(cid: str):
    data_str = request.args.get('drones', '')
    if not data_str:
        return jsonify({'error': 'Missing drones parameter'}), 400
    try:
        items = parse_json_list(data_str, 'drones')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['drones'] = items
    save_data()
    return jsonify({'status': 'drones replaced', 'count': len(items)}), 200

@app.route('/add_drones', methods=['GET'])
@require_client
def add_drones(cid: str):
    data_str = request.args.get('drones', '')
    if not data_str:
        return jsonify({'error': 'Missing drones parameter'}), 400
    try:
        items = parse_json_list(data_str, 'drones')
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    client_data[cid]['drones'].extend(items)
    save_data()
    return jsonify({'status': 'drones appended', 'new_count': len(client_data[cid]['drones'])}), 200

@app.route('/get_drones', methods=['GET'])
@require_client
def get_drones(cid: str):
    return jsonify({'drones': client_data[cid]['drones']}), 200

# -- Optimization ------------------------------------------------------------
def select_shelters(
    supplies: List[Dict],
    shelters: List[Dict],
    max_drones: int
) -> Tuple[List[str], np.ndarray]:
    num = len(shelters)
    if num == 0:
        return [], np.array([])

    shortages = np.array([
        sum(max(sh['demand'].get(s['name'], 0) - sh['stock'].get(s['name'], 0), 0) for s in supplies)
        for sh in shelters
    ], dtype=int)
    need = np.ceil(shortages / DRONE_CAPACITY_KG).astype(int)
    weights = np.array([sh['urgency'] * shortages[i] for i, sh in enumerate(shelters)])
    in_transit = {i for i, sh in enumerate(shelters) if sh.get('in_transit')}

    # Ensure we have enough drones to cover the minimum needs
    total_need = np.sum(need)
    if total_need == 0:
        return [], np.array([])

    # Prioritize shelters with higher urgency and greater shortage
    sorted_indices = np.argsort(-weights)
    sel = []
    current_drones_used = 0

    for i in sorted_indices:
        if current_drones_used + need[i] > max_drones:
            continue
        sel.append(shelters[i]['id'])
        current_drones_used += need[i]
        if current_drones_used == max_drones:
            break

    return sel, need

def assign_supplies(
    supplies: List[Dict],
    shelters: List[Dict],
    selected: List[str],
    max_drones: int
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    shelters_dict = {sh['id']: sh for sh in shelters}
    caps = {sh_id: CAPACITY_PER_SHELTER for sh_id in selected}
    records = []
    leftovers = {sup['name']: sup['count'] for sup in supplies}
    total_used_drones = 0

    for sup in supplies:
        name, w, c, cnt = sup.get('name', ''), sup.get('weight', 0), sup.get('cost', 0), sup.get('count', 0)
        rem = cnt
        for sh_id in selected:
            if total_used_drones >= max_drones:
                break
            need_amt = max(shelters_dict[sh_id]['demand'].get(name, 0) - shelters_dict[sh_id]['stock'].get(name, 0), 0)
            if need_amt <= 0:
                continue
            send = min(rem, need_amt, caps[sh_id] // w)
            if send <= 0:
                continue
            records.append({
                '避難所ID': sh_id,
                '物資名': name,
                '配送個数': int(send),
                '重量合計': int(send * w),
                'コスト': int(send * c)
            })
            caps[sh_id] -= send * w
            rem -= send
            leftovers[name] -= send
            total_used_drones += send // DRONE_CAPACITY_KG + (1 if send % DRONE_CAPACITY_KG != 0 else 0)

    df = pd.DataFrame(records)
    grouped = df.groupby('避難所ID')['重量合計'].sum().to_dict() if not df.empty else {}
    need_map = {sid: int(np.ceil(wt / DRONE_CAPACITY_KG)) for sid, wt in grouped.items()}

    # Ensure the number of selected shelters does not exceed the number of available drones
    actual_used_drones = sum(need_map.values())

    # Adjust need_map to ensure it doesn't exceed available drones
    while actual_used_drones > max_drones:
        # Find the shelter with the lowest urgency that uses the most drones
        max_drones_shelter = max(need_map.items(), key=lambda item: (item[1], -shelters_dict[item[0]].get('urgency', 0)))
        need_map[max_drones_shelter[0]] -= 1
        actual_used_drones -= 1

    return df, need_map, leftovers

@app.route('/optimize', methods=['GET'])
@require_client
def optimize(cid: str):
    try:
        data = client_data[cid]
        if not all(data[key] for key in ('supplies', 'shelters', 'drones')):
            return jsonify({'error': 'Incomplete data'}), 400
        
        max_dr = sum(1 for d in data['drones'] if d.get('available'))
        if max_dr == 0:
            return jsonify({'error': 'No available drones'}), 400
        
        sel, need_arr = select_shelters(data['supplies'], data['shelters'], max_dr)
        df, need_map, leftovers = assign_supplies(data['supplies'], data['shelters'], sel, max_dr)
        
        if df.empty:
            logging.info(f"No assignments could be made for client {cid}. Selected shelters: {sel}, Need map: {need_map}, Max drones: {max_dr}")
            return jsonify({'error': 'No assignments could be made'}), 400
        
        return jsonify({
            'selected_shelters': sel,
            'need_drones': need_map,
            'total_available_drones': max_dr,
            'used_drones': sum(need_map.values()),
            'assignments': df.to_dict(orient='records'),
            'total_weight': int(df['重量合計'].sum()) if not df.empty else 0,
            'total_cost': int(df['コスト'].sum()) if not df.empty else 0,
            'leftovers': leftovers
        }), 200
    except Exception as e:
        logging.exception('Optimization error')
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
