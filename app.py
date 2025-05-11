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
) -> Tuple[List[int], np.ndarray]:
    num = len(shelters)
    shortages = np.array([sum(max(sh['demand'].get(s['name'],0)-sh['stock'].get(s['name'],0),0) for s in supplies) for sh in shelters],int)
    need = np.ceil(shortages/DRONE_CAPACITY_KG).astype(int)
    weights = np.array([sh['urgency']*shortages[i] for i,sh in enumerate(shelters)])
    in_transit = {i for i,sh in enumerate(shelters) if sh.get('in_transit')}
    model=jm.Problem('sel',sense=jm.ProblemSense.MINIMIZE)
    w_ph=jm.Placeholder('w',ndim=1); c_ph=jm.Placeholder('c',ndim=1)
    x=jm.BinaryVar('x',shape=[num]); idx=jm.Element('i',belong_to=(0,num))
    total=jm.sum(idx,c_ph[idx]*x[idx]); imp=jm.sum(idx,w_ph[idx]*x[idx])
    pen=PENALTY_SCALE*(total-max_drones)**2
    for i in in_transit: pen+=IN_TRANSIT_PENALTY*x[i]
    model+=-WEIGHT_SCALE*imp+pen
    inst={'w':weights.tolist(),'c':need.tolist()}
    qubo, _=transpile_to_pubo(compile_model(model,inst)).get_qubo_dict()
    res=SASampler().sample_qubo(qubo,num_reads=200).first.sample
    sel=[i for i in range(num) if res.get(f'x[{i}]',0)==1]
    if sum(need[i] for i in sel)!=max_drones and max_drones>0:
        sel=[]
        for i in np.argsort(-weights):
            if i in in_transit: continue
            if sum(need[j] for j in sel)+need[i]<=max_drones:
                sel.append(int(i))
                if sum(need[j] for j in sel)==max_drones: break
    return sel, need

def assign_supplies(
    supplies: List[Dict],
    shelters: List[Dict],
    selected: List[int]
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    caps={i:CAPACITY_PER_SHELTER for i in selected}
    records=[]
    for sup in supplies:
        name, w, c, cnt = sup.get('name',''), sup.get('weight',0), sup.get('cost',0), sup.get('count',0)
        rem=cnt
        for phase in ('demand','leftover'):
            for i in sorted(selected, key=lambda j:shelters[j]['urgency'], reverse=True):
                if rem<=0: break
                if phase=='demand': need_amt=max(shelters[i]['demand'].get(name,0)-shelters[i]['stock'].get(name,0),0)
                else: need_amt=caps[i]//w
                if need_amt<=0: continue
                send=min(rem,need_amt,caps[i]//w)
                if send<=0: continue
                records.append({'避難所ID':shelters[i]['id'],'物資名':name,'配送個数':int(send),'重量合計':int(send*w),'コスト':int(send*c)})
                caps[i]-=send*w; rem-=send
    df=pd.DataFrame(records)
    grouped=df.groupby('避難所ID')['重量合計'].sum().to_dict() if not df.empty else {}
    need_map={sid: int(np.ceil(wt/DRONE_CAPACITY_KG)) for sid,wt in grouped.items()}
    return df, need_map

@app.route('/optimize', methods=['GET'])
@require_client
def optimize(cid:str):
    try:
        data=client_data[cid]
        if not all(data[key] for key in ('supplies','shelters','drones')):
            return jsonify({'error':'Incomplete data'}),400
        max_dr=sum(1 for d in data['drones'] if d.get('available'))
        sel,need_arr=select_shelters(data['supplies'],data['shelters'],max_dr)
        df,need_map=assign_supplies(data['supplies'],data['shelters'],sel)
        return jsonify({
            'selected_shelters':[data['shelters'][i]['id'] for i in sel],
            'need_drones':need_map,
            'total_available_drones':max_dr,
            'used_drones':sum(need_map.values()),
            'assignments':df.to_dict(orient='records'),
            'total_weight':int(df['重量合計'].sum()) if not df.empty else 0,
            'total_cost':int(df['コスト'].sum()) if not df.empty else 0,
            'leftovers':{sup['name']:sup['count']-sum(rec['配送個数'] for rec in df.to_dict('records') if rec['物資名']==sup['name']) for sup in data['supplies']}
        }),200
    except Exception as e:
        logging.exception('Optimization error')
        return jsonify({'error':'Internal server error','detail':str(e)}),500

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
