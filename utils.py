import os
import json
import math
import csv
from math import radians, cos, sin, asin, sqrt
from runner_env import RunnerEnv, load_json, ACTIONS
import datetime


def ask_circuit():
    track_data = []
    circuit_map = {}

    for filename in os.listdir("data/maps"):
        if filename.endswith(".json"):
            track_data.append(filename[:-5])

    print("\n🏃‍♂️ Choose where you want to run:")
    for i, track in enumerate(track_data, start=1):
        print(f"{i}. {track}")
        circuit_map[str(i)] = track

    circuit = circuit_map.get(input(f"Enter the number of the circuit [1 - {len(track_data) }]: ").strip(), "acquedotti")  

    return circuit  

def ask_training():
    training_path = "data/trainings.json"
    training_data = load_json(training_path)
    training_map = {}

    print("\n🏋️ Choose your training session:")
    for i, training in enumerate(training_data, start=1):
        print(f"{i}. {training}")
        training_map[str(i)] = training

    training_choice = training_map.get(input(f"Enter the number of the training type [1 - {len(training_data)}]: ").strip(), "endurance")  
    return training_choice

def ask_mqtt():
    print("\n📡 Do you want to send MQTT messages during the training session? (y/n)")
    choice = input("Enter your choice: ").strip().lower()
    if choice in ['y', 'yes']:
        print("✅ MQTT communication enabled.")
        return True
    else:
        print("❌ MQTT communication disabled.")
        return False

def send_mqtt(state, action, client, topic):
    payload = json.dumps({
        "second": state.get("segment_index", 0),
        "phase": state.get("phase_label", "unknown"),
        "fatigue": state.get("fatigue_level", "unknown"),
        "action": action
    })
    client.publish(topic, payload)
    print(f"📤 MQTT message : {payload}")

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[k] - p2[k])**2 for k in p1))

def get_profile_label(athlete):
    '''Load the correct q-table based on the athlete and training profile'''
    base_profiles = load_json("data/athletes.json")
    distances = {
        label: euclidean_distance(athlete, base_profiles[label])
        for label in base_profiles
    }
    return min(distances, key=distances.get)

def get_training_label(training_name):
    '''Load the correct q-table based on the training type'''
    trainings = load_json("data/trainings.json")
    training = trainings[training_name]
    return training

def load_qtable(profile_label, training_type):
    q_table_path = f"data/q-table/q_table_{profile_label}_{training_type}.json"
    with open(q_table_path) as f:
        raw_q_table = json.load(f)
    Q = {eval(k): v for k, v in raw_q_table.items()}

    return Q

def get_state_key(state):
    return (
        state['HR_zone'],
        state['power_zone'],
        state['fatigue_level'],
        state['phase_label'],
        state['target_hr_zone'],
        state['target_power_zone'],
        state['slope_level']    )


def save_training_session(session_data, athlete_profile, training_type, track_data, circuit_name):
    '''Save the training session data to a CSV file'''
    os.makedirs("training_logs", exist_ok=True)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"training_logs/{athlete_profile}_{training_type}_{circuit_name}_{today_date}.csv"
    
    fieldnames = [
        'second', 
        'phase', 
        'fatigue', 
        'action', 
        'HR_zone', 
        'power_zone', 
        'target_HR', 
        'target_power',
        'fatigue_score',
        'fatigue_level',
        'slope', 
        'reward', 
        'lat', 
        'lon', 
        'elevation'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, data in enumerate(session_data):
            if i < len(track_data):
                data.update({
                    'lat': track_data[i].get('lat'),
                    'lon': track_data[i].get('lon'),
                    'elevation': track_data[i].get('elevation', None)
                })
            writer.writerow(data)
    
    print(f"\n📊 Data saved in: {filename}")
    return filename


def haversine(lat1, lon1, lat2, lon2):
    '''Calculate the great-circle distance between two points on the Earth specified in decimal degrees'''
    R = 6371000  #Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2 #calculate the square of half the chord length between the points
    c = 2 * asin(sqrt(a))
    return R * c

def slope_level(delta_elev):
    ''' Labels the slope based on elevation change '''
    if delta_elev > 0.5:
        return "uphill"
    elif delta_elev < -0.5:
        return "downhill"
    else:
        return "flat"
    
def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 File saved in '{output_path}'")
