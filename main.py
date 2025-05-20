import json
import time
import os
import math
import csv
from runner_env import RunnerEnv, load_json, ACTIONS
from datetime import datetime

import paho.mqtt.client as mqtt


# ==== INITIAL SETTINGS ==== #

MQTT_COMMUNICATION = True

# ==== FUNCTIONS ==== #
def begin_session():
    print("\n👤 Let's configure your session.")
    
    def ask_float(prompt):
        value = input(f"{prompt}: ")
        if value != "":
            try:
                value = float(value)
                return value
            except ValueError:
                print("⚠️ Invalid input. Please enter a number.")
                return ask_float(prompt)

    hr_rest = ask_float("Resting heart rate (HR_rest)")
    hr_max = ask_float("Maximum heart rate (HR_max)")
    ftp = ask_float("Functional Threshold Power (FTP)")
    weight = ask_float("Body weight in kg")
    fitness = ask_float("Fitness factor (0.7=elite, 1.0=runner, 1.3=amatour)")

    input_athlete = {
        "HR_rest": hr_rest,
        "HR_max": hr_max,
        "FTP": ftp,
        "weight_kg": weight,
        "fitness_factor": fitness
    }

    print("\n🏃‍♂️ Choose where you want to run:")
    print("1. Mantova - Belfiore")
    print("2. Mantova - Giro dei 3 laghi")
    print("3. Roma - Parco degli Acquedotti")
    
    circuit_map = {
        "1": "belfiore",
        "2": "3laghi",
        "3": "acquedotti"
    }
    circuit_choice = input("Enter the number of the circuit [1-3]: ").strip()
    circuit = circuit_map.get(circuit_choice, "acquedotti")  

    print("\n🏋️ Choose your training session:")
    print("1. Fartlek")
    print("2. Progressions")
    print("3. Endurance")
    print("4. Recovery")

    training_map = {
        "1": "fartlek",
        "2": "progressions",
        "3": "endurance",
        "4": "recovery"
    }

    choice = input("Enter the number of the training type [1-4]: ").strip()
    training_name = training_map.get(choice, "fartlek")  # default fallback

    print(f"\n✅ Configuration complete! Now let's train !\n")
    return input_athlete, training_name, circuit

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
    base_profiles = load_json("data/json/athletes.json")
    distances = {
        label: euclidean_distance(athlete, base_profiles[label])
        for label in base_profiles
    }
    return min(distances, key=distances.get)

def get_training_label(training_name):
    '''Load the correct q-table based on the training type'''
    trainings = load_json("data/json/trainings.json")
    training = trainings[training_name]
    return training

def save_training_session(session_data, athlete_profile, training_type, track_data):
    '''Save the training session data to a CSV file'''
    os.makedirs("training_logs", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"training_logs/{athlete_profile}_{training_type}_{timestamp}.csv"
    
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

def load_qtable(profile_label, training_type):
    q_table_path = f"data/q-table/q_table_{profile_label}_{training_type}_2000ep_0.01a_0.95g_0.01e_v2.json"
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

def append_data(state, action, reward):
    '''Append data to the session data list'''
    session_data.append({
        'second': state['segment_index'],
        'phase': state['phase_label'],
        'fatigue': state['fatigue_level'],
        'action': action,
        'HR_zone': state['HR_zone'],
        'power_zone': state['power_zone'],
        'target_HR': state['target_hr_zone'],
        'target_power': state['target_power_zone'],
        'slope': state['slope_level'],
        'reward': reward,
        'fatigue_score': env.fatigue_score,
    })

# ==== MAIN SIMULATION ==== #

input_athlete, training_name, circuit = begin_session()
profile_label = get_profile_label(input_athlete)
print("Nearest profile : ", profile_label)

circuit = load_json(f"data/json/{circuit}.json")
training = get_training_label(training_name)

Q = load_qtable(profile_label, training)
env = RunnerEnv(input_athlete, training, track_data=circuit, verbose=True)
state = env.reset()
training_completed = False
total_reward = 0
session_data = []

if MQTT_COMMUNICATION:
    client = mqtt.Client()
    client.connect("broker.emqx.io", 1883, 60)
    topic = "smartpacer/action"

while not training_completed:
    state_key = get_state_key(state)
    if state_key in Q:
        action = max(Q[state_key], key=Q[state_key].get)
    else:
        action = 'keep going'

    state, reward, training_completed = env.step(action)
    total_reward += reward

    append_data(state, action, reward)

    if MQTT_COMMUNICATION:
        send_mqtt(state, action, client, topic)
        time.sleep(0.5)


save_training_session(session_data, profile_label, training_type, track)    
print(f"\n🏁 Training completed! Total reward: {total_reward:.2f}")

