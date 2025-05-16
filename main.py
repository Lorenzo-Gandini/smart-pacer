import json
import time
import os
import math
from runner_env import RunnerEnv, load_json

import paho.mqtt.client as mqtt

# ==== INPUT OF THE ATHLET ==== #
input_athlete = {
    "HR_rest": 50,
    "HR_max": 180,
    "FTP": 280,
    "weight_kg": 72,
    "fitness_factor": 1.0
}

training_type = "fartlek"
gpx_track_path = "data/json/track_data.json"

alpha = 0.1
gamma = 0.95
epsilon = 0.01

# ==== Distance from models  ==== #
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[k] - p2[k])**2 for k in p1))

# ==== Load Q-table corrispondente ==== #
def get_profile_label(athlete):
    base_profiles = load_json("data/json/athletes.json")
    distances = {
        label: euclidean_distance(athlete, base_profiles[label])
        for label in base_profiles
    }
    return min(distances, key=distances.get)

def send_mqtt(state, action, client, topic):
    payload = json.dumps({
        "second": state.get("segment_index", 0),
        "phase": state.get("phase_label", "unknown"),
        "fatigue": state.get("fatigue_level", "unknown"),
        "action": action
    })
    client.publish(topic, payload)
    print(f"📤 MQTT messaggio pubblicato: {payload}")


profile_label = get_profile_label(input_athlete)
print("Profilo atleta più vicino caricato : ", profile_label)
q_table_path = f"data/q-table/q_table_{profile_label}_{training_type}_2000ep_{alpha}a_{gamma}g_{epsilon}e.json"

with open(q_table_path) as f:
    raw_q_table = json.load(f)
Q = {eval(k): v for k, v in raw_q_table.items()}

# ==== Preparazione ambiente ====
trainings = load_json("data/json/trainings.json")
track = load_json(gpx_track_path) 
training = trainings[training_type]

actions = ['slow down', 'keep going', 'accelerate']

def get_state_key(state):
    return (
        state['HR_zone'],
        state['power_zone'],
        state['fatigue_level'],
        state['phase_label'],
        state['target_hr_zone'],
        state['target_power_zone'],
        state['slope_level']
    )

env = RunnerEnv(input_athlete, training, track_data=track, verbose=True)
state = env.reset()
done = False
total_reward = 0

# ==== MQTT Setup ====

client = mqtt.Client()
client.connect("broker.emqx.io", 1883, 60)
topic = "smartpacer/action"

# ==== Simulation ====
while not done:
    state_key = get_state_key(state)
    if state_key in Q:
        action = max(Q[state_key], key=Q[state_key].get)
    else:
        action = 'keep going'

    state, reward, done = env.step(action)
    total_reward += reward

    send_mqtt(state, action, client, topic)
    time.sleep(1)

print(f"\n🏁 Reward totale ottenuto: {total_reward:.2f}")
