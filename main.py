import time
from runner_env import RunnerEnv, load_json
from utils import *
import paho.mqtt.client as mqtt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



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
step_timestamp = 0
print_banner()
input_athlete, training_name, circuit_name, mqtt_communication = begin_session()
profile_label = get_profile_label(input_athlete)
circuit = load_json(f"data/maps/{circuit_name}.json")
training = get_training_label(training_name)

print_summary(profile_label, input_athlete, circuit_name, training_name, mqtt_communication)

Q = load_qtable(profile_label, training_name)
env = RunnerEnv(input_athlete, training, track_data=circuit, verbose=True)
state = env.reset()
training_completed = False
total_reward = 0
session_data = []

if mqtt_communication:
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

    if mqtt_communication:
        step_timestamp += 1
        state["timestamp"] = step_timestamp
        send_mqtt(state, action, reward, client, topic)
        time.sleep(1)

save_training_session(session_data, profile_label, training_name, circuit, circuit_name)    
print(f"\n🏁 Training completed! Total reward: {total_reward:.2f}")

