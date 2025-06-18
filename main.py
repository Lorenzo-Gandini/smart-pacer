import time
from runner_env import RunnerEnv, load_json
from utils import *

import paho.mqtt.client as mqtt

# Start the script, asking the users for input parameters. These will be used to configure the training session looking for the closest athlete profile.
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
    fitness = ask_float("Fitness factor (0.7=elite, 1.0=runner, 1.3=amateur)")

    input_athlete = {
        "HR_rest": hr_rest,
        "HR_max": hr_max,
        "FTP": ftp,
        "weight_kg": weight,
        "fitness_factor": fitness
    }

    circuit = ask_circuit()
    training_name = ask_training()
    mqtt_communication = ask_mqtt()

    print(f"\n✅ Configuration complete! Now let's train !\n")
    return input_athlete, training_name, circuit, mqtt_communication

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
input_athlete, training_name, circuit_name, mqtt_communication = begin_session()
profile_label = get_profile_label(input_athlete)
circuit = load_json(f"data/maps/{circuit_name}.json")
training = get_training_label(training_name)

print("Nearest profile : ", profile_label)

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
        send_mqtt(state, action, client, topic)
        time.sleep(0.5)

save_training_session(session_data, profile_label, training_name, circuit, circuit_name)    
print(f"\n🏁 Training completed! Total reward: {total_reward:.2f}")

