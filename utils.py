import os
import json
import math
import csv
from math import radians, cos, sin, asin, sqrt
from runner_env import load_json
import datetime
from pyfiglet import Figlet

def print_banner():
    f = Figlet(font='puffy')  # oppure 'standard', 'doom', '3d', ecc.
    print("════════════════════════════════════════════════════════════════════════")
    print(f.renderText('Smart Pacer'))             
    print("════════════════════════════════════════════════════════════════════════")

def begin_session():
    print("\n📋📏 Let's define your athlete profile ! \n")

    def ask_float(prompt, icon=""):
        value = input(f"{icon} {prompt}: ").strip()
        if value != "":
            try:
                return float(value)
            except ValueError:
                print("⚠️ Invalid input. Please enter a number.")
                return ask_float(prompt, icon)
        return ask_float(prompt, icon)

    hr_rest = ask_float(" Resting HR ", "❤️ 😴")
    hr_max = ask_float(" Max HR ", "❤️ 🥵")
    ftp = ask_float(" FTP (Functional Threshold Power) ", "💥💪")
    weight = ask_float(" Weight (kg) ", "⚖️ 🪶 ")

    input_athlete = {
        "HR_rest": hr_rest,
        "HR_max": hr_max,
        "FTP": ftp,
        "weight_kg": weight
    }

    circuit = ask_circuit()
    training_name = ask_training()
    mqtt_communication = ask_mqtt()

    return input_athlete, training_name, circuit, mqtt_communication
  
def ask_circuit():
    track_data = []
    circuit_map = {}

    for filename in os.listdir("data/maps"):
        if filename.endswith(".json"):
            track_data.append(filename[:-5])

    print("\n  🌍  Select your training circuit:")
    print("  " + "   ".join([f"{i+1}.  {track}" for i, track in enumerate(track_data)]))

    choice = input(f"  🖊   Circuit [1–{len(track_data)}]: ").strip()
    selected = circuit_map.get(choice, track_data[int(choice)-1] if choice.isdigit() and int(choice) <= len(track_data) else "Parco acquedotti (Roma)")
    print(f"  ✔   {selected} selected\n")
    return selected

 

def ask_training():
    training_path = "data/trainings.json"
    training_data = load_json(training_path)
    training_map = {}

    training_names = list(training_data.keys())
    print("  🏋️   Choose your training session:")
    print("  " + "   ".join([f"{i+1}.  {t.capitalize()}" for i, t in enumerate(training_names)]))

    choice = input(f"  🖊   Training [1–{len(training_names)}]: ").strip()
    selected = training_map.get(choice, training_names[int(choice)-1] if choice.isdigit() and int(choice) <= len(training_names) else "endurance")
    print(f"  ✔   {selected.capitalize()} selected\n")
    return selected

def ask_mqtt():
    choice = input("  📡  Enable MQTT communication? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        print("  ✔   MQTT enabled")
        print("  📲  Run `python mqtt.py` in another terminal to monitor live pacing.\n")
        return True
    else:
        print("  ❌  MQTT disabled\n")
        return False

def send_mqtt(state, action, reward, client, topic):
    payload = json.dumps({
        "second": state.get("segment_index", 0),
        "phase": state.get("phase_label", "unknown"),
        "fatigue": state.get("fatigue_level", "unknown"),
        "action": action,
        "hr_zone": state.get("HR_zone", "?"),
        "power_zone": state.get("power_zone", "?"),
        "reward": reward,
        "timestamp": state["timestamp"],
        "slope": state.get("slope_level", "?")
    })
    client.publish(topic, payload)

def print_summary(profile_name, athlete, circuit, training_type, mqtt_enabled):
    print("\n" + "═" * 50)
    print("✅  CONFIGURATION COMPLETE – SMART PACER READY")
    print("═" * 50)

    print(f"\n🧾  Session Setup")
    print(f"   • Profile        : {profile_name.capitalize()}")
    print(f"   • Circuit        : {circuit.capitalize()}")
    print(f"   • Training Type  : {training_type}")
    print(f"   • MQTT Enabled   : {'YES' if mqtt_enabled else 'NO'}")

    print(f"\n🧍  Athlete Parameters")
    print(f"   • HR_rest        : {athlete['HR_rest']} bpm")
    print(f"   • HR_max         : {athlete['HR_max']} bpm")
    print(f"   • FTP            : {athlete['FTP']} W")
    print(f"   • Weight         : {athlete['weight_kg']:.1f} kg")

    if mqtt_enabled:
        print(f"\n  📲  Open the new terminal where is running : mqtt.py")

    print("\n🏃💨  Starting training...")

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
    q_table_path = f"data/q-tables/q_{profile_label}_{training_type}.json"
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
    os.makedirs("simulation_training_logs", exist_ok=True)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"simulation_training_logs/{athlete_profile}_{training_type}_{circuit_name}_{today_date}.csv"
    
    fieldnames = [
        'second', 
        'phase',
        'action', 
        'reward', 
        'fatigue',  
        'HR_zone', 
        'power_zone', 
        'target_HR', 
        'target_power',
        'fatigue_score',
        'fatigue_level',
        'slope', 
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
