import json
from runner_env import RunnerEnv, load_json
import sys

# Caricamento Q-table
with open("data/q-table/q_table_elite_fartlek_2000ep.json") as f:
    raw_q_table = json.load(f)

# Ricostruzione della Q-table con tuple come chiavi
Q = {eval(k): v for k, v in raw_q_table.items()}

athletes = load_json("athletes.json")
trainings = load_json("trainings.json")
track = load_json("track_data.json")

athlete = athletes["elite"]
training = trainings["progressions"]

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

# Inizializza ambiente con tracciato e verbose attivo
env = RunnerEnv(athlete, training, track_data=track, verbose=True)

state = env.reset()
done = False
total_reward = 0

# Simulazione usando la Q-table appresa
while not done:
    state_key = get_state_key(state)
    if state_key in Q:
        action = max(Q[state_key], key=Q[state_key].get)
    else:
        action = 'keep going'  # fallback
    state, reward, done = env.step(action)
    total_reward += reward

print(f"\n🏁 Reward totale accumulato: {total_reward:.2f}")
