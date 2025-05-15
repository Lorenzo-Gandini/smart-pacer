import json
import random
import os
import time

import matplotlib.pyplot as plt
from runner_env import RunnerEnv, load_json

import paho.mqtt.client as mqtt


# Hyperparametri
alpha = 0.1       # learning rate
gamma = 0.95      # discount factor
epsilon = 0.2     # esplorazione iniziale
min_epsilon = 0.01
decay_rate = 0.99
num_episodes = 2000

#"elite", "runner"
athlete_profile = "amatour"
training_plan = "endurance"
#"fartlek", "progressions", "endurance", "recovery"

actions = ['slow down', 'keep going', 'accelerate']

# Q-table: Q[state][action] = value
Q = {}

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

def choose_action(state_key):
    if random.random() < epsilon or state_key not in Q:
        return random.choice(actions)
    return max(Q[state_key], key=Q[state_key].get)

def send_mqtt(state, action, client, topic):
    payload = json.dumps({
        "minute": state.get("segment_index", 0),
        "phase": state.get("phase_label", "unknown"),
        "fatigue": state.get("fatigue_level", "unknown"),
        "action": action
    })
    client.publish(topic, payload)
    print(f"📤 MQTT messaggio pubblicato: {payload}")

USE_MQTT = False

if USE_MQTT:
    client = mqtt.Client()
    client.connect("broker.emqx.io", 1883, 60)
    topic = "smartpacer/action"


athletes = load_json("athletes.json")
trainings = load_json("trainings.json")
track = load_json("track_data.json")  
athlete = athletes[athlete_profile]
training = trainings[training_plan]

episode_rewards = []

for episode in range(num_episodes):
    env = RunnerEnv(athlete, training, track_data=track, verbose=False)
    state = env.reset()
    state_key = get_state_key(state)
    total_reward = 0
    done = False

    while not done:
        # Initialize Q-value for the state-action pair if not present
        if state_key not in Q:
            Q[state_key] = {a: 0.0 for a in actions} # Initialize Q-values for all actions

        action = choose_action(state_key)
        next_state, reward, done = env.step(action)
        next_key = get_state_key(next_state)

        if next_key not in Q:
            Q[next_key] = {a: 0.0 for a in actions}

        # Q-learning update
        best_next = max(Q[next_key].values())
        Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

        state_key = next_key
        total_reward += reward
    
    episode_rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * decay_rate)

    if episode % 200 == 0:
        print(f"Episode {episode+1}: reward totale = {total_reward:.2f}")


# Graph
plt.plot(episode_rewards)
plt.title(f"Episodic Total Reward - {athlete_profile} - {training_plan}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(f"data/q_learn_{athlete_profile}_{training_plan}_{num_episodes}ep_{alpha}a_{gamma}g_{epsilon}e.jpg")
plt.show()

# Final evaluation with episoln = 0
print("\n\n🎯 Esecuzione con policy appresa (senza esplorazione):")
env = RunnerEnv(athlete, training, verbose=False)
state = env.reset()
state_key = get_state_key(state)
total_reward = 0
done = False

while not done:
    if state_key not in Q:
        action = random.choice(actions)
    else:
        action = max(Q[state_key], key=Q[state_key].get)

    next_state, reward, done = env.step(action)
    total_reward += reward
    state_key = get_state_key(next_state)
    state = next_state 

    if USE_MQTT:
        send_mqtt(state, action, client, topic)
    time.sleep(0.01)

print(f"\n🏁 Reward totale ottenuto dalla policy appresa: {total_reward:.2f}")

# Save the Q-table to a file

# Ensure the directory exists
os.makedirs("data/q-table", exist_ok=True)

# Save the Q-table as a JSON file
q_table_path = f"data/q-table/q_table_{athlete_profile}_{training_plan}_{num_episodes}ep.json"

# Salvataggio Q-table
with open(q_table_path, "w") as f:
    json.dump({str(k): v for k, v in Q.items()}, f, indent=2)
print(f"\n💾 Q-table salvata in {q_table_path}")

print(f"Q-table saved to {q_table_path}")