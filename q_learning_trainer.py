import json
import random

import matplotlib.pyplot as plt
from runner_env import RunnerEnv, load_json

# Hyperparametri
alpha = 0.1       # learning rate
gamma = 0.95      # discount factor
epsilon = 0.1
num_episodes = 500 # numero di episodi per il training


# Azioni possibili
actions = ['slow down', 'keep going', 'accelerate']

# Q-table: Q[state][action] = valore
Q = {}

def get_state_key(state):
    # Serializza lo stato in una tupla hashabile (senza slope per ora)
    return (
        state['HR_zone'],
        state['power_zone'],
        state['fatigue_level'],
        state['phase_label'],
        state['target_hr_zone'],
        state['target_power_zone']
    )

def choose_action(state_key):
    if random.random() < epsilon or state_key not in Q:
        return random.choice(actions)
    return max(Q[state_key], key=Q[state_key].get)

# Carica i dati
athletes = load_json("athletes.json")
trainings = load_json("trainings.json")
athlete = athletes["runner"]
training = trainings["fartlek"]

episode_rewards = []

# Inizia training
for episode in range(num_episodes):
    env = RunnerEnv(athlete, training, verbose=False)
    state = env.reset()
    state_key = get_state_key(state)
    total_reward = 0

    initial_epsilon = 0.2
    min_epsilon = 0.01
    decay_rate = 0.99
    

    done = False
    while not done:
        if state_key not in Q:
            Q[state_key] = {a: 0.0 for a in actions}

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

    if episode % 50 == 0:
        print(f"Episode {episode+1}: reward totale = {total_reward:.2f}")


plt.plot(episode_rewards)
plt.title("Episodic Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
