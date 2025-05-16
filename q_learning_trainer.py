import json
import random
import os
import time
import matplotlib.pyplot as plt
from runner_env import RunnerEnv, load_json
from tqdm import tqdm

# ------------------------------------------------------------------------------
# HYPERPARAMETER CONFIGURATIONS
hyperparameter_sets = [
    {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.97},
    {"alpha": 0.05, "gamma": 0.99, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
    {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98}
]

num_episodes = 2000

# ------------------------------------------------------------------------------
actions = ['slow down', 'keep going', 'accelerate']
athletes = load_json("data/json/athletes.json")
trainings = load_json("data/json/trainings.json")
track = load_json("data/json/track_data.json")

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

def choose_action(Q, state_key, epsilon):
    if random.random() < epsilon or state_key not in Q:
        return random.choice(actions)
    return max(Q[state_key], key=Q[state_key].get)

def run_training(athlete_profile, training_plan, hparams):
    alpha = hparams["alpha"]
    gamma = hparams["gamma"]
    initial_epsilon = hparams["initial_epsilon"]
    min_epsilon = hparams["min_epsilon"]
    decay_rate = hparams["decay_rate"]

    Q = {}
    epsilon = initial_epsilon
    episode_rewards = []

    athlete = athletes[athlete_profile]
    training = trainings[training_plan]

    print(f"\nRunning: {athlete_profile} - {training_plan} | α={alpha}, γ={gamma}, ε={initial_epsilon}, decay={decay_rate}")
    for episode in tqdm(range(num_episodes), desc="", ncols=50, leave=False, bar_format="{l_bar}{bar}"):
        env = RunnerEnv(athlete, training, track_data=track, verbose=False)
        state = env.reset()
        state_key = get_state_key(state)
        total_reward = 0
        done = False

        while not done:
            if state_key not in Q:
                Q[state_key] = {a: 0.0 for a in actions}

            action = choose_action(Q, state_key, epsilon)
            next_state, reward, done = env.step(action)
            next_key = get_state_key(next_state)

            if next_key not in Q:
                Q[next_key] = {a: 0.0 for a in actions}

            best_next = max(Q[next_key].values())
            Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

            state_key = next_key
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)


    os.makedirs("data/figures", exist_ok=True)
    plot_path = f"data/figures/q_learn_{athlete_profile}_{training_plan}_{num_episodes}ep_{alpha}a_{gamma}g.jpg"
    plt.plot(episode_rewards)
    plt.title(f"Total Reward - {athlete_profile} - {training_plan}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.clf()

    # Final evaluation with epsilon = 0
    env = RunnerEnv(athlete, training, track_data=track, verbose=False)
    state = env.reset()
    state_key = get_state_key(state)
    total_reward_eval = 0
    done = False

    while not done:
        action = max(Q[state_key], key=Q[state_key].get) if state_key in Q else random.choice(actions)
        next_state, reward, done = env.step(action)
        total_reward_eval += reward
        state_key = get_state_key(next_state)

    # save Q-table
    os.makedirs("data/q-table", exist_ok=True)
    q_table_path = f"data/q-table/q_table_{athlete_profile}_{training_plan}_{num_episodes}ep_{alpha}a_{gamma}g_{epsilon}e.json"
    with open(q_table_path, "w") as f:
        json.dump({str(k): v for k, v in Q.items()}, f, indent=2)

    return total_reward_eval

# ------------------------------------------------------------------------------
# MAIN
athlete_profiles = ["elite", "runner", "amatour"]
training_plans = ["fartlek", "progressions", "endurance", "recovery"]

for hparams in hyperparameter_sets:
    alpha = hparams["alpha"]
    gamma = hparams["gamma"]
    epsilon = hparams["initial_epsilon"]

    history_filename = f"data/history-qtraining/history_{alpha}a_{gamma}g_{epsilon}e.txt"
    with open(history_filename, "w") as history_file:
        for athlete_profile in athlete_profiles:
            for training_plan in training_plans:
                reward = run_training(athlete_profile, training_plan, hparams)
                history_file.write(f"{athlete_profile}-{training_plan}: {reward:.2f}\n")
                history_file.flush()
