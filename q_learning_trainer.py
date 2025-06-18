# import json
# import random
# import os
# from datetime import datetime
# import matplotlib.pyplot as plt
# from runner_env import RunnerEnv, load_json, ACTIONS
# from tqdm import tqdm

# # ------------------------------------------------------------------------------
# # HYPERPARAMETER CONFIGURATIONS
# # hyperparameter_sets = [
# #     {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
# #     {"alpha": 0.05, "gamma": 0.95, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
# #     {"alpha": 0.1, "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
# #     {"alpha": 0.01, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
# #     {"alpha": 0.1, "gamma": 0.90, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
# #     {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.97},
# #     {"alpha": 0.05, "gamma": 0.99, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
# #     {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98}
# # ]


# hyperparameter_sets = [
#     #Migliore fin'ora
#     {"alpha": 0.1, "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
#     # Più conservativa:
#     {"alpha": 0.05, "gamma": 0.98, "initial_epsilon": 0.15, "min_epsilon": 0.01, "decay_rate": 0.985},
#     # Apprendimento più esplorativo:
#     {"alpha": 0.1, "gamma": 0.98, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.96},
#     # Mix stabile:
#     {"alpha": 0.08, "gamma": 0.99, "initial_epsilon": 0.25, "min_epsilon": 0.01, "decay_rate": 0.975},

# ]

# num_episodes = 2000 #500

# # ------------------------------------------------------------------------------
# athletes = load_json("data/athletes.json")
# trainings = load_json("data/trainings.json")
# track = load_json("data/maps/acquedotti.json")

# # ------------------------------------------------------------------------------
# def get_state_key(state):
#     return (
#         state['HR_zone'],
#         state['power_zone'],
#         state['fatigue_level'],
#         state['phase_label'],
#         state['target_hr_zone'],
#         state['target_power_zone'],
#         state['slope_level']
#     )

# def choose_action(Q, state_key, epsilon):
#     if random.random() < epsilon or state_key not in Q:
#         return random.choice(ACTIONS)
#     return max(Q[state_key], key=Q[state_key].get)

# def run_training(athlete_profile, training_plan, hparams, base_folder):
#     alpha = hparams["alpha"]
#     gamma = hparams["gamma"]
#     initial_epsilon = hparams["initial_epsilon"]
#     min_epsilon = hparams["min_epsilon"]
#     decay_rate = hparams["decay_rate"]

#     Q = {}
#     epsilon = initial_epsilon
#     episode_rewards = []

#     athlete = athletes[athlete_profile]
#     training = trainings[training_plan]

#     print(f"\nRunning: {athlete_profile} - {training_plan} | α={alpha}, γ={gamma}, ε={initial_epsilon}, decay={decay_rate}")
#     for episode in tqdm(range(num_episodes), desc="", ncols=50, leave=False, bar_format="{l_bar}{bar}"):
#         env = RunnerEnv(athlete, training, track_data=track, verbose=False)
#         state = env.reset()
#         state_key = get_state_key(state)
#         total_reward = 0
#         done = False

#         while not done:
#             if state_key not in Q:
#                 Q[state_key] = {a: 0.0 for a in ACTIONS}

#             action = choose_action(Q, state_key, epsilon)
#             next_state, reward, done = env.step(action)
#             next_key = get_state_key(next_state)

#             if next_key not in Q:
#                 Q[next_key] = {a: 0.0 for a in ACTIONS}

#             best_next = max(Q[next_key].values())
#             Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

#             state_key = next_key
#             total_reward += reward

#         episode_rewards.append(total_reward)
#         epsilon = max(min_epsilon, epsilon * decay_rate)


#     plot_path = f"{base_folder}/figures/{athlete_profile}_{training_plan}.jpg"
#     plt.plot(episode_rewards)
#     plt.title(f"Reward Trend – {athlete_profile} / {training_plan}")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(plot_path)
#     plt.clf()


#     # Final evaluation with epsilon = 0
#     env = RunnerEnv(athlete, training, track_data=track, verbose=False)
#     state = env.reset()
#     state_key = get_state_key(state)
#     total_reward_eval = 0
#     done = False

#     while not done:
#         action = max(Q[state_key], key=Q[state_key].get) if state_key in Q else random.choice(ACTIONS)
#         next_state, reward, done = env.step(action)
#         total_reward_eval += reward
#         state_key = get_state_key(next_state)

#     # save Q-table
#     os.makedirs("data/q-table", exist_ok=True)
#     q_table_path = f"{base_folder}/q-tables/q_table_{athlete_profile}_{training_plan}.json"
#     with open(q_table_path, "w") as f:
#         json.dump({str(k): v for k, v in Q.items()}, f, indent=2)

#     return total_reward_eval


# # ------------------------------------------------------------------------------
# # MAIN
# if __name__ == "__main__":
#     athlete_profiles = ["elite", "runner", "amateur"]
#     training_plans = ["fartlek", "progressions", "endurance", "recovery"]

#     # Create folder for saving results
#     now = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_folder = f"data/q-table/q-tables-{now}"
#     os.makedirs(base_folder, exist_ok=True)
#     os.makedirs(f"{base_folder}/figures", exist_ok=True)
#     os.makedirs(f"{base_folder}/q-tables", exist_ok=True)

#     for hparams in hyperparameter_sets:
#         alpha = hparams["alpha"]
#         gamma = hparams["gamma"]
#         epsilon = hparams["initial_epsilon"]

#         history_filename = f"data/history-qtraining/history.txt"

#         with open(history_filename, "w") as history_file:
#             for athlete_profile in athlete_profiles:
#                 for training_plan in training_plans:
#                     reward = run_training(athlete_profile, training_plan, hparams, base_folder)
#                     history_file.write(f"{athlete_profile}-{training_plan}: {reward:.2f}\n")
#                     history_file.flush()



import json
import random
import os
import matplotlib.pyplot as plt
from runner_env import RunnerEnv, load_json, ACTIONS
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# CONFIG
# hyperparameter_sets = [
#     {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
#     {"alpha": 0.05, "gamma": 0.95, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
#     {"alpha": 0.1, "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
#     {"alpha": 0.01, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
#     {"alpha": 0.1, "gamma": 0.90, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
#     {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.97},
#     {"alpha": 0.05, "gamma": 0.99, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
#     {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98},

#     {"alpha": 0.08, "gamma": 0.995, "initial_epsilon": 0.25, "min_epsilon": 0.01, "decay_rate": 0.98},
#     {"alpha": 0.05, "gamma": 0.995, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.985},
#     {"alpha": 0.03, "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.997},
#     {"alpha": 0.07, "gamma": 0.98, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98},
#     {"alpha": 0.1, "gamma": 0.97, "initial_epsilon": 0.5, "min_epsilon": 0.01, "decay_rate": 0.98},
#     {"alpha": 0.05, "gamma": 0.995, "initial_epsilon": 0.05, "min_epsilon": 0.01, "decay_rate": 0.99},
# ]

top_configs = [
    # amateur
    ("amateur", {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98}),
    ("amateur", {"alpha": 0.05, "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.985}),

    # Runner
    ("runner", {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99}),
    ("runner", {"alpha": 0.1, "gamma": 0.97, "initial_epsilon": 0.5, "min_epsilon": 0.01, "decay_rate": 0.98}),

    # Elite
    ("elite", {"alpha": 0.1, "gamma": 0.95, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.97}),
    ("elite", {"alpha": 0.1, "gamma": 0.97, "initial_epsilon": 0.5, "min_epsilon": 0.01, "decay_rate": 0.98}),
]


num_episodes = 2000
athletes = load_json("data/athletes.json")
trainings = load_json("data/trainings.json")
track = load_json("data/maps/acquedotti.json")

# OUTPUT DIR
now = datetime.now().strftime("%Y%m%d")
base_folder = f"data/qtraining_runs_2000/{now}"
os.makedirs(base_folder, exist_ok=True)
os.makedirs(f"{base_folder}/q-tables", exist_ok=True)
os.makedirs(f"{base_folder}/rewards", exist_ok=True)

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
        return random.choice(ACTIONS)
    return max(Q[state_key], key=Q[state_key].get)

def train_combo(args):
    athlete_profile, training_plan, hparams = args
    alpha = hparams["alpha"]
    gamma = hparams["gamma"]
    initial_epsilon = hparams["initial_epsilon"]
    min_epsilon = hparams["min_epsilon"]
    decay_rate = hparams["decay_rate"]
    suffix = f"a{int(alpha*100):02d}_g{int(gamma*100):02d}_e{int(initial_epsilon*100):02d}"

    Q = {}
    epsilon = initial_epsilon
    episode_rewards = []

    athlete = athletes[athlete_profile]
    training = trainings[training_plan]
    env = RunnerEnv(athlete, training, track_data=track, verbose=False)

    for episode in range(num_episodes):
        #reset environment at the beginning of each episode
        state = env.reset()
        state_key = get_state_key(state)
        total_reward = 0
        done = False 

        while not done:
            # Initialize Q-values for unseen state
            if state_key not in Q:
                Q[state_key] = {a: 0.0 for a in ACTIONS}

            action = choose_action(Q, state_key, epsilon)
            
            #take the action, observe the next state, reward, and done flag
            next_state, reward, done = env.step(action)
            next_key = get_state_key(next_state)
            
            # Initialize Q-values for unseen next state
            if next_key not in Q:
                Q[next_key] = {a: 0.0 for a in ACTIONS}

            #Q-learning update rule, take the max
            best_next = max(Q[next_key].values())
            Q[state_key][action] += alpha * (reward + gamma * best_next - Q[state_key][action])

            # Transition to the next state
            state_key = next_key
            total_reward += reward

        # Store the total reward of this episode
        episode_rewards.append(total_reward)
        # Decay exploration rate for next episode
        epsilon = max(min_epsilon, epsilon * decay_rate)

    #save  the Q-table
    qpath = f"{base_folder}/q-tables/q_{athlete_profile}_{training_plan}_{suffix}.json"
    with open(qpath, "w") as f:
        json.dump({str(k): v for k, v in Q.items()}, f, indent=2)

    # Save reward trend
    rpath = f"{base_folder}/rewards/{athlete_profile}_{training_plan}_{suffix}.json"
    with open(rpath, "w") as f:
        json.dump(episode_rewards, f)

    return athlete_profile, training_plan, episode_rewards

def plot_all_rewards(base_folder):
    reward_dir = f"{base_folder}/rewards"
    plt.figure(figsize=(10, 6))
    for file in os.listdir(reward_dir):
        path = os.path.join(reward_dir, file)
        with open(path) as f:
            rewards = json.load(f)
        label = file.replace(".json", "")
        plt.plot(rewards, label=label)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_folder}/reward_trends.png")
    plt.close()

if __name__ == "__main__":
    training_plans = ["fartlek", "progressions", "endurance", "recovery"]

    combos = [(athlete, training, hparams) for athlete, hparams in top_configs for training in training_plans]

    print(f"🔄 Training {len(combos)} combinations using {min(cpu_count(), 12)} processes...")

    with Pool(processes=min(cpu_count(), 12)) as pool:
        results = list(tqdm(pool.imap(train_combo, combos), total=len(combos)))

    plot_all_rewards(base_folder)
    print(f"\n✅ All trainings complete. Results in: {base_folder}")
