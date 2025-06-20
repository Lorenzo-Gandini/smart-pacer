import json
import random
import os
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from runner_env import RunnerEnv, load_json, ACTIONS
from tqdm import tqdm

# === CONFIG ===
num_episodes = 500  # first block of experiments
track = load_json("data/maps/acquedotti.json")  # single circuit only
athletes = list(load_json("data/athletes.json").keys())
training_plans = list(load_json("data/trainings.json").keys())

# Hyperparameter sets for testing
hyperparameter_sets = [
    {"alpha": 0.1,  "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
    {"alpha": 0.05, "gamma": 0.95, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
    {"alpha": 0.1,  "gamma": 0.99, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
    {"alpha": 0.01, "gamma": 0.95, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.99},
    {"alpha": 0.1,  "gamma": 0.90, "initial_epsilon": 0.2, "min_epsilon": 0.01, "decay_rate": 0.98},
    {"alpha": 0.1,  "gamma": 0.95, "initial_epsilon": 0.4, "min_epsilon": 0.01, "decay_rate": 0.97},
    {"alpha": 0.05, "gamma": 0.99, "initial_epsilon": 0.3, "min_epsilon": 0.01, "decay_rate": 0.995},
    {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.1, "min_epsilon": 0.01, "decay_rate": 0.98}
]

# hyperparameter_sets_1000 = [
#     {"alpha": 0.05, "gamma": 0.90, "initial_epsilon": 0.10, "min_epsilon": 0.01, "decay_rate": 0.98}, 
#     {"alpha": 0.10, "gamma": 0.95, "initial_epsilon": 0.20, "min_epsilon": 0.01, "decay_rate": 0.99},  
#     {"alpha": 0.10, "gamma": 0.99, "initial_epsilon": 0.20, "min_epsilon": 0.01, "decay_rate": 0.98},  
#     {"alpha": 0.05, "gamma": 0.95, "initial_epsilon": 0.30, "min_epsilon": 0.01, "decay_rate": 0.995}
# ]

# hyperparameter_sets = [
#     {"alpha": 0.10, "gamma": 0.95, "initial_epsilon": 0.20, "min_epsilon": 0.01, "decay_rate": 0.99}
# ]

# OUTPUT setup
today = datetime.now().strftime("%Y%m%d")
base_folder = f"data/qtraining_runs_{today}_episodes{num_episodes}"
os.makedirs(f"{base_folder}/q-tables", exist_ok=True)
os.makedirs(f"{base_folder}/rewards", exist_ok=True)

# --- Helpers ---
def get_state_key(state):
    return (
        state['HR_zone'], state['power_zone'], state['fatigue_level'],
        state['phase_label'], state['target_hr_zone'],
        state['target_power_zone'], state['slope_level']
    )


def choose_action(Q, state_key, epsilon):
    if random.random() < epsilon or state_key not in Q:
        return random.choice(ACTIONS)
    return max(Q[state_key], key=Q[state_key].get)

# --- Plotting ---
def plot_convergence(results):
    # results: list of (athlete, training, label, rewards, med_frac, high_frac)
    groups = defaultdict(list)
    for ath, tr, lbl, rw, *_ in results:
        groups[(ath, tr)].append((lbl, rw))
    for (ath, tr), runs in groups.items():
        plt.figure(figsize=(8, 6))
        for lbl, rw in runs:
            plt.plot(rw, label=lbl)
        plt.title(f"Convergence: {ath} / {tr}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{base_folder}/convergence_{ath}_{tr}.png")
        plt.close()


def plot_heatmap_final(results):
    # pivot final reward (mean last 50) per athlete
    data = []
    for ath, tr, lbl, rw, *_ in results:
        data.append((ath, tr, lbl, np.mean(rw[-50:])))
    df = pd.DataFrame(data, columns=['athlete', 'training', 'config', 'final'])

    for ath in df['athlete'].unique():
        sub = df[df['athlete'] == ath]
        pivot = sub.pivot(index='training', columns='config', values='final')

        plt.figure(figsize=(pivot.shape[1] * 1.2, pivot.shape[0] * 1.2))
        im = plt.imshow(pivot, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Final Reward')

        for i, training in enumerate(pivot.index):
            for j, config in enumerate(pivot.columns):
                val = pivot.loc[training, config]
                plt.text(j, i, f"{val:.1f}", ha='center', va='center', color='white', fontsize=8)

        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
        plt.title(f"Heatmap Final Reward - {ath}")
        plt.tight_layout()
        plt.savefig(f"{base_folder}/heatmap_final_{ath}.png")
        plt.close()


def plot_convergence_grid(results, episodes):
    # create grid of convergence curves: rows=athletes, cols=training_plans
    athletes_list = sorted({r[0] for r in results})
    trainings_list = sorted({r[1] for r in results})
    configs = sorted({r[2] for r in results})

    fig, axes = plt.subplots(
        len(athletes_list), len(trainings_list),
        figsize=(len(trainings_list)*4, len(athletes_list)*3),
        sharex=True, sharey=True
    )

    for i, ath in enumerate(athletes_list):
        for j, tr in enumerate(trainings_list):
            ax = axes[i, j] if axes.ndim>1 else axes[max(i,j)]
            for a2, t2, cfg, rw, *_ in results:
                if a2 == ath and t2 == tr:
                    ax.plot(rw[:episodes], label=cfg, lw=1)
            if i == 0:
                ax.set_title(tr)
            if j == 0:
                ax.set_ylabel(ath)
            ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Convergence Curves (first {episodes} eps)")
    plt.tight_layout(rect=[0,0,0.9,0.95])
    out_path = os.path.join(base_folder, f"convergence_grid_{episodes}ep.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🔳 Saved convergence grid: {out_path}")


def create_summary_report(results, base_folder):
    report_lines = []
    report_lines.append("🏃‍♂️ Q-Learning Summary Report\n")
    report_lines.append(f"Date: {datetime.now().isoformat()}\n")
    report_lines.append("Best Final Rewards and Fatigue Metrics:\n")

    best = {}
    for athlete, training, cfg, rewards, med, high in results:
        key = (athlete, training)
        final = np.mean(rewards[-50:])
        if key not in best or final > best[key][2]:
            best[key] = (cfg, final, med, high)

    for (ath, tr), (cfg, final, med, high) in best.items():
        report_lines.append(
            f"- {ath:8} | {tr:12} | cfg={cfg:15} | "
            f"final={final:7.1f} | %med={med*100:5.1f}% | %high={high*100:5.1f}%"
        )

    report_path = os.path.join(base_folder, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"📄 Summary report written to {report_path}")

# === MAIN ===
if __name__ == '__main__':
    results = []

    print(f"🔄 Running single-threaded: {len(athletes)*len(training_plans)*len(hyperparameter_sets)} experiments, {num_episodes} episodes each...")
    for athlete in athletes:
        for training in training_plans:
            for params in hyperparameter_sets:
                a, g = params['alpha'], params['gamma']
                e0, min_e, decay = params['initial_epsilon'], params['min_epsilon'], params['decay_rate']
                label = f"a{int(a*100)}_g{int(g*100)}_e{int(e0*100)}"
                print(f"➡️ {athlete} | {training} | alpha={a}, gamma={g}, eps0={e0}, decay={decay}")

                Q = {}
                eps = e0
                env = RunnerEnv(
                    load_json("data/athletes.json")[athlete],
                    load_json("data/trainings.json")[training],
                    track_data=track, verbose=False
                )

                rewards, med_frac, high_frac = [], [], []

                for ep in tqdm(range(num_episodes), desc="Episodes", ncols=60):
                    state = env.reset()
                    key = get_state_key(state)
                    done = False
                    total_r = 0
                    med_c = high_c = steps = 0

                    while not done:
                        Q.setdefault(key, {act:0.0 for act in ACTIONS})
                        action = choose_action(Q, key, eps)
                        nxt, r, done = env.step(action)
                        nk = get_state_key(nxt)
                        Q.setdefault(nk, {act:0.0 for act in ACTIONS})
                        best_next = max(Q[nk].values())
                        Q[key][action] += a * (r + g * best_next - Q[key][action])
                        key = nk
                        total_r += r
                        if nxt['fatigue_level']=="medium": med_c += 1
                        if nxt['fatigue_level']=="high":   high_c += 1
                        steps += 1

                    rewards.append(total_r)
                    med_frac.append(med_c/steps)
                    high_frac.append(high_c/steps)
                    eps = max(min_e, eps * decay)

                qpath = f"{base_folder}/q-tables/q_{athlete}_{training}_{label}.json"
                with open(qpath, 'w') as f: json.dump({str(k):v for k,v in Q.items()}, f, indent=2)
                rpath = f"{base_folder}/rewards/r_{athlete}_{training}_{label}.json"
                with open(rpath, 'w') as f: json.dump(rewards, f)

                results.append((athlete, training, label, rewards, np.mean(med_frac[-50:]), np.mean(high_frac[-50:])))

    # generate plots, grid, heatmaps, and report
    plot_convergence(results)
    plot_convergence_grid(results, num_episodes)
    plot_heatmap_final(results)
    create_summary_report(results, base_folder)

    print(f"✅ All done. Outputs in {base_folder}")