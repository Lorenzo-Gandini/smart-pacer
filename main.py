from runner_env import RunnerEnv
import os
import sys
import json
import random

if __name__ == "__main__":
        
    # Caricamento JSON
    with open("athletes.json") as f:
        athletes = json.load(f)

    with open("trainings.json") as f:
        trainings = json.load(f)


    athlete = athletes["runner"]
    training = trainings["fartlek"]

    env = RunnerEnv(athlete, training)

    state = env.reset()
    total_reward = 0

    print("TRAINING SESSION :")
    print("Initial state:", state)



    for step_n in range(10):
        step_n, r, done = env.step("keep going")
        total_reward += r

    print("\n🔹After 10 steps: ", env.state, "\nTotal reward : ", total_reward)

    for step_n in range(15):
        action = "accelerate" if step_n%3==0 else "slow down"
        step_n, r, done = env.step(action)
        total_reward += r

    print("\n🔹After 15 steps: ", env.state, "\nTotal reward : ", total_reward)

    for step_n in range(30):
        step_n, r, done = env.step("keep going")
        total_reward += r
    print("Total reward: ", total_reward)