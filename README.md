
# 🏃‍♂️ Smart Pacer

**Smart Pacer** is a reinforcement-learning system that coaches runners in real time, second by second. It uses Q-learning and an MDP framework to pick the best action (speed up, keep pace, slow down) based on your heart-rate zone, power zone, slope, and fatigue. There’s even a simulated MQTT link—think of it like your smartwatch chatting with a personal pacing assistant.
Developed as an IoT course project at La Sapienza.

---

## 🔍 Objective

Smart Pacer’s goal is simple: help you train **smarter, not harder**. It learns a policy that maps your physiological state (HR & power zones, fatigue) and training ground state (slope level) to actions that maximize long-term reward across a workout. Whether you’re feeling under-the-weather or in peak form, Smart Pacer adjusts so you push when you can and back off when you need to.

---

## 📁 Project Structure

- `main.py`: Run a live training session using the learned Q-table
- `q_learning_trainer.py`: Train the policy and save your Q-table
- `runner_env.py`: Environment logic (state transitions, rewards, track integration)
- `mqtt.py`: MQTT subscriber (simulates the smartwatch link)
- `track.py`: Parse and preprocess elevation data from GPX files
- `data/athletes.json`, `data/trainings.json`: Default athlete profiles & workout plans
- `data/q-tables`: Saved Q-tables for every athlete–workout combo
- `data/maps`: Three different running circuits for your sessions
