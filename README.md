
# 🏃‍♂️ Smart Pacer

**Smart Pacer** is a reinforcement learning-based system that simulates and advises runners second-by-second during training sessions. It integrates Q-learning and a Markov Decision Process approach to guide action selection (accelerate, keep going, slow down) based on physiological state, workout phase, and real-world elevation data. A mqtt communciation is simulated, like a comunciation with a smartwatch that works as pacer-assistant.
This project is developed as a project course for the Internet of Things exam @LaSapienza. 

---

## 🔍 Objective

The goal of my project is to model a pacing assistant capable of optimizing training performance while managing fatigue. The assistant learns an optimal policy that maps a state (heart rate zone, power zone, slope of the training ground, fatigue) to an action that maximizes long-term reward over the course of the workout.
The idea comes from sessions where fatigue hits you hard due to other circumstances (bad sleep or indigestion) that don't allow you to train properly. Or even sometimes you feel in a very good shape and the training session can be pushed a little bit further. These phisiological states can be used to maximize the training or avoid injuries.

---

## 📁 Project Structure

- `main.py`: runs an actual training session using a learned policy
- `q_learning_trainer.py`: trains the policy and saves Q-table
- `runner_env.py`: environment logic (state transition, reward, track integration)
- `mqtt.py`: MQTT listener/subscriber
- `track.py`: parses and preprocesses elevation data from GPX
- `data/athletes.json`, `data/trainings.json`: default data for ethletes and training programs
- `data/q-tables`: q-tables for all the combinations of athlete and training program
- `data/maps`: three different tracks for the training session
