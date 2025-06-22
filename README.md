
# 🏃‍♂️ Smart Pacer

**Smart Pacer** is a reinforcement-learning system that coaches runners in real time, second by second. It uses Q-learning and an MDP framework to pick the best action (speed up, keep pace, slow down) based on your heart-rate zone, power zone, slope, and fatigue.
There’s even a simulated MQTT link—think of it like your smartwatch chatting with a personal pacing assistant.
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
- `training_visualizer.py` : Providing athlete, training program and track, it creates the video with the fatigue condition of the athlete
- `data/athletes.json`, `data/trainings.json`: Default athlete profiles & workout plans
- `data/q-tables`: Saved Q-tables for every athlete–workout combo
- `data/maps`: Three different running circuits for your sessions
- `data/video`: Simulation of the runners in different scenarios

## 🚀 Run the simulation

### 1. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Interactive Simulation (`main.py`)

When you launch `main.py`, you will be prompted to configure your session with:

1. **Resting heart rate (HR_rest)**  
2. **Maximum heart rate (HR_max)**  
3. **Functional Threshold Power (FTP)**  
4. **Body weight in kg**  
5. **Training circuit** (choose from maps in `data/maps/*.json`)  
6. **Training type** (`fartlek`, `endurance`, `progressions`, `recovery`)  
7. **Enable MQTT communication** (yes/no)

After these inputs, the system will identify the **nearest athlete profile** (elite, runner, or amateur) based on physiological similarity and load the corresponding Q-table.

### Terminal Output Example

```bash
👤 Let's configure your session.
❤️ Resting HR: 42
💖 Max HR: 180
🚴 FTP: 360
🏋️ Weight: 74

🌍 Select your training circuit:
  1. acquedotti   2. belfiore   3. tre_laghi
🔢 Circuit [1–3]: 2   ✅ belfiore selected

🏋️ Choose your training session:
  1. fartlek   2. endurance   3. recovery   4. progressions
🔢 Training [1–4]: 2   ✅ endurance selected

📡 Enable MQTT communication? (y/n): y   ✅ MQTT enabled
📲 Open another terminal and run `python mqtt.py` to follow your pacing assistant

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ CONFIGURATION COMPLETE – SMART PACER READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Profile       : runner
🌍 Circuit       : belfiore
🏋️ Training Type : endurance
📡 MQTT Enabled  : YES

📊 Athlete Stats:
  💓 HR_rest : 42.0 bpm
  ❤️ HR_max  : 180.0 bpm
  🚴 FTP     : 360.0 W
  🏋️ Weight  : 74.0 kg

🏃‍♂️ Starting training...
```

### 4. MQTT Communication

If MQTT is enabled, the Smart Pacer publishes messages every second during the simulation to the topic `smartpacer/action`. These simulate smartwatch instructions.

#### a) Configuration

- **Broker**: Default is `broker.emqx.io`  
- **Topic**: Default is `smartpacer/action`

To customize these, edit the top of `mqtt.py`:

```python
broker = "your.broker.address"
topic  = "smartpacer/action"
```

#### b) Run the Subscriber

Open a terminal and run:

```bash
python mqtt.py
```

You should see:

```bash
🏃‍♂️ SMART PACER CONNECTED 🏃‍♀️
🔊 Waiting for pace instructions...
```

#### c) Real-Time Message Format

Each second, a message like the following is received:

```json
{
  "phase":      "push",
  "second":     123,
  "action":     "keep going",
  "hr_zone":    "Z3",
  "power_zone": "Z3",
  "fatigue":    "medium",
  "slope":      "flat",
  "reward":     1.27
}
```

Which is then rendered in the terminal as:

```
━━━━━━━━━━━━━━ 🕒 02:03 ━━━━━━━━━━━━━━
🔥 Phase       : PUSH         🗻 Slope: flat
🎯 Action      : KEEP GOING   💢 Fatigue: MEDIUM
❤️ HR Zone    : Z3           ⚡ Power Zone: Z3
🎁 Reward      : +1.27
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5. Training Log Output

Upon completion, a CSV file is saved in `training_logs/` with filename:

```
<profile>_<training>_<circuit>_<YYYY-MM-DD>.csv
```

Each row corresponds to one simulated second and includes:
- Heart rate
- Power
- Action
- Zone (HR & Power)
- Fatigue level
- Reward
- Segment slope
- Training phase

### 6. Video of Simulation

Each session can be replayed as a video, showing the athlete’s GPS trace over OpenStreetMap with color-coded segments:

- **Green** = low fatigue
- **Orange** = moderate fatigue
- **Red** = high fatigue

The animation includes:
1. **Moving dot** representing the runner’s position
2. **Fatigue-colored trail** showing exertion level
3. **On-screen status** with:
   - Time elapsed
   - Current phase (warmup, push, etc.)
   - Action taken (accelerate, etc.)
   - HR and Power zone

Videos are stored under `data/video/`. Sample visualizations can be found [here](https://github.com/Lorenzo-Gandini/smart-pacer/tree/main/data/video).
