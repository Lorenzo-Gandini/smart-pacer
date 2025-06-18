
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

### 3. Launch interactive Simulation (`main.py`)

When you launch `main.py`, you will be prompted to enter:

1. **Resting heart rate (HR_rest)**  
2. **Maximum heart rate (HR_max)**  
3. **Functional Threshold Power (FTP)**  
4. **Body weight in kg**  
5. **Fitness factor** (0.7=elite, 1.0=runner, 1.3=amateur)  
6. **Circuit** (select from the list provided of tracks inside of `data/maps/*.json`) 
7. **Training type** (`fartlek`, `endurance`, `progressions`, `recovery`)  
8. **Enable MQTT?** (y/n)

These values configure the environment and determine which base “athlete profile” is more close to the user and which Q-table to load.

### Simulator Outputs
After configuration, the script prints:

- **Nearest profile** (the closest matching base athlete profile by Euclidean distance)  
- **Per‐step logs** if `verbose=True` like:

```bash
Second: 123 | Action: ACCELERATE | Reward: 0.45 | Done: False
➤ Phase : push | Target HR Zone: Z4 | Target Power Zone: Z4
➤ Actual State : HR Zone: Z3 | Power Zone: Z3 | Fatigue Level: medium | Slope Level: uphill
```

- **MQTT messages** (if enabled) are published to `smartpacer/action` and echoed in console by `mqtt.py`, showing a formatted status card of phase, second, action, HR zone, power zone, and fatigue level.

- Upon completion, the script saves a CSV under `training_logs/` (named `<profile>_<training>_<circuit>_<YYYY-MM-DD>.csv`) containing one row per simulated second with columns:


### 4. MQTT Communication

The Smart Pacer can publish live pacing instructions over MQTT and you can run a simple console subscriber to see them in real time—just like a smartwatch would display your next move.

#### a) Configuration

- **Broker**: Default is `broker.emqx.io`.
- **Topic**: Default is `smartpacer/action`.
- To change either, edit the top of `mqtt.py`:

  ```python
  broker = "your.broker.address"
  topic  = "smartpacer/action"
  ```

#### b) Run the Subscriber

Open a terminal and start the listener:

```bash
python mqtt.py
```

You should see:

```
🏃‍♂️ SMART PACER CONNECTED 🏃‍♀️
🔊 Waiting for pace instructions...
```

#### c) Publish from the Simulator

When you launch:

```bash
python main.py
```

and answer yes to the question **“Enable MQTT?”**, the simulator will publish messages every simulated second. A typical payload is:

```json
{
  "phase":      "push",
  "second":     123,
  "action":     "keep going",
  "hr_zone":    "Z3",
  "power_zone": "Z3",
  "fatigue":    "medium"
}
```

The subscriber will format and print each message like:

```
💨 PUSH
🕒 Second: 2:03
🔄 KEEP GOING
💓 HR Zone: Z3
🏋️ Power Zone: Z3
😴 Fatigue: MEDIUM
----------------------------------------
```


### 5. Video of Simulation
There are also animated videos that replay each training session. For every run, the runner’s GPS track is overlaid on an OpenStreetMap background and colored in real time to reflect the athlete’s **fatigue level**:

- **Green** segments indicate low fatigue  
- **Orange** segments indicate moderate fatigue  
- **Red** segments indicate high fatigue  

In each frame you’ll see:
1. **The runner’s current position** (a red dot) moving along the path  
2. **The colored trail behind** showing how fatigue evolved over time  
3. **A status panel** with:
   - Elapsed time (minutes:seconds)  
   - Current workout phase (warmup, push, recover, cooldown)  
   - Action taken (accelerate, keep going, slow down)  
   - Heart-rate zone and power zone  

These animations give an at-a-glance view of how the agent paces itself, where it pushes harder, and when recovery phases kick in.  

Videos can be found inside the video folder [here](https://github.com/Lorenzo-Gandini/smart-pacer/tree/main/data/video).
