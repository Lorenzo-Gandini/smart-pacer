import random
import json

class RunnerEnv:
    def __init__(self, athlete_profile, training_plan, track_data=None, verbose=True):
        self.athlete = athlete_profile
        self.training = training_plan
        self.tracking = track_data or [] 
        self.verbose = verbose
        self.training_type = self.training.get("name", "generic").lower()
        self._get_bio_parameters()
        self.reset()

    def _get_bio_parameters(self):
        self.fitness_factor = self.athlete["fitness_factor"]
        self.hr_rest = self.athlete["HR_rest"]
        self.hr_max = self.athlete["HR_max"]
        self.ftp = self.athlete["FTP"]
        self.weight = self.athlete["weight_kg"]

    def reset(self):
        self.second = 0
        self.fatigue_score = 0.0
        self.time_in_high_zones = 0
        self.hr_float = 1.0
        self.expanded_plan = self._expand_training_segments(self.training["segments"])
        self.state = {
            "HR_zone": "Z1",
            "power_zone": "Z1",
            "fatigue_level": "low",
            "segment_index": 0,
            "phase_label": self.expanded_plan[0]["phase"],
            "target_hr_zone": self.expanded_plan[0]["target_hr_zone"],
            "target_power_zone": self.expanded_plan[0]["target_power_zone"],
            "slope_level": self._get_slope_level(0)
        }
        return self.state

    def step(self, action):
        self._update_power_zone(action)
        self._update_hr_zone(action)
        self._update_fatigue(action)
        self._advance_segment()
        reward = self._compute_reward(action)
        done = self.second >= self.training["duration"] * 60
        self._log_state(action, reward, done)
        return self.state.copy(), reward, done

    def _update_power_zone(self, action):
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
        i = zones.index(self.state["power_zone"])
        if action == "accelerate" and i < 4:
            i += 1
        elif action == "slow down" and i > 0:
            i -= 1
        self.state["power_zone"] = zones[i]

    def _update_hr_zone(self, action):
        target = self._get_zone(self.state["power_zone"])
        delta = (target - self.hr_float) * 0.2
        self.hr_float = max(1.0, min(5.0, self.hr_float + delta))
        self.state["HR_zone"] = f"Z{round(self.hr_float)}"

    def _update_fatigue(self, action):
        hr_level = self._get_zone(self.state["HR_zone"])
        power_level = self._get_zone(self.state["power_zone"])

        fatigue_gain = {
            1: 0.01,
            2: 0.05,
            3: 0.12,
            4: 0.3,
            5: 0.5
        }[hr_level]

        if self.state["phase_label"] == "push":
            fatigue_gain *= 1.2
        elif self.state["phase_label"] in ["recover", "cooldown"]:
            fatigue_gain *= 0.5

        if hr_level <= 2 and action == "slow down":
            fatigue_gain -= 0.1

        # Aggiungi accumulo da tempo in Z4-Z5
        if hr_level >= 4:
            self.time_in_high_zones += 1
        else:
            self.time_in_high_zones = max(0, self.time_in_high_zones - 1)
        fatigue_gain += 0.01 * self.time_in_high_zones

        # Effetto combinato HR + Power alti
        if hr_level >= 4 and power_level >= 4:
            fatigue_gain *= 1.3

        modifier = {
            "fartlek": 1.1,
            "interval": 1.0,
            "progressions": 0.9,
            "endurance": 0.8,
            "recovery": 0.7
        }.get(self.training_type, 1.0)

        fatigue_gain *= modifier
        self.fatigue_score += fatigue_gain + random.uniform(-0.02, 0.02)
        self.fatigue_score *= self.fitness_factor
        self.fatigue_score = max(0, min(10, self.fatigue_score))

        if self.fatigue_score <= 3:
            level = "low"
        elif self.fatigue_score <= 7:
            level = "medium"
        else:
            level = "high"
        self.state["fatigue_level"] = level

    def _advance_segment(self):
        self.second += 1
        if self.second < len(self.expanded_plan):
            current = self.expanded_plan[self.second]
            self.state["segment_index"] = self.second
            self.state["phase_label"] = current["phase"]
            self.state["target_hr_zone"] = current["target_hr_zone"]
            self.state["target_power_zone"] = current["target_power_zone"]
        else:
            self.state["segment_index"] = self.second

        self.state["slope_level"] = self._get_slope_level(self.second)

    def _get_slope_level(self, second):
        if not self.tracking or second >= len(self.tracking):
            return "flat"
        return self.tracking[second].get("slope", "flat")

    def _expand_training_segments(self, segments):
        expanded = []
        for segment in segments:
            repeat = segment.get("repeat", 1)
            if "sub_segments" in segment:
                for _ in range(repeat):
                    for sub in segment["sub_segments"]:
                        for _ in range(int(sub["duration_min"] * 60)):
                            expanded.append(sub)
            else:
                for _ in range(repeat):
                    for _ in range(int(segment["duration_min"] * 60)):
                        expanded.append(segment)
        return expanded

    def _compute_reward(self, action):
        reward = 0.0
        hr_zone = self._get_zone(self.state["HR_zone"])
        power_zone = self._get_zone(self.state["power_zone"])
        target_hr = self._get_zone(self.state["target_hr_zone"])
        target_power = self._get_zone(self.state["target_power_zone"])

        hr_diff = abs(hr_zone - target_hr)
        power_diff = abs(power_zone - target_power)

        hr_reward = {0: +1.5, 1: -0.5, 2: -1.5, 3: -3.0, 4: -4.0}.get(hr_diff, -4.0)
        power_reward = {0: +1.5, 1: -0.5, 2: -1.5, 3: -3.0, 4: -4.0}.get(power_diff, -4.0)

        reward += hr_reward + power_reward

        # Penalità fatica
        fatigue_penalty = {"low": 0.0, "medium": -1.0, "high": -3.0}[self.state["fatigue_level"]]
        reward += fatigue_penalty

        # Penalità/bonus fase
        phase = self.state["phase_label"]
        if phase == "recover" and action == "accelerate":
            reward -= 2.0
        elif phase == "cooldown" and action != "slow down":
            reward -= 1.5
        elif phase == "push" and action == "accelerate":
            reward += 0.5
        elif phase == "warmup" and hr_zone >= 4:
            reward -= 1.0

        # Bonus stabilità
        if hr_diff == 0 and power_diff == 0:
            reward += 0.5

        # Penalità per gap HR vs Power
        if abs(hr_zone - power_zone) >= 2:
            reward -= 1.0

        # Bonus per direzione coerente
        if hr_zone < target_hr and action == "accelerate":
            reward += 0.5
        elif hr_zone > target_hr and action == "slow down":
            reward += 0.5

        return reward

    def _get_zone(self, zone):
        return {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}[zone]

    def _log_state(self, action, reward, done):
        if not self.verbose:
            return
        print(f"Second: {self.second} | Action: {action.upper()} | Reward: {reward:.2f} | Done: {done}")
        print(f"➤ Phase : {self.state['phase_label']} | Target HR Zone: {self.state['target_hr_zone']} | Target Power Zone: {self.state['target_power_zone']}")
        print(f"➤ Actual State : HR Zone: {self.state['HR_zone']} | Power Zone: {self.state['power_zone']} | Fatigue Level: {self.state['fatigue_level']} | Slope Level: {self.state['slope_level']}")
        print("--------------------------------------------------")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
