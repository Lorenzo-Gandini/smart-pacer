import random
import json

class RunnerEnv:
    def __init__(self, athlete_profile, training_plan, track_data=None, verbose=True):
        self.athlete = athlete_profile
        self.training = training_plan
        self.tracking = track_data or [] 
        self.verbose = verbose
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
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
        i = zones.index(self.state["HR_zone"])
        slope_mod = {"flat": 0, "uphill": 1, "steep_uphill": 2, "downhill": -1, "steep_down": -2}
        delta = {"accelerate": 1, "keep going": 0, "slow down": -1}[action] + slope_mod.get(self.state["slope_level"], 0)
        i = max(0, min(4, i + delta))
        self.state["HR_zone"] = zones[i]

    def _update_fatigue(self, action):
        base_fatigue = {"Z1": 0.1, "Z2": 0.3, "Z3": 0.5, "Z4": 0.7, "Z5": 1.0}[self.state["HR_zone"]]
        action_mod = {"accelerate": 0.2, "keep going": 0.0, "slow down": -0.1}[action]
        noise = random.uniform(-0.05, 0.05)
        self.fatigue_score += base_fatigue + action_mod + noise
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
        reward = 0
        if self.state["HR_zone"] == self.state["target_hr_zone"]:
            reward += 1
        elif abs(self._get_zone(self.state["HR_zone"]) - self._get_zone(self.state["target_hr_zone"])) == 1:
            reward -= 0.5
        else:
            reward -= 1

        if self.state["power_zone"] == self.state["target_power_zone"]:
            reward += 1
        elif abs(self._get_zone(self.state["power_zone"]) - self._get_zone(self.state["target_power_zone"])) == 1:
            reward -= 0.5
        else:
            reward -= 1

        if self.state["fatigue_level"] == "high":
            reward -= 2
        elif self.state["fatigue_level"] == "medium":
            reward -= 0.5

        if self.state["phase_label"] == "recover" and action == "accelerate":
            reward -= 1
        elif self.state["phase_label"] == "cooldown" and action != "slow down":
            reward -= 1

        return reward

    def _get_zone(self, zone):
        return {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}[zone]

    def _log_state(self, action, reward, done):
        if not self.verbose:
            return
        print(f"Second: {self.second} | Action: {action.upper()} | Reward: {reward} | Done: {done}")
        print(f"➤ Phase : {self.state['phase_label']} | Target HR Zone: {self.state['target_hr_zone']} | Target Power Zone: {self.state['target_power_zone']}")
        print(f"➤ Actual State : HR Zone: {self.state['HR_zone']} | Power Zone: {self.state['power_zone']} | Fatigue Level: {self.state['fatigue_level']} | Slope Level: {self.state['slope_level']}")
        print("--------------------------------------------------")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
