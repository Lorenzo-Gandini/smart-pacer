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

        # Scaling fisiologico: VO2/FTP → più FTP = meno fatica
        ftp_per_kg = self.ftp / self.weight
        ftp_factor = 1.0 / max(0.1, ftp_per_kg)
        fatigue_gain *= ftp_factor

        # Modificatore da tipo allenamento
        modifier = {
            "fartlek": 1.3,
            "interval": 1.2,
            "progressions": 1.1,
            "endurance": 1.0,
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
        # Get numerical values for zones
        hr_zone = self._get_zone(self.state["HR_zone"])
        power_zone = self._get_zone(self.state["power_zone"])
        target_hr = self._get_zone(self.state["target_hr_zone"])
        target_power = self._get_zone(self.state["target_power_zone"])
        phase = self.state["phase_label"]
        fatigue = self.state["fatigue_level"]
        
        # Determine athlete level based on FTP/kg
        ftp_per_kg = self.ftp / self.weight
        if ftp_per_kg > 5.5:
            athlete_level = "elite"
        elif ftp_per_kg > 4.0:
            athlete_level = "runner"
        else:
            athlete_level = "amatour"

        # 1. Zone Matching Reward (weighted 40% HR, 40% Power)
        hr_diff = abs(hr_zone - target_hr)
        power_diff = abs(power_zone - target_power)
        
        hr_reward = {
            0: 2.0,   # Perfect match
            1: 0.5,    # Slightly off
            2: -1.0,   # Moderate deviation
            3: -2.5,   # Significant deviation
            4: -4.0    # Extreme deviation
        }.get(hr_diff, -4.0)
        
        power_reward = {
            0: 2.0,
            1: 0.5,
            2: -1.0,
            3: -2.5,
            4: -4.0
        }.get(power_diff, -4.0)

        # 2. Fatigue Management (scaled by athlete level)
        fatigue_thresholds = {
            "elite": {"low": 5.0, "medium": 7.0},
            "runner": {"low": 4.0, "medium": 6.0},
            "amatour": {"low": 3.0, "medium": 5.0}
        }
        
        if self.fatigue_score <= fatigue_thresholds[athlete_level]["low"]:
            fatigue_penalty = 0.0
        elif self.fatigue_score <= fatigue_thresholds[athlete_level]["medium"]:
            fatigue_penalty = -1.0 * (self.fatigue_score - fatigue_thresholds[athlete_level]["low"])
        else:
            fatigue_penalty = -2.0 * (self.fatigue_score - fatigue_thresholds[athlete_level]["medium"])

        # 3. Physiological Coherence (HR-Power relationship)
        zone_diff = abs(hr_zone - power_zone)
        expected_diff = {
            "elite": 0.5,
            "runner": 1.0,
            "amatour": 1.5
        }[athlete_level]
        
        if zone_diff <= expected_diff:
            coherence_bonus = 1.0
        else:
            coherence_bonus = -1.0 * (zone_diff - expected_diff)

        # 4. Training Phase Adaptation
        phase_bonus = 0.0
        if phase == "warmup":
            if hr_zone > target_hr:
                phase_bonus = -2.0
            elif action == "accelerate" and hr_zone < target_hr:
                phase_bonus = 0.5
        elif phase == "push":
            if action == "accelerate" and hr_zone < target_hr:
                phase_bonus = 1.0
            elif action == "slow down" and hr_zone > target_hr:
                phase_bonus = 0.5
        elif phase in ["recover", "cooldown"]:
            if action == "slow down" and hr_zone > target_hr:
                phase_bonus = 1.0
            elif action == "accelerate":
                phase_bonus = -2.0

        # 5. Athlete Capacity Scaling
        zone_penalty = {
            1: 0.0,
            2: 0.0,
            3: 0.5,
            4: 1.5,
            5: 3.0
        }[hr_zone]
        
        capacity_factor = min(1.0, ftp_per_kg / 6.0)  # 6.0 w/kg is elite threshold
        capacity_adjustment = -zone_penalty * (1.0 - capacity_factor)

        # Combine all components with weights
        total_reward = (
            0.4 * hr_reward +
            0.4 * power_reward +
            0.3 * coherence_bonus +
            0.2 * phase_bonus +
            fatigue_penalty +
            capacity_adjustment
        )

        # Add small randomness to avoid perfect patterns
        total_reward += random.uniform(-0.1, 0.1)

        # Special cases
        # Penalize extreme overexertion for amateurs
        if athlete_level == "amatour" and hr_zone >= 4 and power_zone >= 4:
            total_reward -= 2.0
        
        # Bonus for elite maintaining high zones appropriately
        if athlete_level == "elite" and hr_zone == target_hr and power_zone == target_power and target_hr >= 4:
            total_reward += 1.0

        return total_reward

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
