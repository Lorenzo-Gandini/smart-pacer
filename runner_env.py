import random
import json

class RunnerEnv:
    def __init__(self, athlete_profile, training_plan, verbose=True):
        self.athlete = athlete_profile
        self.training = training_plan
        self.verbose = verbose
        self.reset()

    # Initialize the environment resetting the state of the athlete
    def reset(self):
        self.minute = 0
        self.fatigue_score = 0.0
        self.expanded_plan = self._expand_training_segments(self.training["segments"])
        self.state = {
            "HR_zone": "Z1",
            "power_zone": "Z1",
            "fatigue_level": "low",
            "segment_index": 0,
            "phase_label": "warmup",
            "phase_label": self.expanded_plan[0]["phase"],
            "target_hr_zone": self.expanded_plan[0]["target_hr_zone"],
            "target_power_zone": self.expanded_plan[0]["target_power_zone"],
            "slope_level": "flat",
        }

        self.expanded_plan = self._expand_training_segments(self.training["segments"])

        return self.state

    # Define the action space
    def step(self, action):
        self._update_power_zone(action)
        self._update_hr_zone(action)
        self._update_fatigue(action)
        self._advance_segment()
        reward = self._compute_reward(action)

        done = self.minute >= self.training["duration"] #the training is over when the duration is reached
        
        self._log_state(action, reward, done)

        return self.state.copy(), reward, done

    #TO DO : update the methods that updates the hr zones, power zones and fatigue. Up to now they are not so realistic.

    def _update_power_zone(self, action):
        ''' Update the power zone based on the action taken. '''
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
        i = zones.index(self.state["power_zone"])
        if action == "accelerate" and i < 4:
            i += 1
        elif action == "slow down" and i > 0:
            i -= 1
        self.state["power_zone"] = zones[i]

    def _update_hr_zone(self, action):
        ''' Update the heart rate zone based on the slope of the ground, the action taken and the current heart zone.'''
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
        i = zones.index(self.state["HR_zone"])
        slope_mod = {"flat": 0, "uphill": 1, "steep_uphill": 2, "downhill": -1, "steep_down": -2}
        delta = {"accelerate": 1, "keep going": 0, "slow down": -1}[action] + slope_mod.get(self.state["slope_level"], 0)
        i = max(0, min(4, i + delta))
        self.state["HR_zone"] = zones[i]

    def _update_fatigue(self, action):
        ''' Update the fatigue score based on the action taken and the current heart rate zone. To make it more realistic, we add a small random noise to the fatigue score.'''
        base_fatigue = {"Z1": 0.1, "Z2": 0.3, "Z3": 0.5, "Z4": 0.7, "Z5": 1.0}[self.state["HR_zone"]]
        action_mod = {"accelerate": 0.2, "keep going": 0.0, "slow down": -0.1}[action]
        noise = random.uniform(-0.05, 0.05)
        self.fatigue_score += base_fatigue + action_mod + noise
        self.fatigue_score = max(0, min(10, self.fatigue_score))
        if self.fatigue_score <= 3:
            level = "low"
        elif self.fatigue_score <= 7:
            level = "medium"
        else:
            level = "high"
        self.state["fatigue_level"] = level

    def _advance_segment(self):
        ''' Advance to the next segment of the training plan. '''
        self.minute += 1
        if self.minute < len(self.expanded_plan):
            current = self.expanded_plan[self.minute]
            self.state["segment_index"] = self.minute
            self.state["phase_label"] = current["phase"]
            self.state["target_hr_zone"] = current["target_hr_zone"]
            self.state["target_power_zone"] = current["target_power_zone"]

    def _expand_training_segments(self, segments):
        expanded = []
        for segment in segments:
            repeat = segment.get("repeat", 1)
            if "sub_segments" in segment:
                for _ in range(repeat):
                    for sub in segment["sub_segments"]:
                        for _ in range(int(sub["duration_min"])):
                            expanded.append(sub)
            else:
                for _ in range(repeat):
                    for _ in range(int(segment["duration_min"])):
                        expanded.append(segment)
        return expanded
    
    def _compute_reward(self, action):
        reward = 0

        # HR Zone
        if self.state["HR_zone"] == self.state["target_hr_zone"]:
            reward += 1
        elif abs(self._get_zone(self.state["HR_zone"]) - self._get_zone(self.state["target_hr_zone"])) == 1:
            reward -= 0.5
        else:
            reward -= 1

        # Power Zone
        if self.state["power_zone"] == self.state["target_power_zone"]:
            reward += 1
        elif abs(self._get_zone(self.state["power_zone"]) - self._get_zone(self.state["target_power_zone"])) == 1:
            reward -= 0.5
        else:
            reward -= 1

        # Fatigue
        if self.state["fatigue_level"] == "high":
            reward -= 2
        elif self.state["fatigue_level"] == "medium":
            reward -= 0.5

        #If the action taken by the athlete is not consistent with the action of the training plan, a penalty is applied
        if self.state["phase_label"] == "recover" and action == "accelerate":
            reward -= 1
        elif self.state["phase_label"] == "cooldown" and action != "slow down":
            reward -= 1

        return reward
    
    def _get_zone(self, zone):
        ''' Get the level of the zone. '''
        return {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}[zone]
    
    def _log_state(self, action, reward, done):
        if not self.verbose:
            return
        print(f"Minute: {self.minute} | Action: {action.upper()} | Reward: {reward} | Done: {done}")
        print(f"➤ Phase : {self.state['phase_label']} | Target HR Zone: {self.state['target_hr_zone']} | Target Power Zone: {self.state['target_power_zone']}")
        print(f"➤ Actual State : {self.state['HR_zone']} | HR Zone: {self.state['HR_zone']} | Power Zone: {self.state['power_zone']} | Fatigue Level: {self.state['fatigue_level']} | Slope Level: {self.state['slope_level']}")
        print("--------------------------------------------------")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
