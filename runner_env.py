import random
import json
import math
import numpy as np

ACTIONS = ['slow down', 'keep going', 'accelerate']

class RunnerEnv:
    def __init__(self, athlete_profile, training_plan, track_data=None, verbose=True):
        self.athlete = athlete_profile
        self.training = training_plan
        self.tracking = track_data or [] 
        self.verbose = verbose
        self.actions = ACTIONS
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
        # Reset the environment state to initial values
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
        # Validate action taken and proceed with the simulation step
        self._update_power_zone(action)
        self._update_hr_zone(action)
        self._update_fatigue(action)
        self._advance_segment()
        reward = self._compute_reward(action)
        done = self.second >= self.training["duration"] * 60
        self._log_state(action, reward, done)

        if not hasattr(self, '_fatigue_thresholds'):
            self._fatigue_thresholds = {"low": 33, "high": 67}  # percentiles for fatigue levels
            self._fatigue_values = []

        self._fatigue_values.append(self.fatigue_score)
        # low_th = np.percentile(self._fatigue_values, self._fatigue_thresholds["low"])
        # high_th = np.percentile(self._fatigue_values, self._fatigue_thresholds["high"])
        low_th = 35
        high_th = 70

        if self.fatigue_score >= high_th:
            self.state["fatigue_level"] = "high"
        elif self.fatigue_score >= low_th:
            self.state["fatigue_level"] = "medium"
        else:
            self.state["fatigue_level"] = "low"
        return self.state.copy(), reward, done

    def _update_power_zone(self, action): #THIS update logic is so easy?
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
        i = zones.index(self.state["power_zone"])
        if action == "accelerate" and i < 4:
            i += 1
        elif action == "slow down" and i > 0:
            i -= 1
        self.state["power_zone"] = zones[i]

    def _update_hr_zone(self, action): #THIS action is not used
        target = self._get_zone(self.state["power_zone"])
        delta = (target - self.hr_float) * 0.2
        self.hr_float = max(1.0, min(5.0, self.hr_float + delta))
        self.state["HR_zone"] = f"Z{round(self.hr_float)}"

    def _update_fatigue(self, action):
        ''' Update the fatigue score based on the current state and action taken by the athlete. 
        - Recovery/Cooldown: exponential + sigmoid decay with floor to simulate how fatigue naturally decreases during rest periods, with a minimum fatigue
        Are defined costants k and decay to control the rate of fatigue decay, and a floor to ensure fatigue does not drop below a certain level.
        - Warmup/Push: fatigue accumulates based on HR and Power zones, with adjustments for training type and athlete's FTP
        The fatigue score is capped between 0 and 10, with a label indicating low, medium, or high fatigue level. gain costants are defined to control how much fatigue accumulates based on the current HR zone, Power zone, and training session type (some are more fatigue demanding than others).
        '''

        hr_level = self._get_zone(self.state["HR_zone"])
        power_level = self._get_zone(self.state["power_zone"])
        phase = self.state["phase_label"]

        # Recovery/Cooldown
        if phase in ["recover","cooldown"]:
            k = 0.05 * self.fitness_factor
            decayed = self.fatigue_score * math.exp(-k)
            sig = 1/(1+math.exp(-10*(self.fatigue_score-5)))
            decay = 0.1 * self.fitness_factor * sig
            floor = 0.1 * self.fitness_factor
            self.fatigue_score = max(floor, decayed - decay)
        else:
            # Warmup/Push accumulation
            gain = {1:0.01,2:0.05,3:0.12,4:0.3,5:0.5}[hr_level]
            if phase == "push": gain *= 1.2
            if hr_level <= 2 and action == "slow down": gain -= 0.1
            if hr_level >= 4:
                self.time_in_high_zones += 1
            else:
                self.time_in_high_zones = max(0,self.time_in_high_zones-1)
            gain += 0.01 * self.time_in_high_zones
            if hr_level>=4 and power_level>=4:
                gain *= 1.3
            ftp_factor = 1.0/max(0.1, self.ftp/self.weight)
            gain *= ftp_factor
            gain *= {"fartlek":1.3,"interval":1.2,
                     "progressions":1.1,"endurance":1.0,
                     "recovery":0.7}.get(self.training_type,1.0)
            self.fatigue_score += gain + random.uniform(-0.02,0.02)
            self.fatigue_score *= self.fitness_factor
            self.fatigue_score = max(0,min(10,self.fatigue_score))

        # Update fatigue level label
        if self.fatigue_score <= 3:
            lvl = "low"
        elif self.fatigue_score <= 7:
            lvl = "medium"
        else:
            lvl = "high"
        self.state["fatigue_level"] = lvl

    def _advance_segment(self):
        ''' Advance to the next segment in the training plan, updating the state accordingly.'''
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
        ''' since some training segments can have sub-segments or repeat multiple times, this function expands them into a flat list of segments with their durations in seconds.'''

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
        ''' Compute the reward based on the current state, action taken, and athlete's performance.
        - ftp_per_kg: it's a score used to determine the athlete's level based on their FTP relative to their weight. Since FTP is a measure of the maximum power output an athlete can sustain, dividing it by weight gives a relative performance metric (if you are lighter and you have the seme FTP of a heavier athlete, you are more efficient).
        - Zone Matching Reward : based on the difference between current HR and Power zones and their target zones. Calculate the difference between current zones and target zones and assigna reward if they match. Each zones of hr and power have a different reward based on the difference (a large difference results in a negative reward).
        - Fatigue Penalty: based on the athlete's profiles, are defined different tresholds (an elite can manage better the fatigue than an amateur). based on them, a penalty is applied to the reward if the fatigue score exceeds certain levels. 
        - Physiological coherence : if the difference between HR and Power zones is within expected limits, a bonus is applied, otherwise a penalty is applied. An elite athlete should be able to maintain a smaller difference than an amateur. 
        - Phase-specific bonuses : different phases of the training session have different expected actions and rewards. For example, during the warmup phase, accelerating when below target HR is rewarded, while slowing down is penalized. In the push phase, accelerating when below target HR is rewarded (the goal is to reach that zone), while slowing down when above target HR is penalized (a behaviour that we expect from an amateur when suffers from a high fatigue).
        - Pacing-coherence with slope : This is a bonus or penalty based on the slope of the track and the action taken.
        - Capacity scaling : The capacity adjustment is a penalty that represents the athlete's ability to sustain high power outputs relative to their FTP. for each hr zone a penalty is defined, which is scaled by the athlete's FTP per kg. We do this because a lighter athlete with the same FTP as a heavier athlete is more efficient, so they should have a lower penalty.
        - Dynamic tolerance & funnel : The tolerance shrinks as the session progresses, and a funnel bonus is applied when the athlete is in the target zones. The funnel bonus is a bonus that represents the athlete's ability to maintain the target zones as the session progresses. 
        # These values are want to emulate the fact that an athlete can maintain a target HR and Power zone for a longer time as they progress through the session, but the tolerance shrinks as they get closer to the end of the session.
        - Combine and randomize the final reward.
        '''

        hr_zone = self._get_zone(self.state["HR_zone"])
        power_zone = self._get_zone(self.state["power_zone"])
        target_hr = self._get_zone(self.state["target_hr_zone"])
        target_power = self._get_zone(self.state["target_power_zone"])
        phase = self.state["phase_label"]
        slope = self.state["slope_level"]
        
        # Athlete level by FTP per kg
        #THIS check if in the athletes profiles the ratios are maintened
        ftp_per_kg = self.ftp / self.weight
        if ftp_per_kg > 5.5:
            athlete_level = "elite"
        elif ftp_per_kg > 4.0:
            athlete_level = "runner"
        else:
            athlete_level = "amateur"

        # Zone Matching Reward
        hr_diff = abs(hr_zone - target_hr)
        power_diff = abs(power_zone - target_power)
        hr_reward = {0:2.0,1:0.5,2:-1.0,3:-2.5,4:-4.0}.get(hr_diff, -4.0)
        power_reward = {0:2.0,1:0.5,2:-1.0,3:-2.5,4:-4.0}.get(power_diff, -4.0)

        # Fatigue penalty
        thresholds = {"elite":{"low":5.0,"medium":7.0},
                    "runner":{"low":4.0,"medium":6.0},
                    "amateur":{"low":3.0,"medium":5.0}}[athlete_level]
        fat = self.fatigue_score
        if fat <= thresholds["low"]:
            fatigue_penalty = 0.0
        elif fat <= thresholds["medium"]:
            fatigue_penalty = -1.0 * (fat - thresholds["low"])
        else:
            fatigue_penalty = -2.0 * (fat - thresholds["medium"])

        # Physiological coherence
        diff = abs(hr_zone - power_zone)
        expected = {"elite":0.5, "runner":1.0, "amateur":1.5}[athlete_level]
        coherence_bonus = 1.0 if diff <= expected else -1.0 * (diff - expected)

        # Phase-specific bonuses
        phase_bonus = 0.0
        if phase == "warmup":
            if action == "slow down":
                phase_bonus = -1.0
            elif hr_zone < target_hr and action == "accelerate":
                phase_bonus = 0.5
            elif hr_zone == target_hr and action == "keep going":
                phase_bonus = 1.0
        elif phase == "push":
            if action == "accelerate" and hr_zone < target_hr:
                phase_bonus = 1.0
            elif action == "slow down" and hr_zone > target_hr:
                phase_bonus = 0.5
        elif phase in ["recover","cooldown"]:
            if action == "slow down" and hr_zone > target_hr:
                phase_bonus = 1.0
            elif action == "accelerate":
                phase_bonus = -2.0

        # Pacing-coherence with slope.
        slope_penalty = 0.0
        if slope == 'uphill' and action == 'accelerate':
            slope_penalty = -2.0
        elif slope == 'downhill' and action == 'slow down':
            slope_penalty = -0.5

        # Capacity scaling
        cap_pen = {1:0.0,2:0.0,3:0.5,4:1.5,5:3.0}[hr_zone]
        cap_factor = min(1.0, ftp_per_kg/6.0)
        capacity_adj = -cap_pen * (1.0 - cap_factor)

        # Dynamic funnel bonus
        total_steps = self.training["duration"]*60
        prog = min(1.0, self.second/total_steps)
        tol = 1 if prog < 0.5 else 0
        in_zone = (hr_diff <= tol and power_diff <= tol)
        if not hasattr(self, '_prev_in_zone'):
            self._prev_in_zone = False
        if not self._prev_in_zone and in_zone:
            funnel_bonus = 2.0
        elif self._prev_in_zone and in_zone:
            funnel_bonus = 0.5
        else:
            funnel_bonus = 0.0
        self._prev_in_zone = in_zone

        # Final reward calculation
        total = (0.4*hr_reward + 0.4*power_reward + 0.3*coherence_bonus +
                0.2*phase_bonus + fatigue_penalty + capacity_adj +
                slope_penalty + funnel_bonus)

        fatigue_decay = 1.0 - min(self.fatigue_score / 200.0, 0.4)  # max decay -40%
        total *= fatigue_decay

        # Random noise to the reward to simulate real-world variability
        total += random.uniform(-0.1, 0.1)
        return total

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
