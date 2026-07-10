"""
trains the SAC agent in a tiny 2D sim
cat moves with a state machine
(wander -> stalk -> pounce -> flee)
no real camera needed

output:
    src/shared/mouse_policy.pt

usage:
    `python train/train_sac.py`

TODO: train longer (500k+ steps) if avg_dist isn't settling in the play zone
TODO: try curriculum learning (start cat closer, increase distance over time)
TODO: export policy to onnx for faster inference on pi
TODO: fine tune the actor directly on real transitions, currently using real videos / logs to recalibrate the sim
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) # for importing from shared/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

from shared.actor import Actor, OBS_DIM, ACT_DIM, HIDDEN
from train.train_config import (
    TOTAL_STEPS, BUFFER_SIZE, BATCH, LR, GAMMA, TAU, LEARN_START,
    LOG_ALPHA_CLAMP_MIN, GRAD_CLIP_MAX_NORM, TARGET_ENTROPY, LOG_WINDOW_SIZE,
    MAX_V, MAX_OMEGA,
    SIM_DT, WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX,
    OBS_DIST_SCALE, VISIBILITY_RANGE, MAX_EP_STEPS, OUTPUT_DIM,
    STALK_ENTER_DIST_RANGE, STALK_EXIT_BUFFER, STALK_STEPS_REQ_RANGE,
    POUNCE_SPEED_RANGE, WANDER_SPEED_MAX_RANGE,
    CAT_START_VX_RANGE, CAT_START_VY_RANGE,
    POUNCE_MISS_DIST, WANDER_ACCEL_NOISE,
    CLOSE_DIST, LEAVE_ENTER_STEPS, LEAVE_SPEED, LEAVE_EXIT_DIST, LEAVE_MAX_STEPS,
    WOBBLE_FREQ_HZ_RANGE, WOBBLE_AMP_RANGE,
    CAT_STATIONARY_SPEED_THRESH,
    CAPTURE_DIST, DANGER_DIST, PLAY_DIST_HI, APPROACH_DIST,
    CAPTURE_FINAL_REWARD, CAPTURE_PENALTY, DANGER_BASE_PENALTY,
    PLAY_ZONE_REWARD, TOO_FAR_PENALTY, DODGE_BONUS, REWARD_SCALER,
    MOUSE_TEASE_SPEED_THRESH, TEASE_BONUS, TEASE_STATIONARY_STEPS,
    POUNCE_FREEZE_PENALTY_BASE, POUNCE_FREEZE_SCALE_DIVISOR,
    MAX_CAPTURE_COUNT,
    UNPREDICTABILITY_WINDOW, UNPREDICTABILITY_BONUS, UNPREDICTABILITY_STD_NORM,
    SMOOTHNESS_PENALTY_WEIGHT,
    VISIBILITY_BONUS,
    ENERGY_COST_WEIGHT,
    COVERAGE_GRID_SIZE, COVERAGE_BONUS,
    WALL_MARGIN, WALL_PENALTY,
    TIREDNESS_RATE, TIREDNESS_RECOVERY_RATE, MAX_TIREDNESS,
    TIREDNESS_SPEED_SCALE, TIREDNESS_POST_REVIVE,
    ENGAGEMENT_BONUS,
    PLAY_DEAD_DURATION, PLAY_DEAD_COOLDOWN,
    SURVIVAL_REWARD,
    FREEZE_SPEED_THRESHOLD, FREEZE_PENALTY,
)


class Critic(nn.Module):
    """
    double Q network, training only, not deployed to pi
    """
    def __init__(self):
        super().__init__()
        def _q():
            return nn.Sequential(
                nn.Linear(OBS_DIM + ACT_DIM, HIDDEN), nn.ReLU(), # torch randomly seeds each layer
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, OUTPUT_DIM),
            )
        self.q1, self.q2 = _q(), _q()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1) # concatenates along dim
        # returns q values directly because no random sampling from distribution is involved in critic
        # returns two and not just a min of the two because we need both to train both networks in mse_loss with different q's
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap) # pops overflow automatically, we never need to pop manually

    def push(self, *transition): # takes many arguments and bundles as one tuple
        self.buf.append(transition) # appends tuple as single item

    def sample(self, n):
        batch = random.sample(self.buf, n)

        # *batch unpacks batch of tuples into zip
        # zip groups same index in tuples to new tuples
        # make each new tuple into its own tensor, each with batch size length
        # return an array of tensors, length is amount of transitions
        return [torch.FloatTensor(np.array(x)) for x in zip(*batch)]

    def __len__(self):
        return len(self.buf)


class MouseSACAgent:
    """
    wraps all SAC components: actor, double critic, target critic,
    three adam optimizers, and learnable entropy coefficient log_alpha

    actor lives in shared/actor.py (also used on pi, don't modify it here)
    critic is training-only
    """

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        # make copy of critic to solve moving target problem
        # the target network is updated by critic with TAU, not an optimizer
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict()) # replace random weights so this is an identical copy of self.critic

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        # log_alpha sets how much actor cares about the entropy (log_std sets actual entropy)
        # is a parameter because we need to constantly adjust it
        # log_alpha scales how much a_loss pushes log_std up / down, based on whether current entropy (lp, driven by log_std) is above or below TARGET_ENTROPY
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, obs_np, explore=True):
        """
        obs_np: np.ndarray (OBS_DIM,)
        returns np.ndarray (ACT_DIM,) in [-1, 1]
        explore=True: stochastic (used during training)
        explore=False: greedy (used for eval)
        """
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0) # convert np -> tensor and add batch dim (1 obs in batch)
        with torch.no_grad(): # don't need to build autograd computation graph (needed in backprop in mse loss), we're not training just deciding what to do next, it's required in update
            if explore:
                action, _ = self.actor.get_action(obs_t)
            else:
                action = self.actor.get_deterministic_action(obs_t)
        # squeeze() w/o args removes every dim of size 1
        return action.squeeze().numpy()

    def update(self, batch):
        """
        one SAC gradient step.
        batch: [Observation, Action, Reward, NextObservation, Done] FloatTensors from ReplayBuffer.sample()
        """
        O, A, R, NO, D = batch
        alpha = self.log_alpha.exp().detach() # detaches this tensor from autograd graph (gets rid of computation history, treated like a constant)

        # critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(NO) # sample next action from current policy
            q1_target, q2_target = self.critic_target(NO, next_action) # get Q values from target network (more stable)
            # target = reward + scaling factor * (future value - entropy)
            # (1-D) to get only valid steps
            target = R + GAMMA * (1 - D) * (torch.min(q1_target, q2_target) - alpha * next_log_prob)

        q1, q2 = self.critic(O, A) # Q values for the actions we actually took
        self.critic_opt.zero_grad() # zeros optimizer in case of fine tuning nn (like one layer, dont wanna zero everything)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q1_loss.backward() # computes and accumulates new gradients on old one (in entire critic parameter space)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        self.critic_opt.step()

        # actor update
        new_action, log_prob = self.actor.get_action(O) # sample actions from current obs
        q1n, q2n = self.critic(O, new_action) # how good does critic thinks current actions are
        # want to minimize a_loss -> minimize -Q -> maximize Q, with entropy, then take mean of batch
        a_loss = (alpha * log_prob - torch.min(q1n, q2n)).mean() # although this uses critic's weights, doesnt matter in terms of autograd graph bc actor_opt is only constructed with actor parameters
        self.actor_opt.zero_grad()
        a_loss.backward() # retain_graph = False by default, so after this previously built autograd graph is gone
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        self.actor_opt.step()

        # log_alpha (entropy) update: auto-tune how much we weight exploration vs exploitation
        # if entropy is too low (log_prob + TARGET_ENTROPY > 0), increase log_alpha to encourage more exploration
        al_loss = -(self.log_alpha * (log_prob + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad()
        al_loss.backward()
        self.alpha_opt.step()
        # clamp log_alpha so log_alpha never goes below exp(-3) ~= 0.05
        # without this the agent stops exploring completely around episode 40 and gets stuck
        with torch.no_grad():
            self.log_alpha.clamp_(min=LOG_ALPHA_CLAMP_MIN)

        # slowly blend target critic toward current critic (TAU=0.005 = 0.5% per step)
        # using a slowly-updated target makes training much more stable than updating it directly
        for s, t in zip(self.critic.parameters(), self.critic_target.parameters()):
            t.data.copy_(TAU * s.data + (1 - TAU) * t.data)

        return {
            "critic_loss": (q1_loss + q2_loss).item(), # just for rough gage of whether model is learning
            "actor_loss": a_loss.item(), # .item() converts 0d tensor -> float
            "alpha_loss": al_loss.item(),
            "log_alpha": self.log_alpha.exp().item(),
        }

    def save(self, path):
        """
        save actor weights only (critic stays on the laptop)
        """
        torch.save(self.actor.state_dict(), path)
        print(f"policy saved -> {path}")


class CatSimEnv:
    """
    2D sim with a multi-state cat and a mouse that gets tired, plays dead,
    and earns rewards for engaging, unpredictable, smooth movement.

    obs: [dist, angle, visible, cat_vx, cat_vy, sensor_front, mouse_speed]
    act: [v_raw, omega_raw] in [-1, 1]

    # TODO: add more realistic cat behaviors (circling, fake-outs, etc.)
    # TODO: add multiple cat sim to train for scenarios with multiple cats in view
    """

    def __init__(self): # gets called once
        self.cat_vx = self.cat_vy = 0.0
        self.mouse_speed = 0.0
        self.cat_state = "wandering"
        self.stalk_steps = self.stationary_steps = self.capture_count = self.close_steps = self.leave_steps = 0

        self.tiredness = 0.0
        self.effective_max_v = MAX_V
        self.playing_dead = False
        self.play_dead_timer = 0
        self.play_dead_cooldown_timer = 0
        self.action_history = deque(maxlen=UNPREDICTABILITY_WINDOW)
        self.prev_action = None
        self.current_action = None
        self.visited_cells = set()
        self.just_visited_new_cell = False

        self.reset()

    # per episode domain randomization of FSM thresholds + wobble
    def _sample_cat_params(self):
        self.stalk_enter_dist = np.random.uniform(*STALK_ENTER_DIST_RANGE) # unpacks tuple into two arguments
        self.stalk_exit_dist = self.stalk_enter_dist + STALK_EXIT_BUFFER
        self.stalk_steps_req = np.random.randint(*STALK_STEPS_REQ_RANGE)
        self.pounce_speed = np.random.uniform(*POUNCE_SPEED_RANGE)
        self.wander_speed_max = np.random.uniform(*WANDER_SPEED_MAX_RANGE)
        self.wobble_freq_hz = np.random.uniform(*WOBBLE_FREQ_HZ_RANGE)
        self.wobble_amp = np.random.uniform(*WOBBLE_AMP_RANGE)

        # random phase offset so the wobble doesn't always start at the same point in its cycle when stalking begins
        self.wobble_phase0 = np.random.uniform(0, 2 * np.pi)

    def reset(self): # gets called every episode
        # sample this episode's randomized FSM thresholds + wobble params
        self._sample_cat_params()

        # randomize cat starting xy
        dist = np.random.uniform(PLAY_DIST_HI, APPROACH_DIST)
        angle = np.random.uniform(0, 2 * np.pi)
        self.cat_x = dist * OBS_DIST_SCALE * np.sin(angle)
        self.cat_y = dist * OBS_DIST_SCALE * np.cos(angle)

        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.mouse_heading = 0.0
        self.mouse_speed = 0.0
        self.cat_vx = np.random.uniform(*CAT_START_VX_RANGE)
        self.cat_vy = np.random.uniform(*CAT_START_VY_RANGE)
        self.cat_state = "wandering"
        self.stalk_steps = 0
        self.stationary_steps = 0
        self.capture_count = 0
        self.close_steps = 0
        self.leave_steps = 0
        self.step_n = 0

        self.tiredness = 0.0
        self.effective_max_v = MAX_V
        self.playing_dead = False
        self.play_dead_timer = 0
        self.play_dead_cooldown_timer = 0
        self.action_history.clear()
        self.prev_action = None
        self.current_action = None
        self.visited_cells = set()
        self.just_visited_new_cell = False

        return self._obs()

    def _obs(self): # returned in step()
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0)

        world_angle = np.arctan2(dx, dy) # angle to cat in world frame, positive = cat is to the right of where we're facing (matches detect.py)
        # subtract mouse heading to get angle relative to which way the robot is pointing
        # the +pi) % 2pi - pi trick wraps the result into [-pi, pi] cleanly
        rel_angle = (world_angle - self.mouse_heading + np.pi) % (2 * np.pi) - np.pi
        angle = np.clip(rel_angle / np.pi, -1.0, 1.0) # normalize to [-1, 1]

        visible = float(dist < VISIBILITY_RANGE)

        return np.array([dist, angle, visible, self.cat_vx, self.cat_vy, 1.0, self.mouse_speed], dtype=np.float32) # TODO: change 1.0 placeholder to be actual sensor stuff

    def _update_cat(self, dist):
        """
        advance cat state machine and move cat
        returns cat_speed
        """
        self.just_dodged = False

        # track how long mouse has been uncomfortably close
        if dist < CLOSE_DIST:
            self.close_steps += 1
        else:
            self.close_steps = 0

        # cat leave overrides everything if mouse has been too close too long
        if self.cat_state != "leaving" and self.close_steps >= LEAVE_ENTER_STEPS:
            self.cat_state = "leaving"
            self.leave_steps = 0
            self.stalk_steps = 0

        if self.cat_state == "leaving":
            self.leave_steps += 1
            if dist > LEAVE_EXIT_DIST or self.leave_steps >= LEAVE_MAX_STEPS:
                self.cat_state = "wandering"
                self.close_steps = 0
                self.leave_steps = 0

        elif self.cat_state == "wandering":
            if dist < self.stalk_enter_dist:
                self.cat_state = "stalking"
                self.stalk_steps = 0

        elif self.cat_state == "stalking":
            if dist > self.stalk_exit_dist:
                self.cat_state = "wandering"
                self.stalk_steps = 0
            elif self.stalk_steps >= self.stalk_steps_req:
                self.cat_state = "pouncing"
                self.stalk_steps = 0
                self.pounce_entry_stationary_steps = self.stationary_steps  # remember how locked in cat was right before pouncing
            else:
                self.stalk_steps += 1

        elif self.cat_state == "pouncing":
            if dist < CAPTURE_DIST:
                self.cat_state = "wandering"
                self.stalk_steps = 0
            elif dist > POUNCE_MISS_DIST:
                self.cat_state = "wandering"
                self.stalk_steps = 0
                self.just_dodged = True

        # set velocity based on current state
        dx_to_mouse = self.mouse_x - self.cat_x
        dy_to_mouse = self.mouse_y - self.cat_y
        dist_to_mouse = np.hypot(dx_to_mouse, dy_to_mouse) + 1e-6
        dir_x = dx_to_mouse / dist_to_mouse  # unit vector toward mouse, computed ONCE
        dir_y = dy_to_mouse / dist_to_mouse

        if self.cat_state == "wandering":
            self.cat_vx = np.clip(self.cat_vx + np.random.uniform(-WANDER_ACCEL_NOISE, WANDER_ACCEL_NOISE), -self.wander_speed_max, self.wander_speed_max)
            self.cat_vy = np.clip(self.cat_vy + np.random.uniform(-WANDER_ACCEL_NOISE, WANDER_ACCEL_NOISE), -self.wander_speed_max, self.wander_speed_max)
        elif self.cat_state == "leaving":
            self.cat_vx = -dir_x * LEAVE_SPEED
            self.cat_vy = -dir_y * LEAVE_SPEED
        elif self.cat_state == "stalking": # before pouncing
            perp_x, perp_y = -dir_y, dir_x # perpendicular to the approach direction

            # actual s passed = steps * SIM_DT, rad/s = 2*pi*f
            phase = self.wobble_phase0 + self.stalk_steps * SIM_DT * 2 * np.pi * self.wobble_freq_hz
            wobble = np.sin(phase) * self.wobble_amp

            self.cat_vx = perp_x * wobble
            self.cat_vy = perp_y * wobble
        else: # pouncing
            self.cat_vx = dir_x * self.pounce_speed
            self.cat_vy = dir_y * self.pounce_speed

        # move cat, stop at walls
        new_x = self.cat_x + self.cat_vx * SIM_DT
        new_y = self.cat_y + self.cat_vy * SIM_DT
        clipped_x = np.clip(new_x, WORLD_X_MIN, WORLD_X_MAX)
        clipped_y = np.clip(new_y, WORLD_Y_MIN, WORLD_Y_MAX)
        if clipped_x != new_x: self.cat_vx *= 0.0
        if clipped_y != new_y: self.cat_vy *= 0.0
        self.cat_x, self.cat_y = clipped_x, clipped_y

        cat_speed = np.hypot(self.cat_vx, self.cat_vy)
        if cat_speed < CAT_STATIONARY_SPEED_THRESH:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0

    def _wall_closeness(self):
        closest = min(
            self.mouse_x - WORLD_X_MIN,
            WORLD_X_MAX - self.mouse_x,
            self.mouse_y - WORLD_Y_MIN,
            WORLD_Y_MAX - self.mouse_y,
        )
        if closest < WALL_MARGIN:
            return 1.0 - (closest / WALL_MARGIN) # closer you are, higher the return
        return 0.0

    def _update_coverage(self):
        gx = int((self.mouse_x - WORLD_X_MIN) / COVERAGE_GRID_SIZE)
        gy = int((self.mouse_y - WORLD_Y_MIN) / COVERAGE_GRID_SIZE)
        cell = (gx, gy)
        self.just_visited_new_cell = cell not in self.visited_cells # used in reward for novelty
        self.visited_cells.add(cell)

    def _update_tiredness(self):
        speed_ratio = self.mouse_speed / MAX_V
        self.tiredness += TIREDNESS_RATE * speed_ratio
        self.tiredness -= TIREDNESS_RECOVERY_RATE * (1.0 - speed_ratio) # the slower you ar the faster you recover
        self.tiredness = np.clip(self.tiredness, 0.0, MAX_TIREDNESS)
        self.effective_max_v = MAX_V * (1.0 - self.tiredness * TIREDNESS_SPEED_SCALE) # max v of untired parts of self

    def _reward(self, dist, captured): # returned in step()
        """
        zone based reward with teasing, pounce freeze, unpredictability,
        smoothness, visibility, energy cost, coverage, wall avoidance,
        engagement, survival, tiredness, and play-dead mechanics
        """
        # no reward (neutral) while playing dead when not captured, like a sleep
        if self.playing_dead and not captured:
            return 0.0

        r = 0.0

        # base zone rewards
        if captured:
            # positive reward if exceeded max capture count per episode to make sure you're letting the cat win once in a while to keep it engaged
            r = CAPTURE_FINAL_REWARD if self.capture_count >= MAX_CAPTURE_COUNT else CAPTURE_PENALTY
        elif dist < DANGER_DIST:
            r = DANGER_BASE_PENALTY + (self.mouse_speed / MAX_V)
        elif dist < PLAY_DIST_HI:
            r = PLAY_ZONE_REWARD
        elif dist < APPROACH_DIST:
            t = (APPROACH_DIST - dist) / (APPROACH_DIST - PLAY_DIST_HI)
            r = float(np.clip(t, 0.0, 1.0))
        else:
            r = TOO_FAR_PENALTY

        # dodge bonus
        if self.just_dodged:
            r += DODGE_BONUS

        # tease bonus (cat is still and watching, mouse is actively moving near it)
        if self.stationary_steps >= TEASE_STATIONARY_STEPS and self.mouse_speed > MOUSE_TEASE_SPEED_THRESH:
            r += TEASE_BONUS

        # freeze during pounce penalty
        if self.cat_state == "pouncing" and self.mouse_speed < MOUSE_TEASE_SPEED_THRESH:
            scale = min(self.pounce_entry_stationary_steps / POUNCE_FREEZE_SCALE_DIVISOR, 1.0)
            r -= POUNCE_FREEZE_PENALTY_BASE * (0.5 + 0.5 * scale)

        # unpredictability bonus
        if len(self.action_history) >= UNPREDICTABILITY_WINDOW:
            recent = np.array(self.action_history)
            action_std = np.std(recent, axis=0).mean() # mean collapses std of v and omega into single scalar
            r += UNPREDICTABILITY_BONUS * min(action_std / UNPREDICTABILITY_STD_NORM, 1.0)

        # smoothness penalty (penalize jerky acceleration)
        if self.prev_action is not None and self.current_action is not None:
            jerk = np.linalg.norm(self.current_action - self.prev_action) # euclidean norm (sqrt(sum of squares))
            r -= SMOOTHNESS_PENALTY_WEIGHT * jerk

        # visibility maintenance bonus
        if dist < VISIBILITY_RANGE:
            r += VISIBILITY_BONUS

        # energy / speed cost (to prevent going max speed all the time)
        r -= ENERGY_COST_WEIGHT * (self.mouse_speed / MAX_V)

        # coverage / novelty bonus
        if self.just_visited_new_cell:
            r += COVERAGE_BONUS

        # anti wall hugging
        wall_closeness = self._wall_closeness()
        if wall_closeness > 0:
            r -= WALL_PENALTY * wall_closeness

        # engagement bonus (cat is actively interested)
        if self.cat_state in ("stalking", "pouncing"):
            r += ENGAGEMENT_BONUS

        # survival reward
        r += SURVIVAL_REWARD

        # general freeze penalty (outside of play dead)
        if not self.playing_dead and self.mouse_speed < FREEZE_SPEED_THRESHOLD:
            r -= FREEZE_PENALTY

        return float(r)

    def step(self, action):
        v_raw, omega_raw = float(action[0]), float(action[1])

        # track actions for unpredictability and smoothness
        action_np = np.array([v_raw, omega_raw])
        self.prev_action = self.current_action
        self.current_action = action_np
        self.action_history.append(action_np)

        # tick play dead cooldown
        if self.play_dead_cooldown_timer > 0:
            self.play_dead_cooldown_timer -= 1

        # if playing dead, force zero movement (mimics mouse dying)
        if self.playing_dead:
            self.play_dead_timer -= 1
            if self.play_dead_timer <= 0:
                self.playing_dead = False
                self.play_dead_cooldown_timer = PLAY_DEAD_COOLDOWN
                self.tiredness = TIREDNESS_POST_REVIVE
                self.effective_max_v = MAX_V * (1.0 - self.tiredness * TIREDNESS_SPEED_SCALE)
            v = 0.0
            omega = 0.0
        else:
            self._update_tiredness()
            v = v_raw * self.effective_max_v
            omega = omega_raw * MAX_OMEGA

        # really order doesn't matter since this hapens very quickly and loops
        # move the mouse first so the cat reacts to its actual current step position
        self.mouse_heading += omega * SIM_DT # rotate first
        self.mouse_x += v * np.sin(self.mouse_heading) * SIM_DT # sin / cos converts heading to x / y movement
        self.mouse_y += v * np.cos(self.mouse_heading) * SIM_DT
        self.mouse_speed = abs(v) # track scalar speed for the obs and teasing reward

        self._update_coverage()

        # dist after mouse moves but before cat moves, this is what the cat reacts to
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = float(np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0))
        self._update_cat(dist)

        # dist after both have moved, this is what capture / reward use
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist_post = float(np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0))

        captured = dist_post < CAPTURE_DIST
        if captured:
            self.capture_count = min(self.capture_count + 1, MAX_CAPTURE_COUNT)
            if not self.playing_dead and self.play_dead_cooldown_timer <= 0:
                self.playing_dead = True
                self.play_dead_timer = PLAY_DEAD_DURATION

        self.step_n += 1

        reward = self._reward(dist_post, captured)
        done = (self.step_n >= MAX_EP_STEPS) or (self.capture_count >= MAX_CAPTURE_COUNT and captured)

        obs = self._obs()
        info = {
            "cat_state": self.cat_state,
            "capture_count": self.capture_count,
            "dist": dist_post,
            "captured": captured,
            "mouse_speed": self.mouse_speed,
            "close_steps": self.close_steps,
            "tiredness": self.tiredness,
            "playing_dead": self.playing_dead,
        }
        return obs, reward, done, info


def _reset_window():
    """
    fresh accumulator for the per-window episode log
    """
    return {
        "reward_sum": 0.0,
        "dist_sum": 0.0,
        "dist_count": 0,
        "capture_count": 0,
        "steps_in_capture_zone": 0,
        "steps_in_danger_zone": 0,
        "steps_in_play_zone": 0,
        "steps_in_approach_zone": 0,
        "steps_in_far_zone": 0,
        "steps_cat_wandering": 0,
        "steps_cat_stalking": 0,
        "steps_cat_pouncing": 0,
        "steps_cat_leaving": 0,
        "critic_loss_sum": 0.0,
        "actor_loss_sum": 0.0,
        "loss_update_count": 0,
        "linear_action_sum": 0.0,
        "angular_action_sum": 0.0,
        "action_count": 0,
        "tiredness_sum": 0.0,
        "play_dead_steps": 0,
    }


def _accumulate_stats(window, action, reward, info):
    """
    update the logging window with this step's results
    """
    dist = info["dist"]
    cat_state = info["cat_state"]

    window["reward_sum"] += reward
    window["dist_sum"] += dist
    window["dist_count"] += 1
    window["linear_action_sum"] += abs(float(action[0]))
    window["angular_action_sum"] += abs(float(action[1]))
    window["action_count"] += 1

    if info["captured"]: window["steps_in_capture_zone"] += 1
    elif dist < DANGER_DIST: window["steps_in_danger_zone"] += 1
    elif dist < PLAY_DIST_HI: window["steps_in_play_zone"] += 1
    elif dist < APPROACH_DIST: window["steps_in_approach_zone"] += 1
    else: window["steps_in_far_zone"] += 1

    if cat_state == "wandering": window["steps_cat_wandering"] += 1
    elif cat_state == "stalking": window["steps_cat_stalking"] += 1
    elif cat_state == "pouncing": window["steps_cat_pouncing"] += 1
    elif cat_state == "leaving": window["steps_cat_leaving"] += 1

    if info["captured"]:
        window["capture_count"] += 1

    window["tiredness_sum"] += info["tiredness"]
    if info["playing_dead"]:
        window["play_dead_steps"] += 1


def _print_progress(step, ep_count, window, agent):
    """
    format and print the periodic training summary
    """
    # calculates totals to print percentages
    dist_count_safe = max(window["dist_count"], 1)
    zone_total = max(window["steps_in_capture_zone"] +
                     window["steps_in_danger_zone"] +
                     window["steps_in_play_zone"] +
                     window["steps_in_approach_zone"] +
                     window["steps_in_far_zone"],
                    1)
    cat_state_total = max(window["steps_cat_wandering"] +
                          window["steps_cat_stalking"] +
                          window["steps_cat_pouncing"] +
                          window["steps_cat_leaving"],
                        1)
    loss_count_safe = window["loss_update_count"] or 1

    print(
        f"  step {step:6d}  ep {ep_count:4d}  "
        f"avg_r {window['reward_sum']/LOG_WINDOW_SIZE:.2f}  "
        f"avg_dist {window['dist_sum']/dist_count_safe:.3f}  "
        f"captures {window['capture_count']}\n"
        f"    zones   play={100*window['steps_in_play_zone']//zone_total:2d}%  "
        f"approach={100*window['steps_in_approach_zone']//zone_total:2d}%  "
        f"danger={100*window['steps_in_danger_zone']//zone_total:2d}%  "
        f"far={100*window['steps_in_far_zone']//zone_total:2d}%  "
        f"cap={100*window['steps_in_capture_zone']//zone_total:2d}%\n"
        f"    cat     wander={100*window['steps_cat_wandering']//cat_state_total:2d}%  "
        f"stalk={100*window['steps_cat_stalking']//cat_state_total:2d}%  "
        f"pounce={100*window['steps_cat_pouncing']//cat_state_total:2d}%  "
        f"flee={100*window['steps_cat_leaving']//cat_state_total:2d}%\n"
        f"    losses  critic={window['critic_loss_sum']/loss_count_safe:.3f}  "
        f"actor={window['actor_loss_sum']/loss_count_safe:.4f}  "
        f"log_alpha={agent.log_alpha.exp().item():.4f}  "
        f"|v|={window['linear_action_sum']/max(window['action_count'],1):.3f}  "
        f"|w|={window['angular_action_sum']/max(window['action_count'],1):.3f}\n"
        f"    tired   avg={window['tiredness_sum']/dist_count_safe:.3f}  "
        f"dead_steps={window['play_dead_steps']}"
    )


def train():
    env = CatSimEnv()
    agent = MouseSACAgent()
    buf = ReplayBuffer(BUFFER_SIZE)

    obs = env.reset()
    ep_count = 0
    window = _reset_window()

    # : formats value
    print(f"training for {TOTAL_STEPS:,} steps  (learn_start={LEARN_START})\n")

    for step in range(TOTAL_STEPS):
        if step < LEARN_START:
            action = np.random.uniform(-1, 1, ACT_DIM)
        else:
            action = agent.select_action(obs, explore=True)

        next_obs, reward, done, info = env.step(action)
        # divide reward by REWARD_SCALER before storing so q values stay small and stable
        # raw rewards range -2.5 to +1.5, dividing by 3.5 keeps them in -0.7 to +0.4
        # the logged reward is still raw so the numbers are easier to interpret
        buf.push(obs, action, [reward / REWARD_SCALER], next_obs, [float(done)])
        obs = next_obs # roll forward to next state

        _accumulate_stats(window, action, reward, info)

        if done:
            ep_count += 1
            if ep_count % LOG_WINDOW_SIZE == 0:
                _print_progress(step, ep_count, window, agent)
                window = _reset_window()
            obs = env.reset()

        if len(buf) < LEARN_START:
            continue

        # updates losses after learning starts
        losses = agent.update(buf.sample(BATCH))
        window["critic_loss_sum"] += losses["critic_loss"]
        window["actor_loss_sum"] += losses["actor_loss"]
        window["loss_update_count"] += 1

    out_path = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')
    agent.save(out_path)


if __name__ == '__main__':
    train()
