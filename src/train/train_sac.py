"""
trains the SAC agent in a tiny 2D sim
cat moves with a state machine
(wander -> stalk -> pounce -> flee)
no real camera needed

output:
    src/shared/mouse_policy.pt

usage:
    `python train/train_sac.py`
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    MAX_V, MAX_OMEGA, ACTION_EMA_ALPHA,
    SIM_DT, WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX,
    OBS_DIST_SCALE, VISIBILITY_RANGE, MAX_EP_STEPS, OUTPUT_DIM,
    FOV_HALF_ANGLE,
    STALK_ENTER_DIST_RANGE, STALK_EXIT_BUFFER, STALK_STEPS_REQ_RANGE,
    POUNCE_SPEED_RANGE, WANDER_SPEED_MAX_RANGE,
    CAT_START_VX_RANGE, CAT_START_VY_RANGE,
    POUNCE_MISS_DIST, WANDER_ACCEL_NOISE, WANDER_CURIOSITY_BIAS,
    CLOSE_DIST, LEAVE_ENTER_STEPS, LEAVE_SPEED, LEAVE_EXIT_DIST, LEAVE_MAX_STEPS,
    CAT_BORED_STEPS_THRESHOLD, HOLD_DURATION,
    WOBBLE_FREQ_HZ_RANGE, WOBBLE_AMP_RANGE,
    CAT_STATIONARY_SPEED_THRESH,
    CAPTURE_DIST, DANGER_DIST, PLAY_DIST_HI, APPROACH_DIST,
    CAPTURE_PENALTY, DANGER_BASE_PENALTY,
    PLAY_ZONE_REWARD, TOO_FAR_PENALTY, DODGE_BONUS, REWARD_SCALER,
    MOUSE_TEASE_SPEED_THRESH, TEASE_BONUS, TEASE_STATIONARY_STEPS,
    MAX_CAPTURE_COUNT,
    WALL_MARGIN, WALL_PENALTY,
    STRUGGLE_DURATION, STRUGGLE_SPEED_SCALE, STRUGGLE_ESCAPE_BONUS, PLAY_DEAD_DURATION,
    FACING_BONUS,
    SURVIVAL_REWARD,
    FREEZE_SPEED_THRESHOLD, FREEZE_PENALTY,
    LET_CAT_WIN_STEPS, LET_CAT_WIN_SPEED_SCALE,
)


class Critic(nn.Module):
    """
    double Q network, training only, not deployed to pi
    """
    def __init__(self):
        super().__init__()
        def _q():
            return nn.Sequential(
                nn.Linear(OBS_DIM + ACT_DIM, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, OUTPUT_DIM),
            )
        self.q1, self.q2 = _q(), _q()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n):
        batch = random.sample(self.buf, n)
        return [torch.FloatTensor(np.array(x)) for x in zip(*batch)]

    def __len__(self):
        return len(self.buf)


class MouseSACAgent:
    """
    wraps all SAC components: actor, double critic, target critic,
    three adam optimizers, and learnable entropy coefficient log_alpha
    """

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, obs_np, explore=True):
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            if explore:
                action, _ = self.actor.get_action(obs_t)
            else:
                action = self.actor.get_deterministic_action(obs_t)
        return action.squeeze().numpy()

    def update(self, batch):
        O, A, R, NO, D = batch
        alpha = self.log_alpha.exp().detach()

        # critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(NO)
            q1_target, q2_target = self.critic_target(NO, next_action)
            target = R + GAMMA * (1 - D) * (torch.min(q1_target, q2_target) - alpha * next_log_prob)

        q1, q2 = self.critic(O, A)
        self.critic_opt.zero_grad()
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q1_loss.backward()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        self.critic_opt.step()

        # actor update
        new_action, log_prob = self.actor.get_action(O)
        q1n, q2n = self.critic(O, new_action)
        a_loss = (alpha * log_prob - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        self.actor_opt.step()

        # log_alpha update
        al_loss = -(self.log_alpha * (log_prob + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad()
        al_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=LOG_ALPHA_CLAMP_MIN)

        # soft target update
        for s, t in zip(self.critic.parameters(), self.critic_target.parameters()):
            t.data.copy_(TAU * s.data + (1 - TAU) * t.data)

        return {
            "critic_loss": (q1_loss + q2_loss).item(),
            "actor_loss": a_loss.item(),
            "alpha_loss": al_loss.item(),
            "log_alpha": self.log_alpha.exp().item(),
        }

    def save(self, path):
        torch.save(self.actor.state_dict(), path)
        print(f"policy saved -> {path}")


class CatSimEnv:
    """
    2D sim with a multi-state cat and a mouse.

    obs: [dist, angle, visible, cat_vx, cat_vy, cat_state_float, mouse_speed]
    act: [v_raw, omega_raw] in [-1, 1]
    """

    def __init__(self): # removed tiredness, coverage, action history for unexpectability (too detailed, mouse found loopholes)
        self.cat_vx = self.cat_vy = 0.0
        self.mouse_speed = 0.0
        self.cat_state = "wandering"
        self.stalk_steps = self.cat_stationary_steps = self.capture_count = self.close_steps = self.leave_steps = 0
        self.cat_bored_steps = 0
        self.hold_steps = 0
        self.steps_since_last_capture = 0

        self.post_capture_state = None # None / "struggling" / "dead"
        self.post_capture_timer = 0
        self.ema_action = np.zeros(ACT_DIM)

        self.reset()

    def _sample_cat_params(self):
        self.stalk_enter_dist = np.random.uniform(*STALK_ENTER_DIST_RANGE)
        self.stalk_exit_dist = self.stalk_enter_dist + STALK_EXIT_BUFFER
        self.stalk_steps_req = np.random.randint(*STALK_STEPS_REQ_RANGE)
        self.pounce_speed = np.random.uniform(*POUNCE_SPEED_RANGE)
        self.wander_speed_max = np.random.uniform(*WANDER_SPEED_MAX_RANGE)
        self.wobble_freq_hz = np.random.uniform(*WOBBLE_FREQ_HZ_RANGE)
        self.wobble_amp = np.random.uniform(*WOBBLE_AMP_RANGE)
        self.wobble_phase0 = np.random.uniform(0, 2 * np.pi)

    def reset(self):
        self._sample_cat_params()

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
        self.cat_stationary_steps = 0
        self.capture_count = 0
        self.close_steps = 0
        self.leave_steps = 0
        self.cat_bored_steps = 0
        self.hold_steps = 0
        self.steps_since_last_capture = 0
        self.step_n = 0

        self.post_capture_state = None
        self.post_capture_timer = 0
        self.ema_action = np.zeros(ACT_DIM)

        return self._obs()

    def _cat_state_float(self):
        return {"wandering": 0.0, "stalking": 0.25, "pouncing": 1.0, "leaving": -0.5, "holding": -1.0, "manual": 0.0}[self.cat_state]

    def _is_cat_visible(self): # reusable visibility check for reward
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = np.hypot(dx, dy) / OBS_DIST_SCALE
        world_angle = np.arctan2(dx, dy)
        rel_angle = (world_angle - self.mouse_heading + np.pi) % (2 * np.pi) - np.pi
        return dist < VISIBILITY_RANGE and abs(rel_angle) < FOV_HALF_ANGLE

    def _obs(self):
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0)

        world_angle = np.arctan2(dx, dy)
        rel_angle = (world_angle - self.mouse_heading + np.pi) % (2 * np.pi) - np.pi
        angle = np.clip(rel_angle / np.pi, -1.0, 1.0)

        visible = float(dist < VISIBILITY_RANGE and abs(rel_angle) < FOV_HALF_ANGLE)
        obs_cat_vx = self.cat_vx if visible else 0.0
        obs_cat_vy = self.cat_vy if visible else 0.0

        return np.array([dist, angle, visible, obs_cat_vx, obs_cat_vy, self._cat_state_float(), self.mouse_speed], dtype=np.float32)

    def _update_cat(self, dist):
        self.just_dodged = False

        if dist < CLOSE_DIST:
            self.close_steps += 1
        else:
            self.close_steps = 0

        if self.mouse_speed < FREEZE_SPEED_THRESHOLD:
            self.cat_bored_steps += 1
        else:
            self.cat_bored_steps = 0

        if self.cat_state not in ("leaving", "holding") and (self.close_steps >= LEAVE_ENTER_STEPS or self.cat_bored_steps >= CAT_BORED_STEPS_THRESHOLD):
            self.cat_state = "leaving"
            self.leave_steps = 0
            self.stalk_steps = 0
            self.cat_bored_steps = 0

        if self.cat_state == "holding":
            self.hold_steps += 1
            if self.hold_steps >= HOLD_DURATION:
                self.cat_state = "leaving"
                self.leave_steps = 0
                self.hold_steps = 0
                self.cat_bored_steps = 0

        elif self.cat_state == "leaving":
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

        # velocity based on state
        dx_to_mouse = self.mouse_x - self.cat_x
        dy_to_mouse = self.mouse_y - self.cat_y
        dist_to_mouse = np.hypot(dx_to_mouse, dy_to_mouse) + 1e-6
        dir_x = dx_to_mouse / dist_to_mouse
        dir_y = dy_to_mouse / dist_to_mouse

        if self.cat_state == "holding":
            self.cat_vx = 0.0
            self.cat_vy = 0.0
        elif self.cat_state == "wandering":
            self.cat_vx = np.clip(self.cat_vx + np.random.uniform(-WANDER_ACCEL_NOISE, WANDER_ACCEL_NOISE) + dir_x * WANDER_CURIOSITY_BIAS * SIM_DT, -self.wander_speed_max, self.wander_speed_max)
            self.cat_vy = np.clip(self.cat_vy + np.random.uniform(-WANDER_ACCEL_NOISE, WANDER_ACCEL_NOISE) + dir_y * WANDER_CURIOSITY_BIAS * SIM_DT, -self.wander_speed_max, self.wander_speed_max)
        elif self.cat_state == "leaving":
            self.cat_vx = -dir_x * LEAVE_SPEED
            self.cat_vy = -dir_y * LEAVE_SPEED
        elif self.cat_state == "stalking":
            perp_x, perp_y = -dir_y, dir_x
            phase = self.wobble_phase0 + self.stalk_steps * SIM_DT * 2 * np.pi * self.wobble_freq_hz
            wobble = np.sin(phase) * self.wobble_amp
            self.cat_vx = perp_x * wobble
            self.cat_vy = perp_y * wobble
        else: # pouncing
            self.cat_vx = dir_x * self.pounce_speed
            self.cat_vy = dir_y * self.pounce_speed

        new_x = self.cat_x + self.cat_vx * SIM_DT
        new_y = self.cat_y + self.cat_vy * SIM_DT
        clipped_x = np.clip(new_x, WORLD_X_MIN, WORLD_X_MAX)
        clipped_y = np.clip(new_y, WORLD_Y_MIN, WORLD_Y_MAX)
        if clipped_x != new_x: self.cat_vx *= 0.0
        if clipped_y != new_y: self.cat_vy *= 0.0
        self.cat_x, self.cat_y = clipped_x, clipped_y

        cat_speed = np.hypot(self.cat_vx, self.cat_vy)
        if cat_speed < CAT_STATIONARY_SPEED_THRESH:
            self.cat_stationary_steps += 1
        else:
            self.cat_stationary_steps = 0

    def _wall_closeness(self):
        closest = min(
            self.mouse_x - WORLD_X_MIN,
            WORLD_X_MAX - self.mouse_x,
            self.mouse_y - WORLD_Y_MIN,
            WORLD_Y_MAX - self.mouse_y,
        )
        if closest < WALL_MARGIN:
            return 1.0 - (closest / WALL_MARGIN)
        return 0.0

    def _reward(self, dist, captured): # simplified to 8 conditions
        if self.post_capture_state == "dead" and not captured:
            return 0.0

        r = 0.0

        # zone rewards (mutually exclusive)
        if captured:
            r = CAPTURE_PENALTY
        elif dist < DANGER_DIST:
            r = DANGER_BASE_PENALTY * (1.0 - 0.5 * self.mouse_speed / MAX_V) # the higher the speed the less the penalty, but max penalty is still *0.5
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

        # tease bonus
        if dist < PLAY_DIST_HI and self.cat_stationary_steps >= TEASE_STATIONARY_STEPS and self.mouse_speed > MOUSE_TEASE_SPEED_THRESH:
            r += TEASE_BONUS

        # wall penalty
        wall_closeness = self._wall_closeness()
        if wall_closeness > 0:
            r -= WALL_PENALTY * wall_closeness

        # struggle escape bonus
        if self.post_capture_state == "struggling":
            struggle_max = MAX_V * STRUGGLE_SPEED_SCALE
            r += STRUGGLE_ESCAPE_BONUS * (self.mouse_speed / struggle_max)

        # facing bonus
        if self._is_cat_visible():
            r += FACING_BONUS

        # freeze penalty
        if self.post_capture_state != "dead" and self.mouse_speed < FREEZE_SPEED_THRESHOLD:
            r -= FREEZE_PENALTY

        # survival
        r += SURVIVAL_REWARD

        return float(r)

    def step(self, action):
        v_raw, omega_raw = float(action[0]), float(action[1])

        # mechanical action smoothing via ema
        raw_action = np.array([v_raw, omega_raw])
        self.ema_action = ACTION_EMA_ALPHA * raw_action + (1.0 - ACTION_EMA_ALPHA) * self.ema_action
        v_ema, omega_ema = float(self.ema_action[0]), float(self.ema_action[1])

        # post capture mouse state machine: struggling -> dead -> revive
        if self.post_capture_state == "struggling":
            self.post_capture_timer -= 1
            if self.post_capture_timer <= 0:
                self.post_capture_state = "dead"
                self.post_capture_timer = PLAY_DEAD_DURATION
            max_v_this_step = MAX_V * STRUGGLE_SPEED_SCALE
            if self.steps_since_last_capture >= LET_CAT_WIN_STEPS:
                max_v_this_step *= LET_CAT_WIN_SPEED_SCALE
            v = ((v_ema + 1.0) / 2.0) * max_v_this_step # converts bidrectional action into forward only speed (shifts [-1, 1] -> [0, 2], rescales [0, 2] -> [0, 1], scales [0, 1] -> [0, max_v_this_step]
            omega = omega_ema * MAX_OMEGA # scales tanh action [-1, 1] -> [-MAX_OMEGA, MAX_OMEGA]
        elif self.post_capture_state == "dead":
            self.post_capture_timer -= 1
            if self.post_capture_timer <= 0:
                self.post_capture_state = None
            v = 0.0
            omega = 0.0
        else:
            max_v_this_step = MAX_V
            if self.steps_since_last_capture >= LET_CAT_WIN_STEPS:
                max_v_this_step *= LET_CAT_WIN_SPEED_SCALE
            v = ((v_ema + 1.0) / 2.0) * max_v_this_step
            omega = omega_ema * MAX_OMEGA

        self.mouse_heading += omega * SIM_DT
        self.mouse_x = np.clip(self.mouse_x + v * np.sin(self.mouse_heading) * SIM_DT, WORLD_X_MIN, WORLD_X_MAX)
        self.mouse_y = np.clip(self.mouse_y + v * np.cos(self.mouse_heading) * SIM_DT, WORLD_Y_MIN, WORLD_Y_MAX)
        self.mouse_speed = v

        # dist before cat moves (cat reacts to this)
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = float(np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0))
        self._update_cat(dist)

        # dist after both moved (capture/reward use this)
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist_post = float(np.clip(np.hypot(dx, dy) / OBS_DIST_SCALE, 0.0, 1.0))

        captured = dist_post < CAPTURE_DIST
        if captured:
            self.capture_count = min(self.capture_count + 1, MAX_CAPTURE_COUNT)
            self.steps_since_last_capture = 0
            if self.post_capture_state is None:
                self.post_capture_state = "struggling"
                self.post_capture_timer = STRUGGLE_DURATION
                self.cat_state = "holding"
                self.hold_steps = 0
        else:
            self.steps_since_last_capture += 1

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
            "post_capture_state": self.post_capture_state
        }
        return obs, reward, done, info


def _reset_window():
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
        "steps_cat_holding": 0,
        "critic_loss_sum": 0.0,
        "actor_loss_sum": 0.0,
        "loss_update_count": 0,
        "struggle_steps": 0,
        "dead_steps": 0,
    }


def _accumulate_stats(window, action, reward, info):
    dist = info["dist"]
    cat_state = info["cat_state"]

    window["reward_sum"] += reward
    window["dist_sum"] += dist
    window["dist_count"] += 1

    if info["captured"]: window["steps_in_capture_zone"] += 1
    elif dist < DANGER_DIST: window["steps_in_danger_zone"] += 1
    elif dist < PLAY_DIST_HI: window["steps_in_play_zone"] += 1
    elif dist < APPROACH_DIST: window["steps_in_approach_zone"] += 1
    else: window["steps_in_far_zone"] += 1

    if cat_state == "wandering": window["steps_cat_wandering"] += 1
    elif cat_state == "stalking": window["steps_cat_stalking"] += 1
    elif cat_state == "pouncing": window["steps_cat_pouncing"] += 1
    elif cat_state == "leaving": window["steps_cat_leaving"] += 1
    elif cat_state == "holding": window["steps_cat_holding"] += 1

    if info["captured"]:
        window["capture_count"] += 1

    if info["post_capture_state"] == "struggling":
        window["struggle_steps"] += 1
    elif info["post_capture_state"] == "dead":
        window["dead_steps"] += 1


def _print_progress(step, ep_count, window, agent):
    dist_count_safe = max(window["dist_count"], 1)
    zone_total = max(window["steps_in_capture_zone"] +
                     window["steps_in_danger_zone"] +
                     window["steps_in_play_zone"] +
                     window["steps_in_approach_zone"] +
                     window["steps_in_far_zone"], 1)
    cat_state_total = max(window["steps_cat_wandering"] +
                          window["steps_cat_stalking"] +
                          window["steps_cat_pouncing"] +
                          window["steps_cat_leaving"] +
                          window["steps_cat_holding"], 1)
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
        f"flee={100*window['steps_cat_leaving']//cat_state_total:2d}%  "
        f"hold={100*window['steps_cat_holding']//cat_state_total:2d}%\n"
        f"    losses  critic={window['critic_loss_sum']/loss_count_safe:.3f}  "
        f"actor={window['actor_loss_sum']/loss_count_safe:.4f}  "
        f"log_alpha={agent.log_alpha.exp().item():.4f}\n"
        f"    post    struggle={window['struggle_steps']}  dead={window['dead_steps']}"
    )


def train():
    env = CatSimEnv()
    agent = MouseSACAgent()
    buf = ReplayBuffer(BUFFER_SIZE)

    obs = env.reset()
    ep_count = 0
    window = _reset_window()

    print(f"training for {TOTAL_STEPS:,} steps  (learn_start={LEARN_START})\n")

    for step in range(TOTAL_STEPS):
        if step < LEARN_START:
            action = np.random.uniform(-1, 1, ACT_DIM)
        else:
            action = agent.select_action(obs, explore=True)

        next_obs, reward, done, info = env.step(action)
        buf.push(obs, action, [reward / REWARD_SCALER], next_obs, [float(done)])
        obs = next_obs

        _accumulate_stats(window, action, reward, info)

        if done:
            ep_count += 1
            if ep_count % LOG_WINDOW_SIZE == 0:
                _print_progress(step, ep_count, window, agent)
                window = _reset_window()
            obs = env.reset()

        if len(buf) < LEARN_START:
            continue

        losses = agent.update(buf.sample(BATCH))
        window["critic_loss_sum"] += losses["critic_loss"]
        window["actor_loss_sum"] += losses["actor_loss"]
        window["loss_update_count"] += 1

    out_path = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')
    agent.save(out_path)


if __name__ == '__main__':
    train()
