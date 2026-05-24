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
TODO: add domain randomization (vary cat speed, stalk patience, etc)
TODO: export policy to onnx for faster inference on pi
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

from shared.actor import Actor, OBS_DIM, ACT_DIM

# hyperparameters
TOTAL_STEPS = 200_000
BUFFER_SIZE = 50_000
BATCH = 128
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
TARGET_ENTROPY = -float(ACT_DIM)
LEARN_START = 2_000
MAX_V = 0.3
MAX_OMEGA = 2.0

# cat state machine thresholds
STALK_ENTER_DIST = 0.50 # cat starts stalking when mouse gets this close
STALK_EXIT_DIST = 0.55 # hysteresis, cat stops stalking if mouse gets this far
STALK_STEPS_REQ = 35 # how long cat stalks before pouncing
POUNCE_SPEED = 0.12 # how fast the cat moves when pouncing
STALK_DRIFT_SPEED = 0.04 # how fast cat drifts toward mouse while stalking
WANDER_SPEED_MAX = 0.08 # max cat speed when wandering

# reward zone boundaries
CAPTURE_DIST = 0.06 # closer than this = captured
DANGER_DIST = 0.15 # in the danger zone (between capture and here)
PLAY_DIST_HI = 0.35 # ideal play zone is 0.15 to 0.35
APPROACH_DIST = 0.65 # approach zone is 0.35 to 0.65, beyond = too far

# teasing reward thresholds
CAT_STATIONARY_THRESH = 0.025
MOUSE_TEASE_THRESH = 0.10

MAX_EVASIONS = 5

# normalize rewards before pushing to buffer so Q-values don't blow up
# raw rewards range roughly -2.5 to 1.5, dividing by 3.5 keeps them in -0.7 to 0.4
# logs still show raw rewards so they're easier to read
REWARD_SCALE = 3.5

# flee state
CLOSE_DIST = 0.20 # if mouse is this close for too long, cat flees
FLEE_ENTER_STEPS = 10 # how many consecutive close steps before fleeing
FLEE_SPEED = 0.12 # how fast cat runs away
FLEE_EXIT_DIST = 0.55 # cat stops fleeing once it gets this far
FLEE_MAX_STEPS = 40 # max steps cat flees before going back to wandering


class Critic(nn.Module):
    """
    double Q network, training only, not deployed to pi
    """
    def __init__(self):
        super().__init__()
        def _q():
            return nn.Sequential(
                nn.Linear(OBS_DIM + ACT_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 1),
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

    actor lives in shared/actor.py (also used on pi, don't modify it here)
    critic is training-only
    """

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.critic_tgt = Critic()
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, obs_np, explore=True):
        """
        obs_np: np.ndarray (OBS_DIM,)
        returns np.ndarray (ACT_DIM,) in [-1, 1]
        explore=True: stochastic (used during training)
        explore=False: greedy (used for eval)
        """
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            if explore:
                action, _ = self.actor.get_action(obs_t)
            else:
                action = self.actor.get_deterministic_action(obs_t)
        return action.squeeze().numpy()

    def update(self, batch):
        """
        one SAC gradient step.
        batch: [O, A, R, NO, D] FloatTensors from ReplayBuffer.sample()
        """
        O, A, R, NO, D = batch
        alpha = self.log_alpha.exp().detach()

        # critic update
        with torch.no_grad():
            na, nlp = self.actor.get_action(NO) # sample next action from current policy
            q1t, q2t = self.critic_tgt(NO, na) # get Q values from target network (more stable)
            # bellman target: reward + discounted future value, minus entropy bonus
            # (1-D) masks out terminal states so we don't bootstrap past episode end
            # torch.min takes the more pessimistic Q estimate to prevent overestimation
            y = R + GAMMA * (1 - D) * (torch.min(q1t, q2t) - alpha * nlp)

        q1, q2 = self.critic(O, A) # Q values for the actions we actually took
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y) # both heads should predict y
        self.critic_opt.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # prevent gradient explosion
        self.critic_opt.step()

        # actor update
        new_a, lp = self.actor.get_action(O) # sample fresh actions (can't reuse above, need gradients)
        q1n, q2n = self.critic(O, new_a) # how good does critic think these actions are
        # maximize Q - alpha*entropy: actor tries to find high-value actions while staying exploratory
        a_loss = (alpha * lp - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # alpha (entropy) update: auto-tune how much we weight exploration vs exploitation
        # if entropy is too low (lp + TARGET_ENTROPY > 0), increase alpha to encourage more exploration
        al_loss = -(self.log_alpha * (lp + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad(); al_loss.backward(); self.alpha_opt.step()
        # clamp log_alpha so alpha never goes below exp(-3) ~= 0.05
        # without this the agent stops exploring completely around episode 40 and gets stuck
        with torch.no_grad():
            self.log_alpha.clamp_(min=-3.0)

        # slowly blend target critic toward current critic (TAU=0.005 = 0.5% per step)
        # using a slowly-updated target makes training much more stable than updating it directly
        for s, t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            t.data.copy_(TAU * s.data + (1 - TAU) * t.data)

        return {
            "critic_loss": c_loss.item(),
            "actor_loss": a_loss.item(),
            "alpha_loss": al_loss.item(),
            "alpha": self.log_alpha.exp().item(),
        }

    def save(self, path):
        """
        save actor weights only (critic stays on the laptop)
        """
        torch.save(self.actor.state_dict(), path)
        print(f"policy saved -> {path}")


class CatSimEnv:
    """
    2D sim with a four-state cat (wander / stalk / pounce / flee)
    and a capture + evasion mechanic.

    obs: [dist, angle, visible, cat_vx, cat_vy, sensor_front, mouse_speed]
    act: [v_raw, omega_raw] in [-1, 1]

    # TODO: add more realistic cat behaviors (circling, fake-outs, etc.)
    # TODO: add multiple cat sim to train for scenarios with multiple cats in view
    """

    def __init__(self):
        self.cat_vx = self.cat_vy = 0.0
        self.mouse_speed = 0.0
        self.cat_state = "wandering"
        self.stalk_steps = self.stationary_steps = self.evasion_count = 0
        self.close_steps = self.flee_steps = 0
        self.reset()

    def reset(self):
        # keep re-rolling until cat starts at a reasonable distance (approach zone)
        # if cat starts too far away (dist > 0.65) every step gets -0.5 reward with no
        # gradient signal pointing toward the cat, so the agent never learns anything
        # if cat starts too close (dist < 0.30) the agent just runs away from the start
        while True:
            self.cat_x = np.random.uniform(-1, 1)
            self.cat_y = np.random.uniform(0.2, 1.5)
            if 0.30 <= np.hypot(self.cat_x, self.cat_y) / 2.0 <= 0.65: # /2.0 normalizes to [0,1] range
                break

        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.mouse_heading = 0.0
        self.mouse_speed = 0.0
        self.cat_vx = np.random.uniform(-0.05, 0.05)
        self.cat_vy = np.random.uniform(-0.01, 0.04)
        self.cat_state = "wandering"
        self.stalk_steps = 0
        self.stationary_steps = 0
        self.evasion_count = 0
        self.close_steps = 0
        self.flee_steps = 0
        self.step_n = 0
        return self._obs()

    def _obs(self):
        dx = self.cat_x - self.mouse_x # horizontal offset from mouse to cat
        dy = self.cat_y - self.mouse_y # vertical offset from mouse to cat
        # hypot = straight-line distance. /2.0 normalizes the sim space so dist=1 means very far
        dist = np.clip(np.hypot(dx, dy) / 2.0, 0.0, 1.0)
        # robot-relative angle: positive = cat is to the right of where we're facing
        # this matches detect.py where positive angle = cat right of frame center
        world_angle = np.arctan2(dx, dy) # angle to cat in world frame (north = 0)
        # subtract mouse heading to get angle relative to which way the robot is pointing
        # the +pi) % 2pi - pi trick wraps the result into [-pi, pi] cleanly
        rel_angle = (world_angle - self.mouse_heading + np.pi) % (2 * np.pi) - np.pi
        angle = np.clip(rel_angle / np.pi, -1.0, 1.0) # normalize to [-1, 1]
        visible = float(dist < 0.7) # cat counts as visible if it's within 70% of max range
        return np.array(
            [dist, angle, visible, self.cat_vx, self.cat_vy, 1.0, self.mouse_speed],
            dtype=np.float32,
        )

    def _update_cat(self, dist):
        """
        advance cat state machine and move cat
        returns cat_speed
        """

        # track how long mouse has been uncomfortably close
        if dist < CLOSE_DIST:
            self.close_steps += 1
        else:
            self.close_steps = 0

        # flee overrides everything if mouse has been too close too long
        if self.cat_state != "fleeing" and self.close_steps >= FLEE_ENTER_STEPS:
            self.cat_state = "fleeing"
            self.flee_steps = 0
            self.stalk_steps = 0

        if self.cat_state == "fleeing":
            self.flee_steps += 1
            if dist > FLEE_EXIT_DIST or self.flee_steps >= FLEE_MAX_STEPS:
                self.cat_state = "wandering"
                self.close_steps = 0
                self.flee_steps = 0

        elif self.cat_state == "wandering":
            if dist < STALK_ENTER_DIST:
                self.cat_state = "stalking"
                self.stalk_steps = 0

        elif self.cat_state == "stalking":
            if dist > STALK_EXIT_DIST:
                self.cat_state = "wandering"
                self.stalk_steps = 0
            elif self.stalk_steps >= STALK_STEPS_REQ:
                self.cat_state = "pouncing"
                self.stalk_steps = 0
            else:
                self.stalk_steps += 1

        elif self.cat_state == "pouncing":
            if dist > 0.7 or dist < CAPTURE_DIST:
                self.cat_state = "wandering"
                self.stalk_steps = 0

        # set velocity based on current state
        dx_to_mouse = self.mouse_x - self.cat_x
        dy_to_mouse = self.mouse_y - self.cat_y
        mag = np.hypot(dx_to_mouse, dy_to_mouse) + 1e-6

        if self.cat_state == "wandering":
            self.cat_vx = np.clip(
                self.cat_vx + np.random.uniform(-0.01, 0.01), -WANDER_SPEED_MAX, WANDER_SPEED_MAX
            )
            self.cat_vy = np.clip(
                self.cat_vy + np.random.uniform(-0.01, 0.01), -WANDER_SPEED_MAX, WANDER_SPEED_MAX
            )
        elif self.cat_state == "fleeing":
            self.cat_vx = -(dx_to_mouse / mag) * FLEE_SPEED
            self.cat_vy = -(dy_to_mouse / mag) * FLEE_SPEED
        else: # stalking or pouncing
            spd = STALK_DRIFT_SPEED if self.cat_state == "stalking" else POUNCE_SPEED
            self.cat_vx = (dx_to_mouse / mag) * spd
            self.cat_vy = (dy_to_mouse / mag) * spd

        # move cat, bounce off walls
        new_x = self.cat_x + self.cat_vx
        new_y = self.cat_y + self.cat_vy
        clipped_x = np.clip(new_x, -2.0, 2.0)
        clipped_y = np.clip(new_y, 0.0, 3.0)
        if clipped_x != new_x: self.cat_vx *= -1
        if clipped_y != new_y: self.cat_vy *= -1
        self.cat_x, self.cat_y = clipped_x, clipped_y

        cat_speed = np.hypot(self.cat_vx, self.cat_vy)
        if cat_speed < CAT_STATIONARY_THRESH:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0

        return cat_speed

    def _reward(self, dist, captured, cat_speed):
        """
        zone based reward with teasing and pounce freeze bonuses
        """
        if captured:
            r = +1.5 if self.evasion_count >= MAX_EVASIONS else -2.0
        elif dist < DANGER_DIST:
            r = -1.0 + (self.mouse_speed / MAX_V) # reward fast escape
        elif dist < PLAY_DIST_HI:
            r = +1.0 # ideal zone
        elif dist < APPROACH_DIST:
            t = (APPROACH_DIST - dist) / (APPROACH_DIST - PLAY_DIST_HI)
            r = float(np.clip(t, 0.0, 1.0))
        else:
            r = -0.5 # too far, go find the cat

        # bonus for teasing (cat is still, mouse is actively moving near it)
        if cat_speed < CAT_STATIONARY_THRESH and self.mouse_speed > MOUSE_TEASE_THRESH:
            r += 0.3

        # penalty for freezing when cat is pouncing
        if self.cat_state == "pouncing" and self.mouse_speed < MOUSE_TEASE_THRESH:
            r -= 0.5

        return float(r)

    def step(self, action):
        v_raw, omega_raw = float(action[0]), float(action[1])
        v = v_raw * MAX_V
        omega = omega_raw * MAX_OMEGA

        # move mouse forward based on heading, 0.1 is the sim timestep (dt)
        self.mouse_heading += omega * 0.1 # rotate first
        self.mouse_x += v * np.sin(self.mouse_heading) * 0.1 # sin/cos converts heading to x/y movement
        self.mouse_y += v * np.cos(self.mouse_heading) * 0.1
        self.mouse_speed = abs(v)# track scalar speed for the obs and teasing reward

        # dist before cat moves
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = float(np.clip(np.hypot(dx, dy) / 2.0, 0.0, 1.0))

        cat_speed = self._update_cat(dist)

        # recompute dist after cat moves
        dx = self.cat_x - self.mouse_x
        dy = self.cat_y - self.mouse_y
        dist = float(np.clip(np.hypot(dx, dy) / 2.0, 0.0, 1.0))

        captured = dist < CAPTURE_DIST
        if captured:
            self.evasion_count = min(self.evasion_count + 1, MAX_EVASIONS)

        self.step_n += 1

        reward = self._reward(dist, captured, cat_speed)
        done = (self.step_n >= 300) or (self.evasion_count >= MAX_EVASIONS and captured)

        obs = self._obs()
        info = {
            "cat_state": self.cat_state,
            "evasion_count": self.evasion_count,
            "dist": dist,
            "captured": captured,
            "mouse_speed": self.mouse_speed,
            "cat_speed": cat_speed,
            "close_steps": self.close_steps,
        }
        return obs, reward, done, info


def _reset_window():
    """
    fresh accumulator for the per-20-episode log window
    """
    return {
        "reward_sum": 0.0, "dist_sum": 0.0, "dist_n": 0, "captures": 0,
        "z_cap": 0, "z_danger": 0, "z_play": 0, "z_approach": 0, "z_far": 0,
        "cs_wander": 0, "cs_stalk": 0, "cs_pounce": 0, "cs_flee": 0,
        "c_loss": 0.0, "a_loss": 0.0, "loss_n": 0,
        "v_sum": 0.0, "w_sum": 0.0, "act_n": 0,
    }


def train():
    env = CatSimEnv()
    agent = MouseSACAgent()
    buf = ReplayBuffer(BUFFER_SIZE)

    obs = env.reset()
    ep_count = 0
    w = _reset_window()

    print(f"training for {TOTAL_STEPS:,} steps  (learn_start={LEARN_START})\n")

    for step in range(TOTAL_STEPS):

        if step < LEARN_START:
            action = np.random.uniform(-1, 1, ACT_DIM)
        else:
            action = agent.select_action(obs, explore=True)

        next_obs, reward, done, info = env.step(action)
        # divide reward by REWARD_SCALE before storing so Q-values stay small and stable
        # raw rewards range -2.5 to +1.5, dividing by 3.5 keeps them in -0.7 to +0.4
        # the logged reward is still raw so the numbers are easier to interpret
        buf.push(obs, action, [reward / REWARD_SCALE], next_obs, [float(done)])
        obs = next_obs # roll forward to next state

        d = info["dist"]
        cs = info["cat_state"]

        w["reward_sum"] += reward
        w["dist_sum"] += d
        w["dist_n"] += 1
        w["v_sum"] += abs(float(action[0]))
        w["w_sum"] += abs(float(action[1]))
        w["act_n"] += 1

        if info["captured"]: w["z_cap"] += 1
        elif d < DANGER_DIST: w["z_danger"] += 1
        elif d < PLAY_DIST_HI: w["z_play"] += 1
        elif d < APPROACH_DIST: w["z_approach"] += 1
        else: w["z_far"] += 1

        if cs == "wandering": w["cs_wander"] += 1
        elif cs == "stalking": w["cs_stalk"] += 1
        elif cs == "pouncing": w["cs_pounce"] += 1
        elif cs == "fleeing": w["cs_flee"] += 1

        if info["captured"]:
            w["captures"] += 1

        if done:
            ep_count += 1
            if ep_count % 20 == 0:
                n = max(w["dist_n"], 1)
                tz = max(w["z_cap"] + w["z_danger"] + w["z_play"] + w["z_approach"] + w["z_far"], 1)
                tcs = max(w["cs_wander"] + w["cs_stalk"] + w["cs_pounce"] + w["cs_flee"], 1)
                ll = w["loss_n"] or 1

                print(
                    f"  step {step:6d}  ep {ep_count:4d}  "
                    f"avg_r {w['reward_sum']/20:.2f}  "
                    f"avg_dist {w['dist_sum']/n:.3f}  "
                    f"captures {w['captures']}\n"
                    f"    zones   play={100*w['z_play']//tz:2d}%  "
                    f"approach={100*w['z_approach']//tz:2d}%  "
                    f"danger={100*w['z_danger']//tz:2d}%  "
                    f"far={100*w['z_far']//tz:2d}%  "
                    f"cap={100*w['z_cap']//tz:2d}%\n"
                    f"    cat     wander={100*w['cs_wander']//tcs:2d}%  "
                    f"stalk={100*w['cs_stalk']//tcs:2d}%  "
                    f"pounce={100*w['cs_pounce']//tcs:2d}%  "
                    f"flee={100*w['cs_flee']//tcs:2d}%\n"
                    f"    losses  critic={w['c_loss']/ll:.3f}  actor={w['a_loss']/ll:.4f}  "
                    f"alpha={agent.log_alpha.exp().item():.4f}  "
                    f"|v|={w['v_sum']/max(w['act_n'],1):.3f}  "
                    f"|w|={w['w_sum']/max(w['act_n'],1):.3f}"
                )
                w = _reset_window()
            obs = env.reset()

        if len(buf) < LEARN_START:
            continue
        losses = agent.update(buf.sample(BATCH))
        w["c_loss"] += losses["critic_loss"]
        w["a_loss"] += losses["actor_loss"]
        w["loss_n"] += 1

    out = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')
    agent.save(out)


if __name__ == '__main__':
    train()
