"""
animates one episode of the trained (or untrained) mouse policy vs the cat sim

modes:
    --policy random: random actions (sanity check the env itself, no torch needed for the policy)
    --policy trained: loads shared/mouse_policy.pt and uses the real actor (default)
    --cat manual: you control the cat with WASD/arrow keys, mouse responds with its policy

usage:
    python train/visualize_episode.py
    python train/visualize_episode.py --policy random
    python train/visualize_episode.py --save episode.mp4
    python train/visualize_episode.py --cat manual
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon

from train.train_sac import (
    CatSimEnv, WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX,
    CAPTURE_DIST, DANGER_DIST, PLAY_DIST_HI, APPROACH_DIST, OBS_DIST_SCALE,
    SIM_DT,
)

# cat fill color by FSM state
STATE_COLORS = {
    "wandering": "#4189d9",
    "stalking": "#e6a817",
    "pouncing": "#d94141",
    "leaving": "#888888",
    "holding": "#9b59b6",
    "manual": "#27ae60",
}

# fixed border colors for the two shapes
CAT_BORDER_COLOR = "#e6842a" # orange
MOUSE_BORDER_COLOR = "#666666" # gray
MOUSE_FILL_COLOR = "#666666"

# velocity magnitude below this counts as "not really moving", keep facing the last known heading instead of snapping to an arbitrary angle
HEADING_STILL_EPS = 1e-3


def load_policy():
    """
    returns a callable obs -> action, using the real trained actor
    """
    import torch
    from shared.actor import Actor

    actor = Actor()
    policy_path = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')
    actor.load_state_dict(torch.load(policy_path, map_location="cpu"))
    actor.eval()

    def act(obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = actor.get_deterministic_action(obs_t)
        return action.squeeze(0).numpy()

    return act


def random_policy():
    def act(obs):
        return np.random.uniform(-1, 1, 2)
    return act


# builds the 3 vertices of an isosceles triangle pointing along `heading`
# uses the same sin/cos forward direction convention as mouse_heading in the sim
# (forward = (sin(heading), cos(heading))), so a heading of 0 points "up"
def _triangle_vertices(x, y, heading, front=0.18, back=0.10, half_width=0.09):
    fx, fy = np.sin(heading), np.cos(heading) # forward
    rx, ry = np.cos(heading), -np.sin(heading) # perpendicular ("right")
    tip = (x + fx * front, y + fy * front)
    base_l = (x - fx * back + rx * half_width, y - fy * back + ry * half_width)
    base_r = (x - fx * back - rx * half_width, y - fy * back - ry * half_width)
    return [tip, base_l, base_r]


MANUAL_CAT_SPEED = 1.0


class ManualCatEnv(CatSimEnv):
    """
    env where the cat is controlled by keyboard instead of the FSM
    """

    def __init__(self):
        super().__init__()
        self.manual_cat_vx = 0.0
        self.manual_cat_vy = 0.0

    def _update_cat(self, dist):
        self.just_dodged = False
        self.cat_vx = self.manual_cat_vx
        self.cat_vy = self.manual_cat_vy
        self.cat_state = "manual"
        new_x = self.cat_x + self.cat_vx * SIM_DT
        new_y = self.cat_y + self.cat_vy * SIM_DT
        self.cat_x = np.clip(new_x, WORLD_X_MIN, WORLD_X_MAX)
        self.cat_y = np.clip(new_y, WORLD_Y_MIN, WORLD_Y_MAX)


def run_manual(env, act_fn):
    """
    real time interactive mode: user controls cat with WASD / arrows, mouse responds with policy
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(WORLD_X_MIN - 0.3, WORLD_X_MAX + 0.3)
    ax.set_ylim(WORLD_Y_MIN - 0.3, WORLD_Y_MAX + 0.3)
    ax.set_aspect("equal")
    ax.set_title("manual cat control  |  WASD / arrow keys to move cat  |  R to reset")

    for r, color, label in [
        (CAPTURE_DIST, "#d94141", "capture"),
        (DANGER_DIST, "#e67a17", "danger"),
        (PLAY_DIST_HI, "#4caf50", "play"),
        (APPROACH_DIST, "#4189d9", "approach"),
    ]:
        ax.add_patch(Circle((0, 0), r * OBS_DIST_SCALE, fill=False, linestyle="--", linewidth=0.8, color=color, alpha=0.4))

    cat_shape = Polygon(_triangle_vertices(0, 0, 0), closed=True, facecolor=STATE_COLORS["manual"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5)
    mouse_shape = Polygon(_triangle_vertices(0, 0, 0), closed=True, facecolor=MOUSE_FILL_COLOR, edgecolor=MOUSE_BORDER_COLOR, linewidth=1.5)
    ax.add_patch(cat_shape)
    ax.add_patch(mouse_shape)

    safe_play_circle = Circle((0, 0), PLAY_DIST_HI * OBS_DIST_SCALE, fill=False, linestyle=":", linewidth=1.2, color=CAT_BORDER_COLOR, alpha=0.6)
    ax.add_patch(safe_play_circle)

    cat_trail, = ax.plot([], [], "-", linewidth=0.8, color="#aaaaaa", alpha=0.5)
    mouse_trail, = ax.plot([], [], "-", linewidth=0.8, color="#666666", alpha=0.5)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace")

    held_keys = set()
    state = {"obs": env.reset(), "step": 0, "captures": 0}
    cat_xs, cat_ys, mouse_xs, mouse_ys = [], [], [], []
    last_cat_heading = [0.0]

    def on_press(event):
        if event.key:
            held_keys.add(event.key)
            if event.key == 'r':
                state["obs"] = env.reset()
                state["step"] = 0
                state["captures"] = 0
                cat_xs.clear(); cat_ys.clear()
                mouse_xs.clear(); mouse_ys.clear()

    def on_release(event):
        if event.key:
            held_keys.discard(event.key)

    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('key_release_event', on_release)

    def update(_frame):
        vx, vy = 0.0, 0.0
        if 'w' in held_keys or 'up' in held_keys: vy += MANUAL_CAT_SPEED
        if 's' in held_keys or 'down' in held_keys: vy -= MANUAL_CAT_SPEED
        if 'a' in held_keys or 'left' in held_keys: vx -= MANUAL_CAT_SPEED
        if 'd' in held_keys or 'right' in held_keys: vx += MANUAL_CAT_SPEED
        env.manual_cat_vx = vx
        env.manual_cat_vy = vy

        action = act_fn(state["obs"])
        next_obs, reward, done, info = env.step(action)
        state["obs"] = next_obs
        state["step"] += 1
        if info["captured"]:
            state["captures"] += 1

        if done:
            state["obs"] = env.reset()
            state["step"] = 0
            state["captures"] = 0
            cat_xs.clear(); cat_ys.clear()
            mouse_xs.clear(); mouse_ys.clear()

        cat_xs.append(env.cat_x); cat_ys.append(env.cat_y)
        mouse_xs.append(env.mouse_x); mouse_ys.append(env.mouse_y)

        cat_speed = np.hypot(env.cat_vx, env.cat_vy)
        if cat_speed > HEADING_STILL_EPS:
            last_cat_heading[0] = np.arctan2(env.cat_vx, env.cat_vy)

        cat_shape.set_xy(_triangle_vertices(env.cat_x, env.cat_y, last_cat_heading[0]))
        cat_shape.set_facecolor(STATE_COLORS.get("manual", "#27ae60"))
        mouse_shape.set_xy(_triangle_vertices(env.mouse_x, env.mouse_y, env.mouse_heading))
        safe_play_circle.set_center((env.cat_x, env.cat_y))

        cat_trail.set_data(cat_xs[-40:], cat_ys[-40:])
        mouse_trail.set_data(mouse_xs[-40:], mouse_ys[-40:])

        pcs = env.post_capture_state
        state_str = "STRUGGLING!" if pcs == "struggling" else "DEAD" if pcs == "dead" else ""
        captured_str = "CAPTURED!" if info["captured"] else ""
        status_text.set_text(
            f"step {state['step']:3d}   captures: {state['captures']}\n"
            f"dist: {info['dist']:.3f}\n"
            f"reward: {reward:+.2f}\n"
            f"{captured_str or state_str}"
        )
        return cat_shape, mouse_shape, safe_play_circle, cat_trail, mouse_trail, status_text

    import itertools
    anim = animation.FuncAnimation(
        fig, update, frames=itertools.count(), interval=50, blit=True, cache_frame_data=False,
    )
    plt.show()


def run_episode(env, act_fn, max_steps=300):
    """
    rolls out one full episode, returns list of frame dicts
    """
    obs = env.reset()
    frames = []
    for _ in range(max_steps):
        action = act_fn(obs)
        next_obs, reward, done, info = env.step(action)
        frames.append({
            "cat_x": env.cat_x, "cat_y": env.cat_y,
            "cat_vx": env.cat_vx, "cat_vy": env.cat_vy, # needed to derive cat heading
            "mouse_x": env.mouse_x, "mouse_y": env.mouse_y,
            "mouse_heading": env.mouse_heading,
            "cat_state": info["cat_state"],
            "reward": reward, "dist": info["dist"],
            "captured": info["captured"],
            "post_capture_state": info["post_capture_state"],
        })
        obs = next_obs
        if done:
            break
    return frames

def run_episodes(env, act_fn, n_episodes, max_steps=300):
    """
    rolls out several episodes back to back, tagging each frame with which
    episode it belongs to and whether it's the first frame of a new episode
    """
    all_frames = []
    for ep in range(n_episodes):
        ep_frames = run_episode(env, act_fn, max_steps)
        for i, f in enumerate(ep_frames):
            f["episode"] = ep + 1
            f["episode_start"] = (i == 0)
        all_frames.extend(ep_frames)
    return all_frames


def animate(frames, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(WORLD_X_MIN - 0.3, WORLD_X_MAX + 0.3)
    ax.set_ylim(WORLD_Y_MIN - 0.3, WORLD_Y_MAX + 0.3)
    ax.set_aspect("equal")
    ax.set_title("mouse-proj: SAC policy vs cat sim")

    # reward zone rings, centered on the mouse's start (0,0), roughly accurate since the mouse moves, but gives a visual sense of scale
    for r, color, label in [
        (CAPTURE_DIST, "#d94141", "capture"),
        (DANGER_DIST, "#e67a17", "danger"),
        (PLAY_DIST_HI, "#4caf50", "play"),
        (APPROACH_DIST, "#4189d9", "approach"),
    ]:
        ax.add_patch(Circle((0, 0), r * OBS_DIST_SCALE, fill=False, linestyle="--", linewidth=0.8, color=color, alpha=0.4))

    # cat and mouse are triangle patches so they can be rotated to show heading
    cat_shape = Polygon(_triangle_vertices(0, 0, 0), closed=True, facecolor=STATE_COLORS["wandering"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5)
    mouse_shape = Polygon(_triangle_vertices(0, 0, 0), closed=True, facecolor=MOUSE_FILL_COLOR, edgecolor=MOUSE_BORDER_COLOR, linewidth=1.5)
    ax.add_patch(cat_shape)
    ax.add_patch(mouse_shape)

    # dotted "safe play distance" ring, follows the cat every frame
    safe_play_circle = Circle((0, 0), PLAY_DIST_HI * OBS_DIST_SCALE, fill=False, linestyle=":", linewidth=1.2, color=CAT_BORDER_COLOR, alpha=0.6)
    ax.add_patch(safe_play_circle)

    cat_trail, = ax.plot([], [], "-", linewidth=0.8, color="#aaaaaa", alpha=0.5)
    mouse_trail, = ax.plot([], [], "-", linewidth=0.8, color="#666666", alpha=0.5)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace")

    legend_handles = [
        Polygon([(0, 0)], closed=True, facecolor=STATE_COLORS["wandering"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5, label="cat: wandering"),
        Polygon([(0, 0)], closed=True, facecolor=STATE_COLORS["stalking"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5, label="cat: stalking"),
        Polygon([(0, 0)], closed=True, facecolor=STATE_COLORS["pouncing"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5, label="cat: pouncing"),
        Polygon([(0, 0)], closed=True, facecolor=STATE_COLORS["leaving"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5, label="cat: leaving"),
        Polygon([(0, 0)], closed=True, facecolor=STATE_COLORS["holding"], edgecolor=CAT_BORDER_COLOR, linewidth=1.5, label="cat: holding"),
        Polygon([(0, 0)], closed=True, facecolor=MOUSE_FILL_COLOR, edgecolor=MOUSE_BORDER_COLOR, linewidth=1.5, label="mouse"),
        plt.Line2D([0], [0], linestyle=":", color=CAT_BORDER_COLOR, label="safe play distance"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    cat_xs, cat_ys, mouse_xs, mouse_ys = [], [], [], []
    last_cat_heading = [0.0]

    def init():
        cat_shape.set_xy(_triangle_vertices(0, 0, 0))
        mouse_shape.set_xy(_triangle_vertices(0, 0, 0))
        safe_play_circle.set_center((0, 0))
        cat_trail.set_data([], [])
        mouse_trail.set_data([], [])
        status_text.set_text("")
        return cat_shape, mouse_shape, safe_play_circle, cat_trail, mouse_trail, status_text

    def update(i):
        f = frames[i]

        # starting a new episode, clear trails so they don't draw a line
        if f["episode_start"]:
            cat_xs.clear(); cat_ys.clear()
            mouse_xs.clear(); mouse_ys.clear()

        cat_xs.append(f["cat_x"]); cat_ys.append(f["cat_y"])
        mouse_xs.append(f["mouse_x"]); mouse_ys.append(f["mouse_y"])

        # derive cat heading from velocity direction, keep last heading if nearly still
        cat_speed = np.hypot(f["cat_vx"], f["cat_vy"])
        if cat_speed > HEADING_STILL_EPS:
            last_cat_heading[0] = np.arctan2(f["cat_vx"], f["cat_vy"])
        cat_heading = last_cat_heading[0]

        cat_shape.set_xy(_triangle_vertices(f["cat_x"], f["cat_y"], cat_heading))
        cat_shape.set_facecolor(STATE_COLORS.get(f["cat_state"], "#888888"))
        mouse_shape.set_xy(_triangle_vertices(f["mouse_x"], f["mouse_y"], f["mouse_heading"]))
        safe_play_circle.set_center((f["cat_x"], f["cat_y"]))

        cat_trail.set_data(cat_xs[-40:], cat_ys[-40:])
        mouse_trail.set_data(mouse_xs[-40:], mouse_ys[-40:])

        status_text.set_text(
            f"episode {f['episode']}   step {i:3d}\n"
            f"cat_state: {f['cat_state']}\n"
            f"dist: {f['dist']:.3f}\n"
            f"reward: {f['reward']:+.2f}\n"
            f"{'CAPTURED!' if f['captured'] else 'STRUGGLING!' if f.get('post_capture_state')=='struggling' else 'DEAD' if f.get('post_capture_state')=='dead' else ''}"
        )
        return cat_shape, mouse_shape, safe_play_circle, cat_trail, mouse_trail, status_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=50, blit=True, repeat=True,
    )

    if save_path:
        print(f"saving to {save_path} ...")
        anim.save(save_path, fps=20)
        print("done")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["trained", "random"], default="trained")
    ap.add_argument("--cat", choices=["fsm", "manual"], default="fsm", help="fsm = simulated cat AI (default), manual = you control the cat with WASD/arrows")
    ap.add_argument("--save", default=None, help="e.g. episode.mp4 or episode.gif (omit to just show a live window)")
    ap.add_argument("--episodes", type=int, default=5, help="how many episodes to play back to back")
    args = ap.parse_args()

    act_fn = load_policy() if args.policy == "trained" else random_policy()

    if args.cat == "manual":
        env = ManualCatEnv()
        print("manual cat mode: WASD/arrows to move cat, R to reset")
        run_manual(env, act_fn)
    else:
        env = CatSimEnv()
        frames = run_episodes(env, act_fn, args.episodes)
        print(f"ran {args.episodes} episodes, {len(frames)} total steps")
        animate(frames, save_path=args.save)


if __name__ == "__main__":
    main()