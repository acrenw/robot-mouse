"""
animates one episode of the trained (or untrained) mouse policy vs the cat sim

two modes:
    --policy random: random actions (sanity check the env itself, no torch needed for the policy)
    --policy trained: loads shared/mouse_policy.pt and uses the real actor (default)

usage:
    python train/visualize_episode.py
    python train/visualize_episode.py --policy random
    python train/visualize_episode.py --save episode.mp4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from train.train_sac import (
    CatSimEnv, WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX,
    CAPTURE_DIST, DANGER_DIST, PLAY_DIST_HI, APPROACH_DIST, OBS_DIST_SCALE,
)

STATE_COLORS = {
    "wandering": "#4189d9",
    "stalking": "#e6a817",
    "pouncing": "#d94141",
    "leaving": "#888888",
}


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
            "mouse_x": env.mouse_x, "mouse_y": env.mouse_y,
            "mouse_heading": env.mouse_heading,
            "cat_state": info["cat_state"],
            "reward": reward, "dist": info["dist"],
            "captured": info["captured"],
        })
        obs = next_obs
        if done:
            break
    return frames


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

    cat_dot, = ax.plot([], [], "o", markersize=14, color="#888888")
    mouse_dot, = ax.plot([], [], "^", markersize=10, color="#222222")
    cat_trail, = ax.plot([], [], "-", linewidth=0.8, color="#aaaaaa", alpha=0.5)
    mouse_trail, = ax.plot([], [], "-", linewidth=0.8, color="#666666", alpha=0.5)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace")

    cat_xs, cat_ys, mouse_xs, mouse_ys = [], [], [], []

    def init():
        cat_dot.set_data([], [])
        mouse_dot.set_data([], [])
        cat_trail.set_data([], [])
        mouse_trail.set_data([], [])
        status_text.set_text("")
        return cat_dot, mouse_dot, cat_trail, mouse_trail, status_text

    def update(i):
        f = frames[i]
        cat_xs.append(f["cat_x"]); cat_ys.append(f["cat_y"])
        mouse_xs.append(f["mouse_x"]); mouse_ys.append(f["mouse_y"])

        cat_dot.set_data([f["cat_x"]], [f["cat_y"]])
        cat_dot.set_color(STATE_COLORS.get(f["cat_state"], "#888888"))
        mouse_dot.set_data([f["mouse_x"]], [f["mouse_y"]])

        cat_trail.set_data(cat_xs[-40:], cat_ys[-40:])
        mouse_trail.set_data(mouse_xs[-40:], mouse_ys[-40:])

        status_text.set_text(
            f"step {i:3d}\n"
            f"cat_state: {f['cat_state']}\n"
            f"dist: {f['dist']:.3f}\n"
            f"reward: {f['reward']:+.2f}\n"
            f"{'CAPTURED!' if f['captured'] else ''}"
        )
        return cat_dot, mouse_dot, cat_trail, mouse_trail, status_text

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
    ap.add_argument("--save", default=None, help="e.g. episode.mp4 or episode.gif (omit to just show a live window)")
    args = ap.parse_args()

    env = CatSimEnv()
    act_fn = load_policy() if args.policy == "trained" else random_policy()

    frames = run_episode(env, act_fn)
    print(f"episode ran {len(frames)} steps, "
          f"final dist={frames[-1]['dist']:.3f}, captured={frames[-1]['captured']}")

    animate(frames, save_path=args.save)


if __name__ == "__main__":
    main()