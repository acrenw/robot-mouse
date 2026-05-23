"""
main inference loop
camera/video -> yolo -> obs -> SAC actor -> safety layer -> motors

usage:
    # test on a video file (laptop):
    python run/run.py --src data/videos/pounce-5.mp4

    # test on webcam:
    python run/run.py --src 0

    # pi with camera (add --headless if running over SSH):
    python run/run.py --src /dev/video0 --headless

    # test without a trained policy (random actions, still tests the pipeline):
    python run/run.py --src data/videos/pounce-5.mp4 --no-policy

# TODO: add speaker that plays a sound when cat is detected or gets close
# TODO: add snack servo that dispenses a treat after the robot teases the cat
# TODO: add --headless flag properly so this runs cleanly over SSH on pi
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import cv2

from shared.actor import Actor, OBS_DIM
from detect import get_cat_state, draw_debug, FRAME_W, FRAME_H
from motors import safety_layer, send_to_motors, stop, MAX_V, MAX_OMEGA
from sensors import read_front_m

POLICY_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')


def load_actor(path):
    actor = Actor()
    actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    actor.eval()
    print(f"[run] loaded policy from {path}")
    return actor


def build_obs(cat_state, mouse_speed, sensor_front=1.0):
    # [dist_proxy, angle, visible, vx, vy, sensor_front, mouse_speed]
    return np.append(cat_state, [sensor_front, mouse_speed]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='0', help='video file or camera index')
    parser.add_argument('--no-policy', action='store_true', help='run with random actions')
    parser.add_argument('--fps', type=int, default=10, help='control loop target hz')
    parser.add_argument('--headless', action='store_true', help='skip cv2.imshow (use over SSH)')
    args = parser.parse_args()

    actor = None
    if not args.no_policy and os.path.exists(POLICY_PATH):
        actor = load_actor(POLICY_PATH)
    else:
        print("[run] no policy found, using random actions (run train/train_sac.py first)")

    # use picamera2 for pi camera (cv2.VideoCapture can't talk to libcamera on pi 5)
    # fall back to cv2 for video files and laptop webcams
    use_picam = args.src.isdigit() or args.src.startswith('/dev/video')
    picam = None
    cap = None

    if use_picam:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            cfg = picam.create_preview_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
            )
            picam.configure(cfg)
            picam.start()
            print("[run] picamera2 ready")
        except Exception as e:
            print(f"[run] picamera2 failed ({e}), falling back to cv2")
            use_picam = False

    if not use_picam:
        src = int(args.src) if args.src.isdigit() else args.src
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, str) else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            print(f"[run] can't open source: {args.src}"); sys.exit(1)

    dt = 1.0 / args.fps
    prev_bbox = None
    mouse_speed = 0.0

    # stale state tracking:
    # phase 1 (0 to STALE_TIMEOUT seconds): hold last known position, set vis=0
    # phase 2 (after STALE_TIMEOUT): bypass policy and spin toward where cat was last seen
    STALE_TIMEOUT = 0.3 # seconds before switching to spin search
    SEARCH_OMEGA_FRAC = 0.7 # how fast to spin when searching (fraction of MAX_OMEGA)
    last_valid_state = None
    last_detect_time = None

    print(f"[run] starting loop at {args.fps} hz\n")
    try:
        while True:
            t0 = time.time()

            if use_picam:
                frame = picam.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # loop video files
                    continue
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # detect cat
            raw_state, bbox = get_cat_state(frame, prev_bbox)

            now = time.time()
            if raw_state[2] == 1.0: # yolo found the cat this frame
                cat_state = raw_state
                last_valid_state = raw_state.copy() # save a copy so stale mode can use it
                last_detect_time = now
                prev_bbox = bbox # save for velocity calculation next frame
                stale = False
                search_mode = False
            else: # yolo missed this frame
                # how long since we last saw the cat
                since = (now - last_detect_time) if last_detect_time is not None else float('inf')
                if last_valid_state is not None and since <= STALE_TIMEOUT:
                    # still within 0.3s, hold last known position so robot doesn't spaz out
                    cat_state = last_valid_state.copy()
                    cat_state[2] = 0.0 # tell the policy vis=0 so it knows this is stale data
                    cat_state[3:5] = 0.0 # zero velocity because we have no idea where cat moved
                    bbox = None # don't draw a box we're not confident about
                    stale = True
                    search_mode = False
                else:
                    # been more than 0.3s, give up on last known position and start spinning
                    cat_state = raw_state # raw_state is [1.0, 0.0, 0.0, ...] when nothing detected
                    bbox = None
                    stale = False
                    search_mode = True

            # read front distance sensor
            sensor_front = read_front_m()

            obs = build_obs(cat_state, mouse_speed, sensor_front)

            if search_mode:
                # bypass the policy entirely and spin toward wherever the cat was last seen
                # if cat was to the right (positive angle) we spin right (negative omega)
                # -np.sign flips the angle direction to get the spin direction.
                if last_valid_state is not None and abs(last_valid_state[1]) > 0.05:
                    search_dir = -np.sign(last_valid_state[1])  # spin toward last known side
                else:
                    search_dir = 1.0 # no last known position, just spin left by default
                v_safe = 0.0 # don't move forward while searching, just rotate
                omega_safe = float(search_dir) * MAX_OMEGA * SEARCH_OMEGA_FRAC
                mouse_speed = 0.0
                action = np.array([0.0, float(search_dir) * SEARCH_OMEGA_FRAC]) # for logging only
            else:
                if actor is not None:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0) # add batch dimension
                    with torch.no_grad(): # no gradients needed at inference time
                        action = actor.get_deterministic_action(obs_t).squeeze().numpy()
                else:
                    action = np.random.uniform(-1, 1, 2) # random if no policy loaded

                v_des = float(action[0]) * MAX_V # scale from [-1,1] to actual m/s
                omega_des = float(action[1]) * MAX_OMEGA # scale from [-1,1] to actual rad/s

                # when the cat is visible and near center, fade omega toward 0 so
                # the robot drives straight at it instead of spinning in place
                # the policy has a slight spin bias when centered, this corrects it
                CENTRE_BAND = 0.15 # angles smaller than this get the fade applied
                if cat_state[2] == 1.0 and abs(cat_state[1]) < CENTRE_BAND:
                    fade = abs(cat_state[1]) / CENTRE_BAND # 0 at dead center, 1 at the edge of the band
                    omega_des *= fade # smoothly reduce spin as cat approaches center

                # if cat is clearly off to one side, make sure we're turning the right way
                # the policy sometimes outputs the wrong sign for omega, this hard corrects it
                # only applies when tracking, not when very close (dist < 0.10) because
                # at close range we might legitimately evade in any direction
                SIGN_THRESH = 0.30  # only correct when cat is clearly off center
                if abs(cat_state[1]) > SIGN_THRESH and cat_state[0] > 0.10 and cat_state[2] == 1.0:
                    correct_sign = -np.sign(cat_state[1]) # positive angle = cat right = need negative omega
                    if np.sign(omega_des) != correct_sign:
                        omega_des = correct_sign * abs(omega_des) # flip sign, keep magnitude

                v_safe, omega_safe = safety_layer(v_des, omega_des, sensor_front)
                mouse_speed = abs(v_safe) # track speed for the obs next frame

            send_to_motors(v_safe, omega_safe)

            mode_tag = " [search]" if search_mode else (" [stale]" if stale else "")
            print(f"  [state] d={cat_state[0]:.2f} ang={cat_state[1]:.2f} "
                  f"vis={cat_state[2]:.0f}{mode_tag} | "
                  f"action v_raw={action[0]:+.2f} w_raw={action[1]:+.2f}")

            if not args.headless:
                frame = draw_debug(frame, cat_state, bbox, stale=stale)
                action_label = f"v={v_safe:+.2f} w={omega_safe:+.2f}"
                cv2.putText(frame, action_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.imshow("robomouse", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        stop()
        if picam:
            picam.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[run] done.")


if __name__ == '__main__':
    main()
