"""
main inference loop
camera/video -> yolo -> obs -> SAC actor -> safety layer -> motors

usage:
    test on a video file (laptop):
    `python run/run.py --src data/videos/pounce-5.mp4`

    test on webcam:
    `python run/run.py --src 0`

    pi with camera (add --headless if running over SSH):
    `python run/run.py --src /dev/video0 --headless`

    test without a trained policy (random actions, still tests the pipeline):
    `python run/run.py --src data/videos/pounce-5.mp4 --no-policy`
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
from speaker import maybe_squeak
from servo import dispense
from train.train_config import ACTION_EMA_ALPHA, STRUGGLE_DURATION, PLAY_DEAD_DURATION

POLICY_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared', 'mouse_policy.pt')


def load_actor(path):
    actor = Actor()
    actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))  # weights_only=True unpickles weights only and won't execute malicious code
    actor.eval() # switch from rain -> inference mode (doesn't matter for me tho since i don't use Dropout and BatchNorm)
    print(f"[run] loaded policy from {path}")
    return actor

# deployment cat consts # TODO: tweak during deployment
POUNCE_DIST = 0.10
POUNCE_SPEED = 0.3
STALK_DIST = 0.25
STALK_SPEED = 0.1
LEAVE_DIST = 0.40
LEAVE_SPEED_THRESH = 0.2


def estimate_cat_state_float(dist, vx, vy, visible): # nn learns that when this float is high, bad things happen
    if not visible:
        return 0.0
    cat_speed = (vx**2 + vy**2) ** 0.5
    if dist < POUNCE_DIST and cat_speed > POUNCE_SPEED: # pouncing
        return 1.0
    elif dist < STALK_DIST and cat_speed < STALK_SPEED: # stalking
        return 0.25
    elif dist > LEAVE_DIST and cat_speed > LEAVE_SPEED_THRESH: # leaving
        return -0.5
    return 0.0


def build_obs(cat_state, mouse_speed, cat_state_float=0.0): # obs uses cat state, not sensor front (sensor front is only for clipping motors in safety layer now)
    # [dist_proxy, angle, visible, vx, vy, cat_state_float, mouse_speed]
    return np.append(cat_state, [cat_state_float, mouse_speed]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    # -- makes arg an optional flag, w/o it's positional arg
    parser.add_argument('--src', default='0', help='video file or camera index') # default 0 means camera index 0 (camera on pi or laoptop webcam)
    parser.add_argument('--no-policy', action='store_true', help='run with random actions') # store_true makes it so that args.no_policy is True of --no-policy is present in cli
    parser.add_argument('--fps', type=int, default=10, help='control loop target hz') # low default fps to keep YOLO inference calls/s small, lighter pi cpu load
    parser.add_argument('--headless', action='store_true', help='skip cv2.imshow (use over SSH)') # if wanna add gui
    args = parser.parse_args()

    actor = None
    if not args.no_policy and os.path.exists(POLICY_PATH):
        actor = load_actor(POLICY_PATH)
    else:
        print("[run] no policy found, using random actions (run train/train_sac.py first)")

    # find camera to use (both still use opencv), picamera2 (not available on pc) better than cv.VideoCapture
    # fall back to cv2 for video files and laptop webcams
    use_picam = args.src.isdigit() or args.src.startswith('/dev/video') # can be int, /dev/video.*, or video file path 
    picam = None
    cap = None

    if use_picam:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            cfg = picam.create_preview_configuration( # low latency config (unlike max quality configs like create_still_configuration or create_video_configuration)
                main={"size": (FRAME_W, FRAME_H), "format": "BGR888"} # bgr888 matches opencv's native format, 8 bits per colour
            )
            picam.configure(cfg)
            picam.start()
            print("[run] picamera2 ready")
        except Exception as e:
            print(f"[run] picamera2 failed ({e}), falling back to cv2")
            use_picam = False

    if not use_picam:
        src = int(args.src) if args.src.isdigit() else args.src
        # if src is str (ie. /dev/video0) then use linux video capture api to open source, else let open cv auto select wtvr backend is most appropriate
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, str) else cv2.CAP_ANY)
        # set capture properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W) #TODO: set constants to correct w and h
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        if not cap.isOpened():
            print(f"[run] can't open source: {args.src}")
            sys.exit(1)

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

    CATCH_DIST = 0.08 # dist_proxy below this means cat is basically on top of the robot
    CATCH_TIME = 0.3 # cat has to stay that close for 0.3s to avoid false triggers from fast passes
    catch_start = None
    post_capture_state = None
    post_capture_timer = 0
    ema_action = np.zeros(2)

    print(f"[run] starting loop at {args.fps} hz\n")
    try:
        sensor_buf = []
        while True:
            t0 = time.time()

            if use_picam:
                frame = picam.capture_array()
            else:
                ret, frame = cap.read()
                if not ret: # reached end of file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)# sets current playback position of video capture to 0, aka restart to loop the video
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

            # read front distance sensor (keep last three dists and get min)
            sensor_front = read_front_m()
            sensor_buf.append(sensor_front)
            sensor_buf = sensor_buf[-3:]
            sensor_front = min(sensor_buf)
            print(f"  [sensor] {sensor_front:.2f}m")

            csf = estimate_cat_state_float(cat_state[0], cat_state[3], cat_state[4], cat_state[2] == 1.0)
            obs = build_obs(cat_state, mouse_speed, csf)

            if search_mode:
                # bypass the policy entirely and spin toward wherever the cat was last seen
                # if cat was to the right (positive angle) we spin right (negative omega)
                # -np.sign flips the angle direction to get the spin direction.
                # +ive omega is ccw rotation, -ive is cw rotation
                if last_valid_state is not None and abs(last_valid_state[1]) > 0.05:
                    search_dir = -np.sign(last_valid_state[1])  # spin toward last known side
                else:
                    search_dir = 1.0 # no last known position, just spin left by default
                # call safety layer
                v_safe, omega_safe = safety_layer(0.0, float(search_dir) * MAX_OMEGA * SEARCH_OMEGA_FRAC, sensor_front)
                mouse_speed = 0.0
                action = np.array([0.0, float(search_dir) * SEARCH_OMEGA_FRAC]) # for logging only, so not scaled to real units with MAX_OMEGA
            else:
                if actor is not None:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0) # add batch dimension of 1
                    with torch.no_grad(): # no gradients needed at inference time
                        action = actor.get_deterministic_action(obs_t).squeeze().numpy() # squeeze to get rid of batch dim
                else:
                    #                         low, high, size
                    action = np.random.uniform(-1, 1, 2) # random if no policy loaded

                raw_action = np.array([float(action[0]), float(action[1])])
                ema_action = ACTION_EMA_ALPHA * raw_action + (1.0 - ACTION_EMA_ALPHA) * ema_action

                v_des = ((float(ema_action[0]) + 1.0) / 2.0) * MAX_V
                omega_des = float(ema_action[1]) * MAX_OMEGA

                v_safe, omega_safe = safety_layer(v_des, omega_des, sensor_front)
                mouse_speed = abs(v_safe) # track speed for the obs next frame

            send_to_motors(v_safe, omega_safe)

            now_t = time.time()

            # post capture state machine: struggle -> dead -> revive
            if post_capture_state == "struggling":
                post_capture_timer -= 1
                if post_capture_timer <= 0:
                    post_capture_state = "dead"
                    post_capture_timer = PLAY_DEAD_DURATION
                    dispense()
            elif post_capture_state == "dead":
                post_capture_timer -= 1
                if post_capture_timer <= 0:
                    post_capture_state = None

            # detect new capture
            if cat_state[2] == 1.0 and cat_state[0] < CATCH_DIST:
                if catch_start is None:
                    catch_start = now_t
                elif now_t - catch_start >= CATCH_TIME and post_capture_state is None:
                    post_capture_state = "struggling"
                    post_capture_timer = STRUGGLE_DURATION
            else:
                catch_start = None

            # squeaking: panicked during struggle, silent during dead, normal otherwise
            if cat_state[2] == 1.0:
                if post_capture_state == "struggling":
                    maybe_squeak(0.01, cat_state[3], cat_state[4], True, now_t)
                elif post_capture_state != "dead":
                    maybe_squeak(cat_state[0], cat_state[3], cat_state[4], True, now_t)

            mode_tag = " [search]" if search_mode else (" [stale]" if stale else "")
            print(f"  [state] d={cat_state[0]:.2f} ang={cat_state[1]:.2f} "
                  f"vis={cat_state[2]:.0f}{mode_tag} | "
                  f"action v_raw={action[0]:+.2f} w_raw={action[1]:+.2f}")

            if not args.headless:
                frame = draw_debug(frame, cat_state, bbox, stale=stale)
                action_label = f"v={v_safe:+.2f} w={omega_safe:+.2f}"

                #           frame, text, bottom left test anchor coord, font, fotn scale, line thickness
                cv2.putText(frame, action_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.imshow("robomouse", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): # quit with q, & is python bitwise and
                    break

            # min loop rate is dt long
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt: # ctrl c
        pass

    finally:
        stop() # stop motors
        if picam:
            picam.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[run] done.")


if __name__ == '__main__':
    main()
