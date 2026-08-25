"""
all training constants for train_sac.py
"""

import numpy as np
from shared.actor import ACT_DIM

# training hyperparameters
TOTAL_STEPS = 400_000 # was 200k, more room for exploration with higher entropy
BUFFER_SIZE = 50_000
BATCH = 128
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
LEARN_START = 2_000
LOG_ALPHA_CLAMP_MIN = -3.0
GRAD_CLIP_MAX_NORM = 1.0
TARGET_ENTROPY = -float(ACT_DIM)
LOG_WINDOW_SIZE = 20

# mouse physics
MAX_V = 0.5
MAX_OMEGA = 2.0
ACTION_EMA_ALPHA = 0.7 # mechanical action smoothing, replaces reward based smoothness penalty (alpha=1 means no smoothing, alpha=0 means frozen bc too much smoothing)

# simulation world
SIM_DT = 0.1
WORLD_X_MIN, WORLD_X_MAX = -2.0, 2.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 3.0
OBS_DIST_SCALE = 3.6
VISIBILITY_RANGE = 0.7
FOV_HALF_ANGLE = np.pi / 3 # 60 each side = 120 total fov
MAX_EP_STEPS = 300
OUTPUT_DIM = 1

# cat state machine
STALK_ENTER_DIST_RANGE = (0.40, 0.60)
STALK_EXIT_BUFFER = 0.05
STALK_STEPS_REQ_RANGE = (20, 50)
POUNCE_SPEED_RANGE = (0.9, 1.6)
WANDER_SPEED_MAX_RANGE = (0.5, 1.1)
CAT_START_VX_RANGE = (-0.05, 0.05)
CAT_START_VY_RANGE = (-0.01, 0.04)
POUNCE_MISS_DIST = 0.7
WANDER_ACCEL_NOISE = 0.01
WANDER_CURIOSITY_BIAS = 0.15

# cat leave behavior
CLOSE_DIST = 0.20
LEAVE_ENTER_STEPS = 10
LEAVE_SPEED = 1.2
LEAVE_EXIT_DIST = 0.55
LEAVE_MAX_STEPS = 40
CAT_BORED_STEPS_THRESHOLD = 40
HOLD_DURATION = 30

# cat wobble during stalk
WOBBLE_FREQ_HZ_RANGE = (1.5, 4.0)
WOBBLE_AMP_RANGE = (0.15, 0.5)

# cat engagement tracking
CAT_STATIONARY_SPEED_THRESH = 0.25

# reward zones
CAPTURE_DIST = 0.06
DANGER_DIST = 0.15
PLAY_DIST_HI = 0.35
APPROACH_DIST = 0.65

# base reward values
CAPTURE_PENALTY = -3.0
DANGER_BASE_PENALTY = -1.5
PLAY_ZONE_REWARD = 1.5 # was 1.0, widen gap vs approach zone
TOO_FAR_PENALTY = -0.5
DODGE_BONUS = 2.0
REWARD_SCALER = 3.0

# teasing
MOUSE_TEASE_SPEED_THRESH = 0.10
TEASE_BONUS = 0.4
TEASE_STATIONARY_STEPS = 5

# capture / episode
MAX_CAPTURE_COUNT = 5

# anti wall hugging
WALL_MARGIN = 0.3
WALL_PENALTY = 0.3

# post capture cycle: struggle (panicked squeaking) -> dead (snack drop, silence) -> revive
STRUGGLE_DURATION = 20 # TODO: make longer?
STRUGGLE_SPEED_SCALE = 0.3
STRUGGLE_ESCAPE_BONUS = 0.5
PLAY_DEAD_DURATION = 30 # TODO: make longer?

# facing bonus (encourages keeping cat in FOV, on real robot means spin to find cat)
FACING_BONUS = 0.1

# survival reward
SURVIVAL_REWARD = 0.01

# freeze penalty (prevents sitting still in play zone exploit)
FREEZE_SPEED_THRESHOLD = 0.02
FREEZE_PENALTY = 0.2

# let cat win (time based confidence boost)
LET_CAT_WIN_STEPS = 150 # TODO: increase? rn 150 steps * 0.1 s/step = 15s
LET_CAT_WIN_SPEED_SCALE = 0.2
