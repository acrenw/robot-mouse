"""
all training constants for train_sac.py
"""

from shared.actor import ACT_DIM

# training hyperparameters
TOTAL_STEPS = 200_000
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
MAX_V = 0.3
MAX_OMEGA = 2.0

# simulation world
SIM_DT = 0.1
WORLD_X_MIN, WORLD_X_MAX = -2.0, 2.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 3.0
OBS_DIST_SCALE = 3.6
VISIBILITY_RANGE = 0.7
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

# cat leave behavior
CLOSE_DIST = 0.20
LEAVE_ENTER_STEPS = 10
LEAVE_SPEED = 1.2
LEAVE_EXIT_DIST = 0.55
LEAVE_MAX_STEPS = 40

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
CAPTURE_FINAL_REWARD = 1.5
CAPTURE_PENALTY = -2.0
DANGER_BASE_PENALTY = -1.0
PLAY_ZONE_REWARD = 1.0
TOO_FAR_PENALTY = -0.5
DODGE_BONUS = 1.0
REWARD_SCALER = 4.0

# teasing
MOUSE_TEASE_SPEED_THRESH = 0.10
TEASE_BONUS = 0.3
TEASE_STATIONARY_STEPS = 5

# freeze during pounce
POUNCE_FREEZE_PENALTY_BASE = 0.5
POUNCE_FREEZE_SCALE_DIVISOR = 20.0

# capture / episode
MAX_CAPTURE_COUNT = 5

# unpredictability bonus
# reward variance in recent actions so cat can't predict trajectory
UNPREDICTABILITY_WINDOW = 10
UNPREDICTABILITY_BONUS = 0.2
UNPREDICTABILITY_STD_NORM = 0.5

# smoothness penalty
# penalize jerky acceleration changes
SMOOTHNESS_PENALTY_WEIGHT = 0.15

# visibility maintenance bonus
# reward keeping cat in sensor range so mouse stays aware
VISIBILITY_BONUS = 0.1

# energy / speed cost
# small tax on speed to discourage max speed all the time
ENERGY_COST_WEIGHT = 0.05

# coverage / novelty bonus
# reward visiting new grid cells to encourage area exploration
COVERAGE_GRID_SIZE = 0.5
COVERAGE_BONUS = 0.15

# anti wall hugging
# penalty ramp when mouse is within WALL_MARGIN of world boundary
WALL_MARGIN = 0.3
WALL_PENALTY = 0.2

# tiredness simulation
# mouse gets slower over time, mimicking a real mouse getting tired
TIREDNESS_RATE = 0.002
TIREDNESS_RECOVERY_RATE = 0.001
MAX_TIREDNESS = 0.8
TIREDNESS_SPEED_SCALE = 0.6
TIREDNESS_POST_REVIVE = 0.3

# engagement bonus
# reward keeping cat in active states (stalking/pouncing = cat is engaged)
ENGAGEMENT_BONUS = 0.15

# play dead
# after capture, mouse goes still (mimics dying), then revives
PLAY_DEAD_DURATION = 30
PLAY_DEAD_COOLDOWN = 50

# survival reward
# small per step bonus for staying alive
SURVIVAL_REWARD = 0.05

# general freeze penalty
# penalize standing completely still (outside of play dead)
FREEZE_SPEED_THRESHOLD = 0.02
FREEZE_PENALTY = 0.1
