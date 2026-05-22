# robomouse

AI robot mouse that plays with cat using yolov8 detection + SAC reinforcement learning
runs on a raspberry pi 5 with a camera, TT motors, L9110 driver, and HC-SR04 sensor
TODO: try out AI camera, RPi Zero 2W, ...


## setup (laptop)

```bash
pip install -r requirements.txt
```

## training

```bash
python train/train_sac.py
```

runs for 200k steps
saves `shared/mouse_policy.pt`
watch `avg_r` and `avg_dist` (avg_dist should settle around 0.20-0.35 (play zone))

## test detection on video

```bash
python run/detect.py data/videos/pounce-5.mp4
```

should show bounding boxes and print state vectors
verify `vis=1` fires when cat is in frame and `dist`/`angle` values look reasonable

## test full pipeline on video (laptop)

```bash
python run/run.py --src data/videos/pounce-5.mp4
```

prints motor commands to terminal
use this to verify the policy behavior before putting it on the robot

## running on pi

```bash
# clone repo and install deps
git clone <repo-url> ~/mouse-proj
cd ~/mouse-proj
pip install -r requirements-pi.txt

# test motors first
python run/test_motors.py

# test sensor
python run/test_sensor.py

# run the full thing (use tmux so it survives SSH disconnect)
tmux new -s robot
python run/run.py --src /dev/video0 --headless
```