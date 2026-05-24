"""
passive piezo
sweeps frequency like a real mouse chirp instead of one constant beep
runs in a thread so the control loop doesn't stall during a squeak

usage:
    TODO

TODO: TODO
"""

import time, random, threading

BUZZER_PIN = 13
SQUEAK_COOLDOWN = 1.5 # don't squeak more than once per 1.5s

_ok = False
_pwm = None
_last = 0.0

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    _pwm = GPIO.PWM(BUZZER_PIN, 2000)
    _pwm.start(0) # start silent
    _ok = True
    print("[speaker] passive buzzer ready on GPIO 13")
except Exception as e:
    print(f"[speaker] not available: {e}")


def _chirp():
    # one chirp, sweep from a random high freq down to a lower one over 20-50ms
    # randomizing start/end freq makes each squeak sound slightly different
    duration = random.uniform(0.020, 0.050)
    start_freq = random.uniform(2500, 4000)
    end_freq = random.uniform(1200, 2500)
    steps = 15
    step_t = duration / steps

    _pwm.ChangeDutyCycle(50)
    for i in range(steps):
        freq = start_freq + (end_freq - start_freq) * (i / steps)
        _pwm.ChangeFrequency(max(50, int(freq)))
        time.sleep(step_t)
    _pwm.ChangeDutyCycle(0)


def _do_squeak():
    n = random.randint(1, 3)
    for i in range(n):
        _chirp()
        if i < n - 1:
            time.sleep(random.uniform(0.02, 0.07))


def maybe_squeak(dist, vx, vy, visible, now):
    global _last
    if not _ok or not visible:
        return
    if now - _last < SQUEAK_COOLDOWN:
        return
    speed = (vx**2 + vy**2) ** 0.5
    prob = min(0.3, speed * 0.5 / max(dist, 0.05))
    if random.random() < prob:
        _last = now
        threading.Thread(target=_do_squeak, daemon=True).start()
