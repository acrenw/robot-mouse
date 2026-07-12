"""
mg90s snack servo
moves to dispense snacks

TODO: tune OPEN_DUTY once the servo is actually mounted, 
      the right angle depends on which way the horn is pointing when you screw it down
"""

import time

SERVO_PIN = 12
CLOSED_DUTY = 7.5 # ~1.5ms pulse, neutral (treat stays in)
OPEN_DUTY = 12.5 # ~2.5ms pulse, open (treat drops out)

_ok = False
_pwm = None

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    _pwm = GPIO.PWM(SERVO_PIN, 50)
    _pwm.start(0) # start silent, only pulse when moving, stops jitter
    _ok = True
    print("[servo] mg90s ready on GPIO 12")
except Exception as e:
    print(f"[servo] not available: {e}")


def dispense():
    if not _ok:
        print("[servo] dispense called (no hardware)")
        return
    _pwm.ChangeDutyCycle(OPEN_DUTY)
    time.sleep(0.6)
    _pwm.ChangeDutyCycle(CLOSED_DUTY)
    time.sleep(0.3)
    _pwm.ChangeDutyCycle(0) # stop pulsing so servo doesn't jitter
