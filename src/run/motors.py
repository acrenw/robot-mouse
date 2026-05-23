"""
motor abstraction layer (l9110 driver)

on laptop: just prints commands
on pi: motors run

l9110 wiring:
    left motor IA -> GPIO 17
    left motor IB -> GPIO 27
    right motor IA -> GPIO 22
    right motor IB -> GPIO 23
    l9110 VCC -> pi 5V
    all GNDs -> common ground (pi GND + lipo GND + l9110 GND)

# TODO: measure actual wheelbase when chasis done and update WHEELBASE constant
# TODO: add encoder feedback for closed loop speed control if motors drift
"""

import time

MAX_V = 0.3 # m/s
MAX_OMEGA = 2.0 # rad/s
STOP_DIST = 0.15 # metres, safety layer hard stop
WHEELBASE = 0.10 # metres, distance between wheels (measure and update this)

# cap pwm if lipo larger than 3-6v
MOTOR_BATTERY_V = 3.7
MOTOR_MAX_V = 3.7
MAX_DUTY = min(MOTOR_MAX_V / MOTOR_BATTERY_V, 1.0) * 100  # 100% now

try:
    import RPi.GPIO as GPIO

    MOTOR_PINS = dict(
        left_ia=17, left_ib=27,
        right_ia=22, right_ib=23,
    )

    GPIO.setmode(GPIO.BCM)
    for pin in MOTOR_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)

    _pwm = {
        'left_ia': GPIO.PWM(MOTOR_PINS['left_ia'], 100),
        'left_ib': GPIO.PWM(MOTOR_PINS['left_ib'], 100),
        'right_ia': GPIO.PWM(MOTOR_PINS['right_ia'], 100),
        'right_ib': GPIO.PWM(MOTOR_PINS['right_ib'], 100),
    }
    for p in _pwm.values():
        p.start(0)

    _GPIO_AVAILABLE = True
    print("[motors] gpio ready, l9110 active")

except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False
    _pwm = {}


def _drive(side, frac):
    """
    drive one motor. frac is -1.0 to +1.0
    """
    frac = max(-1.0, min(1.0, frac)) # clamp so we never send >100% duty
    duty = abs(frac) * MAX_DUTY # convert fraction to duty cycle percentage, capped at ~81%
    if frac > 0:
        # forward: IA gets the PWM signal, IB stays low
        _pwm[f'{side}_ia'].ChangeDutyCycle(duty)
        _pwm[f'{side}_ib'].ChangeDutyCycle(0)
    elif frac < 0:
        # backward: IB gets the PWM signal, IA stays low
        _pwm[f'{side}_ia'].ChangeDutyCycle(0)
        _pwm[f'{side}_ib'].ChangeDutyCycle(duty)
    else:
        # stop: both low
        _pwm[f'{side}_ia'].ChangeDutyCycle(0)
        _pwm[f'{side}_ib'].ChangeDutyCycle(0)


def safety_layer(v_des, omega_des, sensor_front=1.0):
    """
    hard stop if something is too close in front.
    sensor_front is in metres, 1.0 = clear.
    runs independent of the RL agent so it can't be trained away.
    """
    if sensor_front < STOP_DIST:
        return 0.0, omega_des * 1.5  # spin in place instead of hitting it

    slow_zone = 0.35
    if sensor_front < slow_zone:
        # gradually slow down as we approach the stop distance
        scale = (sensor_front - STOP_DIST) / (slow_zone - STOP_DIST)
        v_des *= scale

    return v_des, omega_des


def send_to_motors(v, omega):
    """
    v: forward speed in m/s (negative = reverse)
    omega: angular rate in rad/s (positive = left turn)
    """
    if not _GPIO_AVAILABLE:
        print(f"  [motors] v={v:+.3f} m/s  w={omega:+.3f} rad/s")
        return

    v_frac = v / MAX_V
    omega_frac = omega / MAX_OMEGA
    left_frac = v_frac - omega_frac
    right_frac = v_frac + omega_frac

    _drive('left', left_frac)
    _drive('right', right_frac)


def stop():
    """
    cut all motor output and clean up gpio
    """
    if _GPIO_AVAILABLE:
        for p in _pwm.values():
            p.ChangeDutyCycle(0)
            p.stop()
        GPIO.cleanup()
    print("  [motors] stop")
