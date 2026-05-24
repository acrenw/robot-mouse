"""
HC-SR04 ultrasonic distance sensor

wiring (bcm pin numbers):
    VCC -> pi 5V
    GND -> pi GND
    TRIG -> GPIO 5
    ECHO -> 1k resistor -> GPIO 6 (pin 31) -> 2k resistor -> GND

the echo pin outputs 5V but pi GPIO is 3.3V max, so the voltage divider is
required, skipping it will damage the pi over time

returns distance in metres
1.0 = clear / no echo
safety layer in motors.py triggers at 0.15m (15cm)

TODO: add left and right and back sensors for better obstacle avoidance
TODO: maybe average a few readings to reduce false triggers
"""

import time

TRIG_PIN = 5
ECHO_PIN = 6
MAX_RANGE = 1.0 # metres, anything beyond this = clear
TIMEOUT_S = 0.04 # 40ms max wait, corresponds to ~6.8m round trip

try:
    import RPi.GPIO as GPIO
    try:
        GPIO.setmode(GPIO.BCM)
    except Exception:
        pass # already set by motors.py, that's fine
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.05) # let sensor settle on first import
    _HW_AVAILABLE = True
    print(f"[sensors] HC-SR04 ready (TRIG=GPIO{TRIG_PIN}, ECHO=GPIO{ECHO_PIN})")
except (ImportError, RuntimeError):
    _HW_AVAILABLE = False
    print("[sensors] RPi.GPIO not available, sensor stubbed at 1.0m")


def read_front_m():
    """
    fire one ping and return distance to obstacle in metres
    1.0 = clear
    """
    if not _HW_AVAILABLE:
        return MAX_RANGE

    # sensor needs a clean low before the trigger pulse, then 10us high to fire
    GPIO.output(TRIG_PIN, False)
    time.sleep(2e-6) # brief low to ensure clean signal
    GPIO.output(TRIG_PIN, True)
    time.sleep(10e-6) # 10 microsecond pulse tells the sensor to fire
    GPIO.output(TRIG_PIN, False)

    # wait for echo pin to go high (sensor starts sending back the ultrasonic pulse)
    deadline = time.time() + TIMEOUT_S
    while GPIO.input(ECHO_PIN) == 0:
        if time.time() > deadline:
            return MAX_RANGE # took too long, assume clear
    t_start = time.time() # echo started, start timing

    # wait for echo pin to go low (pulse came back)
    deadline = time.time() + TIMEOUT_S
    while GPIO.input(ECHO_PIN) == 1:
        if time.time() > deadline:
            return MAX_RANGE # echo never ended, something weird happened
    t_end = time.time()

    # time the echo was high = how long sound took to travel to obstacle and back
    # distance = (time * speed of sound) / 2 because it's a round trip
    dist_m = (t_end - t_start) * 343.0 / 2.0
    return min(dist_m, MAX_RANGE) # cap at 1m, anything further is just "clear"
