"""
usage:
    `python run/test_motors.py`
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from motors import send_to_motors, stop, MAX_V, MAX_OMEGA

DUTY = 0.8 # 80% speed for testing
PAUSE = 2 # seconds per move

tests = [
    ("forward",  DUTY * MAX_V, 0.0),
    ("backward", -DUTY * MAX_V, 0.0),
    ("turn left", 0.0, DUTY * MAX_OMEGA),
    ("turn right", 0.0, -DUTY * MAX_OMEGA),
]

print("=== motor test, watch the wheels ===\n")
try:
    for label, v, omega in tests:
        print(f"  {label}  (v={v:+.2f}, w={omega:+.2f}) for {PAUSE}s ...")
        send_to_motors(v, omega)
        time.sleep(PAUSE)
        send_to_motors(0, 0)
        time.sleep(0.4)
    print("\nall done.")
except KeyboardInterrupt:
    print("\nstopped.")
finally:
    stop()
