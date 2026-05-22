"""
HC-SR04 sensor test

usage:
    python run/test_sensor.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensors import read_front_m
from motors import STOP_DIST

print(f"=== sensor test, ctrl-C to quit ===")
print(f"    safety stop triggers at < {STOP_DIST * 100:.0f}cm\n")

try:
    while True:
        d = read_front_m()
        bar = '#' * int(d * 40)
        warn = '  <- STOP!' if d < STOP_DIST else ('  <- slow' if d < 0.35 else '')
        print(f"  {d*100:5.1f}cm  {bar}{warn}", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\ndone.")
