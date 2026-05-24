"""
interactive servo tuner

usage:
    `python run/test_servo.py`
"""

import time, sys
sys.path.insert(0, '.')

from servo import _ok, _pwm, CLOSED_DUTY, OPEN_DUTY

if not _ok:
    print("no servo hardware found, check wiring and GPIO 12")
    sys.exit(1)

print(f"servo ready (closed={CLOSED_DUTY}, open={OPEN_DUTY})")
print("commands: o=open, c=close, d=dispense, <number>=set duty, q=quit")
print("example: type '10.5' to test duty cycle 10.5\n")

while True:
    cmd = input("> ").strip().lower()
    if cmd == 'q':
        _pwm.ChangeDutyCycle(CLOSED_DUTY)
        time.sleep(0.3)
        _pwm.ChangeDutyCycle(0)
        break
    elif cmd == 'o':
        print(f"opening (duty={OPEN_DUTY})")
        _pwm.ChangeDutyCycle(OPEN_DUTY)
    elif cmd == 'c':
        print(f"closing (duty={CLOSED_DUTY})")
        _pwm.ChangeDutyCycle(CLOSED_DUTY)
    elif cmd == 'd':
        print("dispensing...")
        from servo import dispense
        dispense()
        print("done")
    else:
        try:
            duty = float(cmd)
            print(f"setting duty={duty}")
            _pwm.ChangeDutyCycle(duty)
        except ValueError:
            print("Didn't get that. Try o, c, d, a number like 10.5, or q")
