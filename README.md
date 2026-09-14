# ML-powered autonomous robotic mouse

I built an ML-powered autonomous robotic mouse designed to chase and play with cats, running on a Raspberry Pi 5 with a Pi AI Camera for perception.

The system uses YOLOv8s for cat detection, feeding a SAC reinforcement learning agent that outputs motor commands, and an independent obstacle avoidance safety layer decoupled from the learned policy.

The SAC agent was trained in simulation for 400k episodes against an 5-state cat FSM calibrated from real cat video traces, using a composite reward function.\n\nOn the hardware side, dual L9110 motor drivers drive TT DC gear motors, an HC-SR04 ultrasonic sensor supports obstacle detection, and the system is powered by an 18650 LiPo cell through a boost/charge module. There's also a piezo buzzer for mimicking mouse squeaks and a servo that drops snacks when the mouse is captured.

Future updates: new buck converter, reshape reward function, print an actual case.
