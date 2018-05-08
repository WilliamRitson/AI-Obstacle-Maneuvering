# About
This repository is for the A.I Obstacle Maneuvering team in CMPS 140 (A.I) Spring 2018.
The goal is to use reinforcement learning to train a physics-based agent to maneuver over terrain and obstacles.
The repository contains agents designed to run on OpenAI gym environments.

# Requirements
The code runs in the `miniconda` environment (python 3.4) with the following additions

1. `source activate py34`
2. `pip install tensorflow`
3. `pip install keras`

Some gym environments may require additional packages.

## Running
You can run the example enviroment with `python test.py` the agents can be used on other enviroments by simply passing them the name of the desired enviroment.

# Bipedal Walker Info

https://github.com/openai/gym/wiki/BipedalWalker-v2


## Observation
Type: Box(24)

Num	Observation	Min	Max	Mean
0	hull_angle	0	2*pi	0.5
1	hull_angularVelocity	-inf	+inf	-
2	vel_x	-1	+1	-
3	vel_y	-1	+1	-
4	hip_joint_1_angle	-inf	+inf	-
5	hip_joint_1_speed	-inf	+inf	-
6	knee_joint_1_angle	-inf	+inf	-
7	knee_joint_1_speed	-inf	+inf	-
8	leg_1_ground_contact_flag	0	1	-
9	hip_joint_2_angle	-inf	+inf	-
10	hip_joint_2_speed	-inf	+inf	-
11	knee_joint_2_angle	-inf	+inf	-
12	knee_joint_2_speed	-inf	+inf	-
13	leg_2_ground_contact_flag	0	1	-
14-23	10 lidar readings	-inf	+inf	-


## Actions
Type: Box(4)

Num	Name	Min	Max
0	Hip_1 (Torque / Velocity)	-1	+1
1	Knee_1 (Torque / Velocity)	-1	+1
2	Hip_2 (Torque / Velocity)	-1	+1
3	Knee_2 (Torque / Velocity)	-1	+1


## Reward
Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.
