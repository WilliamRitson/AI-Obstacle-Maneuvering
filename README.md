# AI Obstacle Maneuvering

The goal of this project is to use reinforcement learning to train a physics-based agent to maneuver over terrain and obstacles. The repository contains agents designed to run on OpenAI gym environments.

BipedalWalker-v2  |  BipedalWalkerHardcore-v2
:-------------------------:|:-------------------------:
![](demos/BipedalWalker-v2.gif)  |  TBD

## Requirements

### Unix
1. Python 3.4 (and up) in Anaconda environment
2. `pip install -r requirements.txt`

### Windows
1. Use python 3.4 in Anaconda environment ([here's simple directions](https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md))
2. `conda install -c https://conda.anaconda.org/kne pybox2d`
3. `pip install -r requirements_windows.txt`

Some gym environments may require additional packages.

## Running
You can run the example enviroment with `python test.py`. The agents/environments can be modified in this file.

# Bipedal Walker Environment

https://github.com/openai/gym/wiki/BipedalWalker-v2


## Observation

Type: Box(24)

Num   | Observation                |  Min   |   Max  | Mean
------|----------------------------|--------|--------|------   
0     | hull_angle                 |  0     |  2*pi  |  0.5
1     | hull_angularVelocity       |  -inf  |  +inf  |  -
2     | vel_x                      |  -1    |  +1    |  -
3     |  vel_y                     |  -1    |  +1    |  -
4     | hip_joint_1_angle          |  -inf  |  +inf  |  -
5     | hip_joint_1_speed          |  -inf  |  +inf  |  -
6     | knee_joint_1_angle         |  -inf  |  +inf  |  -
7     | knee_joint_1_speed         |  -inf  |  +inf  |  -
8     | leg_1_ground_contact_flag  |  0     |  1     |  -
9     | hip_joint_2_angle          |  -inf  |  +inf  |  -
10    | hip_joint_2_speed          |  -inf  |  +inf  |  -
11    | knee_joint_2_angle         |  -inf  |  +inf  |  -
12    | knee_joint_2_speed         |  -inf  |  +inf  |  -
13    | leg_2_ground_contact_flag  |  0     |  1     |  -
14-23 | 10 lidar readings          |  -inf  |  +inf  |  -


## Actions

Type: Box(4) - Torque control(default) / Velocity control - Change inside `/envs/box2d/bipedal_walker.py` line 363

Num | Name                        | Min  | Max  
----|-----------------------------|------|------
0   | Hip_1 (Torque / Velocity)   |  -1  | +1
1   | Knee_1 (Torque / Velocity)  |  -1  | +1
2   | Hip_2 (Torque / Velocity)   |  -1  | +1
3   | Knee_2 (Torque / Velocity)  |  -1  | +1

## Reward

Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.
## Starting State

Random position upright and mostly straight legs.

## Episode Termination

The episode ends when the robot body touches ground or the robot reaches far right side of the environment.

## Solved Requirements

BipedalWalker-v2 defines "solving" as getting average reward of 300 over 100 consecutive trials