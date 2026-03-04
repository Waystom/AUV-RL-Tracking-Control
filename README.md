# AUV-RL-Tracking-Control

Reproduction of the reinforcement learning based adaptive PID controller for trajectory tracking of Autonomous Underwater Vehicles (AUV).

This project implements a Q-learning based PID parameter tuning method for AUV trajectory tracking and evaluates the control performance in simulation.

---

## Paper

This project reproduces the method proposed in:

Yan Jing et al.  
**Tracking Control for AUV by Combining Q-Learning and PID Controller**  
Journal of Unmanned Undersea Systems, 2021.

---

## Method Overview

The controller combines a classical PID controller with reinforcement learning.

- PID controller generates control input for AUV trajectory tracking.
- Q-learning is used to adaptively tune the PID parameters.
- The system learns optimal control parameters based on trajectory tracking errors.

Control structure:

---

## Features

- Reinforcement learning based control
- Adaptive PID parameter tuning
- AUV trajectory tracking simulation
- Training visualization
- Tracking error analysis

---

## Simulation Results

Example training and tracking results:

- Training convergence curve
- AUV trajectory tracking performance
- Tracking error analysis

Example figures:

- Training progress
- X trajectory tracking
- Y trajectory tracking
- Tracking error

---

## Tech Stack

Python  
PyBullet  
NumPy  
Matplotlib  

---

## Future Work

- Extend to continuous control methods (PPO / SAC)
- More realistic AUV hydrodynamic model
- 3D trajectory tracking
- Real-world AUV experiments

---

## Author

Wei Wenhao  
Robotics Engineering Undergraduate  
Northeastern University (China)
## Project Structure
main.py # main simulation
controller.py # PID + RL controller
auv_env.py # AUV simulation environment
train.py # RL training process
plots.py # visualization
