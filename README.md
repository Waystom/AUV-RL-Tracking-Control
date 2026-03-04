# Reinforcement-Learning-Based AUV Trajectory Tracking

Reproduction of a reinforcement learning based adaptive PID controller for trajectory tracking of Autonomous Underwater Vehicles (AUV).

This project implements a **Q-learning based adaptive PID controller** for AUV trajectory tracking and evaluates its performance in simulation.

---

# Abstract

Trajectory tracking is a fundamental problem for autonomous underwater vehicles (AUVs). Traditional PID controllers are widely used due to their simplicity and robustness, but fixed PID parameters often perform poorly under uncertain underwater environments.

This project reproduces a reinforcement learning based control strategy that combines **Q-learning and PID control** to adaptively tune controller parameters. The reinforcement learning agent observes trajectory tracking errors and updates PID parameters to improve tracking performance.

Simulation results demonstrate that the proposed method can reduce tracking error and adapt controller parameters during operation.

---

# Paper

This project reproduces the method proposed in:

Yan Jing, Li Wenbiao, Yang Xian, et al.  
**Tracking Control for AUV by Combining Q-Learning and PID Controller**  
Journal of Unmanned Undersea Systems, 2021.

---

# Method

The control system integrates reinforcement learning with classical control theory.

The controller structure is:


Q-learning Agent
↓
Adaptive PID Controller
↓
AUV Dynamics Model
↓
Trajectory Tracking


### PID Controller

The PID controller generates control input based on trajectory error:


τ = Kp * e + Ki * ∫e dt + Kd * de/dt


where:

- **e** = tracking error  
- **Kp, Ki, Kd** = controller gains

---

### Reinforcement Learning

A **Q-learning algorithm** is used to adjust PID parameters.

State:


tracking error
vehicle position


Action:


adjust PID parameters
increase / decrease Kp Ki Kd


Reward:


tracking error penalty


The Q-table is updated iteratively during training.

---

# Simulation Environment

The simulation models a simplified AUV trajectory tracking task.

Target trajectory:


x(t) = 0.02t
y(t) = 50cos(0.01t)


The controller learns PID parameters to minimize trajectory tracking error.

---

# Results

## Training and Tracking Performance

![Training Results](results.png)

Top-left: training progress (average absolute error)

Top-right: X-axis trajectory tracking

Bottom-left: Y-axis trajectory tracking

Bottom-right: tracking errors

---

# Analysis

The training curve shows a gradual reduction in tracking error during reinforcement learning. The controller improves trajectory tracking performance as the Q-learning agent updates PID parameters.

The system successfully tracks the trajectory in the **X direction**, while larger errors remain in the **Y direction**, indicating that further improvements may be required in controller tuning or dynamics modeling.

These results are consistent with the observations reported in the original paper.

---

# Project Structure


AUV-RL-Tracking-Control

main.py # reinforcement learning simulation
results.png # training and tracking visualization
README.md


---

# Tech Stack

Python  
NumPy  
Matplotlib  
Reinforcement Learning (Q-learning)

---

# Future Work

Possible improvements include:

- Continuous control reinforcement learning methods (PPO, SAC)
- More accurate hydrodynamic AUV models
- 3D trajectory tracking
- Real-world robotic experiments
- Integration with ROS or PyBullet simulation

---

# Author

Wei Wenhao  
Robotics Engineering Undergraduate  
Northeastern University (China)

---

# License

MIT License
