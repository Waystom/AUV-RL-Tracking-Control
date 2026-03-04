# -*- coding: utf-8 -*-
"""
AUV tracking control combining Q-learning and PID controller in PyBullet.
Reference: Yan Jing et al., Journal of Unmanned Undersea Systems, 2021, 29(5):565-574.
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import random

# ---------------------------- Parameters ----------------------------
SIMULATION_STEP = 0.05          # simulation step (s)
TOTAL_TIME = 60.0                # total simulation time per episode (s)
EPISODES = 1000                    # number of training episodes
TRAIN = True                     # training mode (False for fixed PID)

# AUV physical parameters (assume fully actuated for simplicity)
AUV_MASS = 100.0                 # kg
AUV_VOLUME = 0.2                 # m^3 (for buoyancy)
WATER_DENSITY = 1000.0           # kg/m^3
DRAG_COEFF_LINEAR = 10.0         # linear drag coefficient
DRAG_COEFF_ANGULAR = 5.0         # angular drag coefficient
MAX_FORCE_X = 500.0               # max longitudinal force (N)
MAX_FORCE_Y = 500.0               # max lateral force (N)
MAX_TORQUE_Z = 500.0              # max yaw torque (Nm)

# PID parameter ranges (for action discretization)
KP_RANGE = (0.0, 20.0)
KI_RANGE = (0.0, 5.0)
KD_RANGE = (0.0, 10.0)
ACTION_NUM = 3                    # number of discrete points per parameter -> total 27 actions

# Q-learning parameters
ALPHA = 0.1                       # learning rate
GAMMA = 0.8                       # discount factor
EPSILON_START = 0.5               # initial exploration rate
EPSILON_END = 0.05                # final exploration rate
EPSILON_DECAY = 0.995             # multiplicative decay per episode
STATE_BINS = 12                   # number of bins for each state dimension (error space)

# State space discretization based on errors e_x, e_y, e_phi
e_x_bins = np.linspace(-10, 10, STATE_BINS)      # x 目标变化小，缩小范围
e_y_bins = np.linspace(-60, 60, STATE_BINS)      # y 目标 ±50 -> 扩到 ±60
e_phi_bins = np.linspace(-np.pi, np.pi, STATE_BINS)
state_bins = [e_x_bins, e_y_bins, e_phi_bins]

# Target trajectory (Eq. 19 in the paper)
def target_trajectory(t):
    x_tar = 0.02 * t
    y_tar = 50.0 * np.cos(0.01 * t)
    phi_tar = 50.0 * np.sin(0.01 * t) * np.pi/180.0   # convert to radians
    return x_tar, y_tar, phi_tar

# ---------------------------- AUV Class ----------------------------
class AUV:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.id = None
        self.create_auv()

    def create_auv(self):
        # main body (box)
        col_shape = self.pb.createCollisionShape(self.pb.GEOM_BOX, halfExtents=[0.5, 0.2, 0.2])
        vis_shape = self.pb.createVisualShape(self.pb.GEOM_BOX, halfExtents=[0.5, 0.2, 0.2],
                                               rgbaColor=[0.2, 0.5, 0.8, 1])
        base_pos = [0, 0, -5]   # initial underwater position
        base_orn = self.pb.getQuaternionFromEuler([0, 0, 0])
        self.id = self.pb.createMultiBody(baseMass=AUV_MASS,
                                           baseCollisionShapeIndex=col_shape,
                                           baseVisualShapeIndex=vis_shape,
                                           basePosition=base_pos,
                                           baseOrientation=base_orn)

    def apply_thrust(self, force_x, force_y, torque_z):
        """Apply forces and torque at the center of mass (fully actuated assumption)"""
        force_x = np.clip(force_x, -MAX_FORCE_X, MAX_FORCE_X)
        force_y = np.clip(force_y, -MAX_FORCE_Y, MAX_FORCE_Y)
        torque_z = np.clip(torque_z, -MAX_TORQUE_Z, MAX_TORQUE_Z)

        pos, orn = self.pb.getBasePositionAndOrientation(self.id)
        rot = np.array(self.pb.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Convert force from body frame to world frame
        force_local = np.array([force_x, force_y, 0.0])
        force_world = rot.dot(force_local)

        # Apply force at center of mass
        self.pb.applyExternalForce(self.id, -1, force_world.tolist(), pos, self.pb.WORLD_FRAME)
        # Apply torque (yaw) in world frame (since torque is same in any frame for rigid body)
        self.pb.applyExternalTorque(self.id, -1, [0, 0, torque_z], self.pb.WORLD_FRAME)

    def apply_hydrodynamics(self):
        """Apply buoyancy, drag, and random disturbance"""
        pos, orn = self.pb.getBasePositionAndOrientation(self.id)

        # buoyancy (upward)
        buoyancy = WATER_DENSITY * AUV_VOLUME * 9.8
        self.pb.applyExternalForce(self.id, -1, [0, 0, buoyancy], pos, self.pb.WORLD_FRAME)

        # linear drag (opposite to velocity)
        vel, ang_vel = self.pb.getBaseVelocity(self.id)
        drag_force = [-DRAG_COEFF_LINEAR * v for v in vel]
        drag_torque = [-DRAG_COEFF_ANGULAR * w for w in ang_vel]
        self.pb.applyExternalForce(self.id, -1, drag_force, pos, self.pb.WORLD_FRAME)
        self.pb.applyExternalTorque(self.id, -1, drag_torque, self.pb.WORLD_FRAME)

        # random disturbance (simulate current)
        disturbance = [random.uniform(-20, 20) for _ in range(3)]
        self.pb.applyExternalForce(self.id, -1, disturbance, pos, self.pb.WORLD_FRAME)

    def reset(self):
        """Reset AUV to initial position and velocity"""
        self.pb.resetBasePositionAndOrientation(self.id, [0, 0, -5],
                                                 self.pb.getQuaternionFromEuler([0,0,0]))
        self.pb.resetBaseVelocity(self.id, [0,0,0], [0,0,0])

# ---------------------------- PID Controller ----------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# ---------------------------- Q-learning Agent ----------------------------
class QLearningAgent:
    def __init__(self, actions, state_bins, alpha=0.1, gamma=0.8, epsilon=0.2):
        self.actions = actions                     # list of actions: each is (Kp, Ki, Kd)
        self.state_bins = [np.asarray(b) for b in state_bins]  # ensure numpy arrays
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}                           # dict: (state_idx_tuple, action_idx) -> Q value

    def discretize_state(self, state_error):
        """Map continuous error vector to discrete indices"""
        idx = []
        for i, val in enumerate(state_error):
            bins = self.state_bins[i]
            if val <= bins[0]:
                idx.append(0)
            elif val >= bins[-1]:
                idx.append(len(bins)-1)
            else:
                idx.append(int(np.digitize(val, bins) - 1))
        return tuple(idx)

    def choose_action(self, state_idx):
        """epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        else:
            q_values = [self.q_table.get((state_idx, a), 0.0) for a in range(len(self.actions))]
            return int(np.argmax(q_values))

    def update(self, state_idx, action_idx, reward, next_state_idx):
        """Q-learning update rule"""
        old_q = self.q_table.get((state_idx, action_idx), 0.0)
        next_max = max([self.q_table.get((next_state_idx, a), 0.0) for a in range(len(self.actions))])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[(state_idx, action_idx)] = new_q

# ---------------------------- Main Program ----------------------------
def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Connect to PyBullet (GUI mode for training)
    physics_client = p.connect(p.GUI)
    is_gui = (physics_client == p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIMULATION_STEP)

    # Create action space: Kp, Ki, Kd discretized into ACTION_NUM points each
    kp_vals = np.linspace(KP_RANGE[0], KP_RANGE[1], ACTION_NUM)
    ki_vals = np.linspace(KI_RANGE[0], KI_RANGE[1], ACTION_NUM)
    kd_vals = np.linspace(KD_RANGE[0], KD_RANGE[1], ACTION_NUM)
    actions = [(kp, ki, kd) for kp in kp_vals for ki in ki_vals for kd in kd_vals]

    # Initialize Q-learning agent with decaying epsilon
    agent = QLearningAgent(actions, state_bins, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)

    # Record training progress
    episode_errors = []      # average absolute error per episode

    # Training loop
    for episode in range(EPISODES):
        # Decay epsilon multiplicatively for smoother exploration reduction
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")  # seafloor

        # Create AUV
        auv = AUV(p)

        # Random initial action if training, else fixed PID
        if TRAIN:
            # 为整个 episode 选择并固定一组 PID 参数（不在每个 time-step 改变）
            action_idx = random.randrange(len(actions))
            Kp, Ki, Kd = actions[action_idx]
        else:
            Kp, Ki, Kd = 10.0, 2.5, 5.0  # fixed PID

        # PID controllers for x, y, phi
        pid_x = PIDController(Kp, Ki, Kd, SIMULATION_STEP)
        pid_y = PIDController(Kp, Ki, Kd, SIMULATION_STEP)
        pid_phi = PIDController(Kp, Ki, Kd, SIMULATION_STEP)

        # 初始化每个 episode 的记录容器（保证缩进正确）
        rewards = []
        errors_x = []
        errors_y = []
        errors_phi = []
        # 计算步数并用整数循环，避免浮点累积
        n_steps = int(round(TOTAL_TIME / SIMULATION_STEP))  # 或 int(TOTAL_TIME // SIMULATION_STEP)
        for step in range(n_steps):
            t = step * SIMULATION_STEP

            # Current state
            pos, orn = p.getBasePositionAndOrientation(auv.id)
            euler = p.getEulerFromQuaternion(orn)
            x, y, _ = pos
            phi = euler[2]

            # Target at current time
            x_tar, y_tar, phi_tar = target_trajectory(t)
            phi_tar = np.arctan2(np.sin(phi_tar), np.cos(phi_tar))  # normalize

            # Errors
            e_x = x_tar - x
            e_y = y_tar - y
            e_phi = phi_tar - phi
            e_phi = np.arctan2(np.sin(e_phi), np.cos(e_phi))

            # 记录（用于统计平均误差）
            errors_x.append(abs(e_x))
            errors_y.append(abs(e_y))
            errors_phi.append(abs(e_phi))

            # Select action using Q-learning if training
            if TRAIN:
                state_error = [e_x, e_y, e_phi]
                state_idx = agent.discretize_state(state_error)
                action_idx = agent.choose_action(state_idx)
                Kp, Ki, Kd = actions[action_idx]
                pid_x.Kp, pid_x.Ki, pid_x.Kd = Kp, Ki, Kd
                pid_y.Kp, pid_y.Ki, pid_y.Kd = Kp, Ki, Kd
                pid_phi.Kp, pid_phi.Ki, pid_phi.Kd = Kp, Ki, Kd

            # Compute control signals
            control_x = pid_x.compute(e_x)
            control_y = pid_y.compute(e_y)
            control_phi = pid_phi.compute(e_phi)

            # Apply thrust (fully actuated: forces in x, y and torque in z)
            auv.apply_thrust(control_x, control_y, control_phi)

            # Apply hydrodynamic forces
            auv.apply_hydrodynamics()

            # Step simulation
            p.stepSimulation()
            if is_gui:
                time.sleep(max(0.0, SIMULATION_STEP / 10))

            # Reward: weighted squared errors + small penalty on large control outputs
            # - weight heading error more (scale factor), penalize aggressive control to encourage smoothness
            heading_scale = 10.0
            control_penalty = 1e-3 * (abs(control_x) + abs(control_y) + abs(control_phi))
            reward = - (e_x**2 + e_y**2 + (heading_scale * e_phi)**2) - control_penalty
            rewards.append(reward)

            # If training, update Q-table using next state and next target
            if TRAIN:
                # Get next state
                pos_next, orn_next = p.getBasePositionAndOrientation(auv.id)
                euler_next = p.getEulerFromQuaternion(orn_next)
                x_next, y_next = pos_next[0], pos_next[1]
                phi_next = euler_next[2]

                # Target at next time step (important: use t+dt)
                x_tar_next, y_tar_next, phi_tar_next = target_trajectory(t + SIMULATION_STEP)
                phi_tar_next = np.arctan2(np.sin(phi_tar_next), np.cos(phi_tar_next))

                e_x_next = x_tar_next - x_next
                e_y_next = y_tar_next - y_next
                e_phi_next = phi_tar_next - phi_next
                e_phi_next = np.arctan2(np.sin(e_phi_next), np.cos(e_phi_next))
                next_state_error = [e_x_next, e_y_next, e_phi_next]
                next_state_idx = agent.discretize_state(next_state_error)

                agent.update(state_idx, action_idx, reward, next_state_idx)

        # Compute average error for this episode (防止空列表导致 NaN/警告)
        if len(errors_x) > 0:
            avg_error = (np.mean(errors_x) + np.mean(errors_y) + np.mean(errors_phi)) / 3.0
        else:
            avg_error = float('nan')
        episode_errors.append(avg_error)
        print(f"Episode {episode+1}/{EPISODES} completed. Avg error = {avg_error:.4f}")

    p.disconnect()

    # ---------------------------- Evaluation and Plotting ----------------------------
    print("\nRunning evaluation with learned policy...")

    # Reconnect in DIRECT mode (no GUI) for faster evaluation
    physics_client = p.connect(p.DIRECT)
    is_gui = (physics_client == p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIMULATION_STEP)
    plane_id = p.loadURDF("plane.urdf")
    auv = AUV(p)

    # Initialize PID controllers (will be updated by learned policy)
    pid_x = PIDController(10.0, 2.5, 5.0, SIMULATION_STEP)
    pid_y = PIDController(10.0, 2.5, 5.0, SIMULATION_STEP)
    pid_phi = PIDController(10.0, 2.5, 5.0, SIMULATION_STEP)

    t = 0.0
    positions_x, positions_y = [], []
    targets_x, targets_y = [], []

    # Evaluation: use fixed step count to avoid float accumulation
    n_steps_eval = int(round(TOTAL_TIME / SIMULATION_STEP))
    for step in range(n_steps_eval):
        t = step * SIMULATION_STEP
        pos, orn = p.getBasePositionAndOrientation(auv.id)
        euler = p.getEulerFromQuaternion(orn)
        x, y = pos[0], pos[1]
        phi = euler[2]
        x_tar, y_tar, phi_tar = target_trajectory(t)
        phi_tar = np.arctan2(np.sin(phi_tar), np.cos(phi_tar))

        e_x = x_tar - x
        e_y = y_tar - y
        e_phi = phi_tar - phi
        e_phi = np.arctan2(np.sin(e_phi), np.cos(e_phi))

        # choose one greedy PID for the whole evaluation run (if training was used)
        if TRAIN and EPISODES > 0:
            pos0, orn0 = p.getBasePositionAndOrientation(auv.id)
            euler0 = p.getEulerFromQuaternion(orn0)
            x0, y0 = pos0[0], pos0[1]
            phi0 = euler0[2]
            x_tar0, y_tar0, phi_tar0 = target_trajectory(0.0)
            phi_tar0 = np.arctan2(np.sin(phi_tar0), np.cos(phi_tar0))
            state_error0 = [x_tar0 - x0, y_tar0 - y0, np.arctan2(np.sin(phi_tar0 - phi0), np.cos(phi_tar0 - phi0))]
            state_idx0 = agent.discretize_state(state_error0)
            q_values = [agent.q_table.get((state_idx0, a), 0.0) for a in range(len(actions))]
            action_idx = int(np.argmax(q_values))
            Kp, Ki, Kd = actions[action_idx]
            pid_x.Kp, pid_x.Ki, pid_x.Kd = Kp, Ki, Kd
            pid_y.Kp, pid_y.Ki, pid_y.Kd = Kp, Ki, Kd
            pid_phi.Kp, pid_phi.Ki, pid_phi.Kd = Kp, Ki, Kd

        control_x = pid_x.compute(e_x)
        control_y = pid_y.compute(e_y)
        control_phi = pid_phi.compute(e_phi)
        auv.apply_thrust(control_x, control_y, control_phi)
        auv.apply_hydrodynamics()
        p.stepSimulation()

        positions_x.append(x)
        positions_y.append(y)
        targets_x.append(x_tar)
        targets_y.append(y_tar)

        t += SIMULATION_STEP

    p.disconnect()

    # Plotting
    plt.figure(figsize=(12, 8))

    # Training error curve
    plt.subplot(2, 2, 1)
    plt.plot(episode_errors, 'b-')
    plt.xlabel('Episode')
    plt.ylabel('Average Absolute Error')
    plt.title('Training Progress')
    plt.grid(True)

    # Trajectories
    time_axis = np.arange(0, TOTAL_TIME, SIMULATION_STEP)[:len(positions_x)]
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, positions_x, 'r-', label='AUV X')
    plt.plot(time_axis, targets_x, 'b--', label='Target X')
    plt.xlabel('Time (s)')
    plt.ylabel('X position (m)')
    plt.legend()
    plt.grid(True)
    plt.title('X Trajectory')

    plt.subplot(2, 2, 3)
    plt.plot(time_axis, positions_y, 'r-', label='AUV Y')
    plt.plot(time_axis, targets_y, 'b--', label='Target Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Y Trajectory')

    plt.subplot(2, 2, 4)
    err_x = np.array(targets_x) - np.array(positions_x)
    err_y = np.array(targets_y) - np.array(positions_y)
    plt.plot(time_axis, err_x, label='Error X')
    plt.plot(time_axis, err_y, label='Error Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Tracking Errors')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()