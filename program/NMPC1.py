import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

# simulation time parameter
SIM_TIME = 62.
TIMESTEP = 0.5
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)

# collision cost
Qc = 5.0
kappa = 4.0

# ego vehicle parameter
ROBOT_RADIUS = 0.5

# ego motion parameter
VMAX = 4

# nmpc parameter
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 3
lower_bound = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 3


def CollisionCost(p_robot, p_obs):
    """
    Cost of collision between two robot_state
    """
    d = np.linalg.norm(p_robot - p_obs)
    cost = Qc / (1 + np.exp(kappa * (d - 2 * ROBOT_RADIUS)))
    return cost


def TatalCollisionCost(path_robot, obstacles):
    total_cost = 0.0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            p_obs = obstacles[j]
            p_rob = path_robot[3 * i:3 * i + 3]
            total_cost += CollisionCost(p_rob, p_obs)
    return total_cost


def TrackingCost(x, xref):
    return np.linalg.norm(x - xref)


def UpdateState(x0, u, timestep):
    """
    Computes the states of the system after applying a sequence of control signals u on
    initial state x0
    """
    N = int(len(u) / 3)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(3))

    new_state = np.vstack([np.eye(3)] * int(N)) @ x0 + kron @ u * timestep

    return new_state


def TotalCost(u, robot_state, obstacles, xref):
    p_robot = UpdateState(robot_state, u, NMPC_TIMESTEP)
    c1 = TrackingCost(p_robot, xref)
    c2 = TatalCollisionCost(p_robot, obstacles)
    total = c1 + c2
    return total


def ComputeVelocity(robot_state, obstacles, xref):
    """
    Computes control velocity of the copter
    """
    u0 = np.random.rand(3 * HORIZON_LENGTH)

    def CostFn(u):
        return TotalCost(u, robot_state, obstacles, xref)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(CostFn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:3]
    return velocity, res.x


def ComputeRefPath(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start)
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape(
        (3 * number_of_steps))


def NMPCLeader(start_pose, goal_pose, obstacles):
    robot_state = start_pose
    p_desired = goal_pose
    robot_state_history = np.empty((3, 0))

    for i in range(NUMBER_OF_TIMESTEPS):
        ref_path = ComputeRefPath(robot_state, p_desired, HORIZON_LENGTH,
                                  NMPC_TIMESTEP)
        vel, velocity_profile = ComputeVelocity(robot_state, obstacles,
                                                ref_path)
        robot_state = UpdateState(robot_state, vel, TIMESTEP)

        robot_state_history = np.hstack(
            (robot_state_history, robot_state.reshape(-1, 1)))
        dis_to_goal = np.linalg.norm(goal_pose - robot_state)
        if dis_to_goal < 0.4:
            print("final distance to goal:", dis_to_goal)
            break

    return robot_state_history


if __name__ == "__main__":
    start_pose = np.array([0, 0, 0])
    goal_pose = np.array([10, 10, 10])
    obstacles = np.array([[4, 4, 4], [6, 6, 6]])
    path = NMPCLeader(start_pose, goal_pose, obstacles)

    # 创建球的参数
    u = np.linspace(0, 2 * np.pi, 100)  # 从 0 到 2π
    v = np.linspace(0, np.pi, 100)  # 从 0 到 π

    # 使用参数方程计算球面上的点
    x0_a = obstacles[0, 0] + ROBOT_RADIUS * np.outer(
        np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
    y0_a = obstacles[0, 1] + ROBOT_RADIUS * np.outer(
        np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
    z0_a = obstacles[0, 2] + ROBOT_RADIUS * np.outer(np.ones(
        np.size(u)), np.cos(v))  # z = z0 + r * cos(φ)

    x1_a = obstacles[1, 0] + ROBOT_RADIUS * np.outer(
        np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
    y1_a = obstacles[1, 1] + ROBOT_RADIUS * np.outer(
        np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
    z1_a = obstacles[1, 2] + ROBOT_RADIUS * np.outer(np.ones(
        np.size(u)), np.cos(v))  # z = z0 + r * cos(φ)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维线条
    ax.plot(path[0], path[1], path[2], label='3D line')

    # 绘制球
    ax.plot_surface(x0_a, y0_a, z0_a, color='b', alpha=0.6)  # 使用半透明的蓝色
    ax.plot_surface(x1_a, y1_a, z1_a, color='b', alpha=0.6)  # 使用半透明的蓝色

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
