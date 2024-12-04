import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

# simulation time parameter
SIM_TIME = 100
TIMESTEP = 0.5
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)

# collision cost
Qc = 1.0
kappa = 1.0

# weight
tracking_weight = 1.0
collsion_weight = 1.1
over_height_weight = 1.0
smoothness_weight = 1000.0

# ego motion parameter
VMAX = 1.0

# nmpc parameter
HORIZON_LENGTH = int(8)
NMPC_TIMESTEP = 0.3
upper_bound_default = [(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 3
lower_bound_default = [-(1 / np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 3


def CollisionCost(p_robot, p_obs, safe_dis):
    """
    Cost of collision between two robot_state
    """
    d = np.linalg.norm(p_robot - p_obs)
    cost = Qc / (1 + np.exp(kappa * (d - 2 * safe_dis)))

    return cost


def TatalCollisionCost(path_robot, obstacles, static_safe_dis):
    total_cost = 0.0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            p_obs = obstacles[j]
            p_rob = path_robot[3 * i:3 * i + 3]
            total_cost += CollisionCost(p_rob, p_obs, static_safe_dis)
    return total_cost


def OverZlimitCost(p_robot, z_limits):
    cost = 0.0
    if p_robot[-1] > z_limits[-1]:
        cost += 1.0
    if p_robot[-1] < z_limits[0]:
        cost += 1.0

    ground_proximity_cost = 0.1 / (p_robot[-1] - z_limits[0] + 0.1)  # 避免除以零
    cost += ground_proximity_cost

    ceiling_proximity_cost = 0.1 / (z_limits[-1] - p_robot[-1] + 0.1)
    cost += ceiling_proximity_cost

    return cost


def TotalOverZlimitCost(path_robot, z_limits):
    total_cost = 0.0
    for i in range(HORIZON_LENGTH):
        p_rob = path_robot[3 * i:3 * i + 3]
        total_cost += OverZlimitCost(p_rob, z_limits)

    return total_cost


def SmoothnessCost(path_robot):
    """
    Calculate the smoothness cost based on velocity change.
    """
    cost = 0.0
    for i in range(1, HORIZON_LENGTH - 1):
        p_prev = path_robot[3 * (i - 1):3 * i]
        p_curr = path_robot[3 * i:3 * (i + 1)]
        p_next = path_robot[3 * (i + 1):3 * (i + 2)]

        # Calculate the angle between consecutive segments
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        angle_cos = np.dot(
            v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cost += (1 - angle_cos)  # Larger angles result in higher cost
    return cost


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


def TotalCost(u, robot_state, obstacles, xref, static_safe_dis, z_limits):
    p_robot = UpdateState(robot_state, u, NMPC_TIMESTEP)
    c1 = TrackingCost(p_robot, xref) * tracking_weight
    c2 = TatalCollisionCost(p_robot, obstacles,
                            static_safe_dis) * collsion_weight
    c3 = TotalOverZlimitCost(p_robot, z_limits) * over_height_weight
    # c4 = SmoothnessCost(p_robot) * smoothness_weight
    # print(f"c1:{c1} c2:{c2} c3:{c3} c4:{c4}")
    total = c1 + c2 + c3

    return total


def ComputeVelocity(robot_state, obstacles, xref, static_safe_dis, lower_bound,
                    upper_bound, z_limits):
    """
    Computes control velocity of the copter
    """
    u0 = np.random.rand(3 * HORIZON_LENGTH)

    def CostFn(u):
        return TotalCost(u, robot_state, obstacles, xref, static_safe_dis,
                         z_limits)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(CostFn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:3]
    return velocity, res.x


def ComputeRefPath(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start)
    norm = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / norm

    if norm < 0.1:
        new_goal = goal
    else:
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape(
        (3 * number_of_steps))


def NMPCLeader(start_pose, goal_pose, obstacles, static_safe_dis, z_limits):
    robot_state = start_pose
    p_desired = goal_pose
    robot_state_history = np.empty((3, 0))
    robot_state_history = np.hstack(
        (robot_state_history, start_pose.reshape(-1, 1)))

    final_step = 0
    vel_list = []
    for i in range(NUMBER_OF_TIMESTEPS):
        dis_to_goal = np.linalg.norm(goal_pose - robot_state)
        upper_bound = upper_bound_default
        lower_bound = lower_bound_default
        if dis_to_goal >= 1.0 and dis_to_goal < 10.0:
            upper_bound = [0.5] * HORIZON_LENGTH * 3
            lower_bound = [-0.5] * HORIZON_LENGTH * 3
        if dis_to_goal < 1.0:
            upper_bound = [0.1] * HORIZON_LENGTH * 3
            lower_bound = [-0.1] * HORIZON_LENGTH * 3
        final_step = i
        ref_path = ComputeRefPath(robot_state, p_desired, HORIZON_LENGTH,
                                  NMPC_TIMESTEP)
        vel, velocity_profile = ComputeVelocity(robot_state, obstacles,
                                                ref_path, static_safe_dis,
                                                lower_bound, upper_bound,
                                                z_limits)
        vel_list.append(vel.tolist() + [np.linalg.norm(vel)])
        robot_state = UpdateState(robot_state, vel, TIMESTEP)

        robot_state_history = np.hstack(
            (robot_state_history, robot_state.reshape(-1, 1)))
        if dis_to_goal < 0.1:
            print("final_step:", final_step, "final distance to goal:",
                  dis_to_goal)
            break

    return robot_state_history, final_step, vel_list, dis_to_goal


def set_axes_equal(ax):
    '''使 3D 图中的坐标轴比例相等'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # 计算每个轴的范围
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    # 计算各轴的中心点
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # 设置范围，使得各轴比例一致
    ax.set_xlim(x_middle - max_range / 2, x_middle + max_range / 2)
    ax.set_ylim(y_middle - max_range / 2, y_middle + max_range / 2)
    ax.set_zlim(z_middle - max_range / 2, z_middle + max_range / 2)


if __name__ == "__main__":
    start_pose = np.array([0, 0, 1.3])
    goal_pose = np.array([5.0, 5.0, 1.3])
    obstacles = np.array([[1.8, 1.8, 1.2], [3.7, 3.7, 1.5]])
    z_limits = np.array([0.4, 1.8])
    obs_rad = 0.2
    path, final_step, val, dis = NMPCLeader(start_pose, goal_pose, obstacles,
                                            obs_rad, z_limits)

    # 创建球的参数
    u = np.linspace(0, 2 * np.pi, 100)  # 从 0 到 2π
    v = np.linspace(0, np.pi, 100)  # 从 0 到 π

    # 使用参数方程计算球面上的点
    x0_a = obstacles[0, 0] + obs_rad * np.outer(
        np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
    y0_a = obstacles[0, 1] + obs_rad * np.outer(
        np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
    z0_a = obstacles[0, 2] + obs_rad * np.outer(np.ones(
        np.size(u)), np.cos(v))  # z = z0 + r * cos(φ)

    x1_a = obstacles[1, 0] + obs_rad * np.outer(
        np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
    y1_a = obstacles[1, 1] + obs_rad * np.outer(
        np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
    z1_a = obstacles[1, 2] + obs_rad * np.outer(np.ones(
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
    set_axes_equal(ax)
    ax.legend()

    plt.show()
