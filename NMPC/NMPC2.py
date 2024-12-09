import numpy as np
from scipy.optimize import minimize, Bounds
import math

# simulation time parameter
SIM_TIME = 100
TIMESTEP = 0.5
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)

# collision cost
Qc = 1.
kappa = 1.

# ego motion parameter
VMAX = 4

# weight
tracking_weight = 1.0
collsion_weight = 1.3
over_height_weight = 1.0
input_change_weight = 2.0

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


def TrackingCost(p_robot, p_stitch):
    """
    Cost of track reference_line
    """
    return np.linalg.norm(p_robot - p_stitch)


def TatalCollisionCost(path_robot, dynamic_obstacles, static_obstacles,
                       dynamic_safe_dis, static_safe_dis):
    total_cost = 0.0
    for i in range(HORIZON_LENGTH):
        for j in range(len(static_obstacles)):
            p_static_obs = static_obstacles[j]
            p_rob = path_robot[3 * i:3 * i + 3]
            total_cost += CollisionCost(p_rob, p_static_obs, static_safe_dis)
        # for k in range(len(dynamic_obstacles)):
        #     p_dynamic_obs = dynamic_obstacles[3 * i:3 * i + 3]
        #     p_rob = path_robot[3 * i:3 * i + 3]
        #     total_cost += CollisionCost(p_rob, p_dynamic_obs, dynamic_safe_dis)
    return total_cost


def OverZlimitCost(p_robot, z_limits):
    if p_robot[-1] > z_limits[-1] or p_robot[-1] < z_limits[0]:
        return 1.0

    cost = 0.0
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


def InputChangeCost(u):
    cost = 0.0
    for i in range(HORIZON_LENGTH - 1):
        delta_u = u[3 * (i + 1):3 * (i + 2)] - u[3 * i:3 * (i + 1)]
        cost += np.linalg.norm(delta_u)**2

    return cost


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


def TotalCost(u, robot_state, dynamic_obs, static_obs, dynamic_obs_safe_dis,
              static_obs_safe_dis, xref, z_limits):
    p_robot = UpdateState(robot_state, u, NMPC_TIMESTEP)
    c1 = TrackingCost(p_robot, xref) * tracking_weight
    c2 = TatalCollisionCost(p_robot, dynamic_obs, static_obs,
                            dynamic_obs_safe_dis,
                            static_obs_safe_dis) * collsion_weight
    c3 = TotalOverZlimitCost(p_robot, z_limits) * over_height_weight
    c4 = InputChangeCost(u) * input_change_weight

    total = c1 + c2 + c3 + c4
    return total


def ComputeVelocity(robot_state, neighbor_traj, static_obs, neighbor_safe_dis,
                    avoid_static_obs_dis, xref, lower_bound, upper_bound,
                    z_limits):
    """
    Computes control velocity of the copter
    """
    u0 = np.random.rand(3 * HORIZON_LENGTH)

    def CostFn(u):
        return TotalCost(u, robot_state, neighbor_traj, static_obs,
                         neighbor_safe_dis, avoid_static_obs_dis, xref,
                         z_limits)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(CostFn, u0, method='SLSQP', bounds=bounds)

    velocity = res.x[:3]
    return velocity, res.x


def ComputeRefPath(start, final_goal, ref_trajectory, time_stamp,
                   number_of_steps, timestep):
    virtual_target = final_goal
    ref_total_time = ref_trajectory.shape[1]

    final_ref_path = np.linspace(virtual_target, virtual_target,
                                 number_of_steps).reshape(
                                     (3 * number_of_steps))

    if time_stamp < ref_total_time:
        virtual_target = ref_trajectory[:, time_stamp].T
        final_ref_path = np.linspace(start, virtual_target,
                                     number_of_steps).reshape(
                                         (3 * number_of_steps))

    return final_ref_path


def GetNeighbourTraj(neighbour_trajectory, time_stamp, number_of_steps,
                     timestep):
    if neighbour_trajectory.size == 0:
        return []
    pre_neighbour_traj = np.empty((3, 0))
    for i in range(number_of_steps):
        next_sim_time_ceil = time_stamp + math.ceil(
            (i + 1) * timestep / TIMESTEP)
        next_sim_time_floor = time_stamp + math.floor(
            (i + 1) * time_stamp / TIMESTEP)
        if next_sim_time_floor >= neighbour_trajectory.shape[1] - 1:
            pre_neighbour_traj = np.hstack(
                (pre_neighbour_traj, neighbour_trajectory[:,
                                                          -1].reshape(-1, 1)))
        else:
            ratio = ((i + 1) * timestep -
                     (next_sim_time_floor - timestep) * TIMESTEP) / TIMESTEP
            tmp_pose = neighbour_trajectory[:, next_sim_time_floor] + (
                neighbour_trajectory[:, next_sim_time_ceil] -
                neighbour_trajectory[:, next_sim_time_floor]) * ratio
            pre_neighbour_traj = np.hstack(
                (pre_neighbour_traj, tmp_pose.reshape(-1, 1)))

    return pre_neighbour_traj.reshape(-1, order="F")


def NMPCFollower(start_pose,
                 goal_pose,
                 leader_trajectory,
                 formation_d,
                 neighbour_trajectory,
                 obstacles,
                 neighbour_safe_dis,
                 avoid_obs_safe_dis,
                 z_limits,
                 use_debug=False):
    robot_state = start_pose
    robot_state_history = np.empty((3, 0))

    ref_trajectory = leader_trajectory + formation_d.reshape(-1, 1)
    robot_state_history = np.hstack(
        (robot_state_history, start_pose.reshape(-1, 1)))

    final_step = 0
    vel_list = []

    for i in range(NUMBER_OF_TIMESTEPS):
        dis_to_goal = np.linalg.norm(goal_pose - robot_state)
        lower_bound = lower_bound_default
        upper_bound = upper_bound_default
        if dis_to_goal >= 1.0 and dis_to_goal < 10.0:
            upper_bound = [0.5] * HORIZON_LENGTH * 3
            lower_bound = [-0.5] * HORIZON_LENGTH * 3
        if dis_to_goal < 1.0:
            upper_bound = [0.1] * HORIZON_LENGTH * 3
            lower_bound = [-0.1] * HORIZON_LENGTH * 3
        final_step = i
        ref_path = ComputeRefPath(robot_state, goal_pose, ref_trajectory, i,
                                  HORIZON_LENGTH, NMPC_TIMESTEP)

        neighbour_traj = GetNeighbourTraj(neighbour_trajectory, i,
                                          HORIZON_LENGTH, NMPC_TIMESTEP)

        vel, velocity_profile = ComputeVelocity(robot_state, neighbour_traj,
                                                obstacles, neighbour_safe_dis,
                                                avoid_obs_safe_dis, ref_path,
                                                lower_bound, upper_bound,
                                                z_limits)
        vel_list.append(vel.tolist() + [np.linalg.norm(vel)])
        if use_debug:
            print(f"vel:{vel}")

        robot_state = UpdateState(robot_state, vel, TIMESTEP)

        robot_state_history = np.hstack(
            (robot_state_history, robot_state.reshape(-1, 1)))
        if dis_to_goal < 0.05:
            print("final_step:", final_step, "final distance to goal:",
                  dis_to_goal)
            break

    robot_state_history = np.hstack(
        (robot_state_history, goal_pose.reshape(-1, 1)))

    return robot_state_history, final_step, vel_list, dis_to_goal
