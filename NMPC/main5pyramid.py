import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from utils import GeneratePyramid
from utils import GenerateRandomFloats
import time
import pandas as pd

# 假設MPC控制器的函數已經定義（mpc1_leader, mpc2_follower, mpc3_distance）
from NMPC1 import NMPCLeader
from NMPC2 import NMPCFollower

import datetime
import os
import csv

now = datetime.datetime.now()

# 格式化时间字符串为“年月日时分秒”的格式
time_str = now.strftime("%Y%m%d%H%M%S")

# 创建一个以当前时间命名的文件夹
folder_path = rf".\results\main7\{time_str}"  # 假设文件夹创建在当前目录下
os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 表示如果文件夹已存在，则不会抛出异常
traj_csv_path = os.path.join(folder_path, "path.csv")
l_f_loc_path = os.path.join(folder_path, "起点终点坐标.csv")
obs_loc_path = os.path.join(folder_path, "障碍物坐标.csv")
step_time_path = os.path.join(folder_path, "运行时间和迭代步数.csv")
vel_path = os.path.join(folder_path, "速度.csv")
dis_to_goal_path = os.path.join(folder_path, "终点误差.csv")
time_step_data = []
vel_list = []
dis_to_goal_list = []

# safe parameter
NEIGHBOUR_SAFE = 0.1
OBS_SAFE = 0.2

# 设置初始位置y的随机值
y_random = GenerateRandomFloats(6, -0.2, 0.2)

# 設定Leader初始位置和目標位置
P_l_start = np.array([0, y_random[4], 1.0])
P_l_goal = np.array([2.0, 0.0, 1.0])
z_limits = np.array([0.4, 2.0])

# 生成金字塔的向量
size = 0.3
vertices = GeneratePyramid(P_l_start, size)

# 设置leader初始位置
l_random = random.uniform

# 隨機設置Follower的初始位置
P_f1_start = vertices[1] + np.array([0, y_random[0], 0])
P_f2_start = vertices[2] + np.array([0, y_random[1], 0])
P_f3_start = vertices[3] + np.array([0, y_random[2], 0])
P_f4_start = vertices[4] + np.array([0, y_random[3], 0])

# 定義向量以形成編隊
d1 = vertices[1] - vertices[0]
d2 = vertices[2] - vertices[0]
d3 = vertices[3] - vertices[0]
d4 = vertices[4] - vertices[0]

obs1_x = 0.5
obs1_y = 0.0
obs1_z = 1.0
obs2_x = 1.1
obs2_y = -0.3
obs2_z = 1.0
# 设置球形障碍物的中心和半径, 下面这是两个障碍物的参数，前三位是x,y,z,第四位是r
obstacles = [[obs1_x, obs1_y, obs1_z, OBS_SAFE],
             [obs2_x, obs2_y, obs2_z, OBS_SAFE]]
obs = [[obs1_x, obs1_y, obs1_z], [obs2_x, obs2_y, obs2_z]]

# leader需要避开的障碍物体
obstacles_new = np.array(obs)

# 計算Leader的軌跡
print(f"P_l_start:{P_l_start} P_l_goal:{P_l_goal}")
time_start = time.time()
P_l_traj, step, vel, dis_to_goal = NMPCLeader(P_l_start, P_l_goal,
                                              obstacles_new, OBS_SAFE,
                                              z_limits)
time_step_data.append(["{:3f}".format(time.time() - time_start), step])
vel_list = vel
dis_to_goal_list.append(dis_to_goal)

# 計算Follower1的轨迹
print(f"P_f1_start:{P_f1_start} P_f1_goal:{P_l_goal + d1}")
time_start = time.time()
P_f1_traj, step, vel, dis_to_goal = NMPCFollower(P_f1_start, P_l_goal + d1,
                                                 P_l_traj, d1, P_l_traj,
                                                 obstacles_new, NEIGHBOUR_SAFE,
                                                 OBS_SAFE, z_limits)
time_step_data.append(["{:3f}".format(time.time() - time_start), step])
vel_list = [vel1 + vel2 for vel1, vel2 in zip(vel_list, vel)]
dis_to_goal_list.append(dis_to_goal)

# 計算Follower2的轨迹
print(f"P_f2_start:{P_f2_start} P_f2_goal:{P_l_goal + d2}")
time_start = time.time()
P_f2_traj, step, vel, dis_to_goal = NMPCFollower(P_f2_start, P_l_goal + d2,
                                                 P_l_traj, d2, P_f1_traj,
                                                 obstacles_new, NEIGHBOUR_SAFE,
                                                 OBS_SAFE, z_limits)
time_step_data.append(["{:3f}".format(time.time() - time_start), step])
vel_list = [vel1 + vel2 for vel1, vel2 in zip(vel_list, vel)]
dis_to_goal_list.append(dis_to_goal)

# 計算Follower3的轨迹
print(f"P_f3_start:{P_f3_start} P_f3_goal:{P_l_goal + d3}")
time_start = time.time()
P_f3_traj, step, vel, dis_to_goal = NMPCFollower(P_f3_start, P_l_goal + d3,
                                                 P_l_traj, d3, P_f2_traj,
                                                 obstacles_new, NEIGHBOUR_SAFE,
                                                 OBS_SAFE, z_limits)
time_step_data.append(["{:3f}".format(time.time() - time_start), step])
vel_list = [vel1 + vel2 for vel1, vel2 in zip(vel_list, vel)]
dis_to_goal_list.append(dis_to_goal)

# 計算Follower4的轨迹
print(f"P_f4_start:{P_f4_start} P_f4_goal:{P_l_goal + d4}")
time_start = time.time()
P_f4_traj, step, vel, dis_to_goal = NMPCFollower(P_f4_start, P_l_goal + d4,
                                                 P_l_traj, d4, P_f3_traj,
                                                 obstacles_new, NEIGHBOUR_SAFE,
                                                 OBS_SAFE, z_limits)
time_step_data.append(["{:3f}".format(time.time() - time_start), step])
vel_list = [vel1 + vel2 for vel1, vel2 in zip(vel_list, vel)]
dis_to_goal_list.append(dis_to_goal)

# 計算Follower的終點，定義為全局變數
p_l_end = P_l_traj[:, -1]
P_f1_end = P_f1_traj[:, -1]
P_f2_end = P_f2_traj[:, -1]
P_f3_end = P_f3_traj[:, -1]
P_f4_end = P_f4_traj[:, -1]

time.sleep(3)


# 插值以增加軌跡的點數，使動畫更加流暢
def interpolate_trajectory(trajectory, num_points):
    t_original = np.linspace(0, 1, trajectory.shape[1])
    t_new = np.linspace(0, 1, num_points)
    interp_func = interp1d(t_original, trajectory, axis=1, kind='cubic')
    return interp_func(t_new)


# 設置更高的插值幀數，這裡假設每條軌跡增加至總幀數的4倍
num_frames = len(P_l_traj[0])
num_frames_interpolated = num_frames * 4
P_l_traj_interpolated = interpolate_trajectory(P_l_traj,
                                               num_frames_interpolated)
P_f1_traj_interpolated = interpolate_trajectory(P_f1_traj,
                                                num_frames_interpolated)
P_f2_traj_interpolated = interpolate_trajectory(P_f2_traj,
                                                num_frames_interpolated)
P_f3_traj_interpolated = interpolate_trajectory(P_f3_traj,
                                                num_frames_interpolated)
P_f4_traj_interpolated = interpolate_trajectory(P_f4_traj,
                                                num_frames_interpolated)

# 记录csvs
leader_pose = {
    "leader_x": P_l_traj_interpolated[0, :num_frames_interpolated],
    "leader_y": P_l_traj_interpolated[1, :num_frames_interpolated],
    "leader_z": P_l_traj_interpolated[2, :num_frames_interpolated],
    "follow1_x": P_f1_traj_interpolated[0, :num_frames_interpolated],
    "follow1_y": P_f1_traj_interpolated[1, :num_frames_interpolated],
    "follow1_z": P_f1_traj_interpolated[2, :num_frames_interpolated],
    "follow2_x": P_f2_traj_interpolated[0, :num_frames_interpolated],
    "follow2_y": P_f2_traj_interpolated[1, :num_frames_interpolated],
    "follow2_z": P_f2_traj_interpolated[2, :num_frames_interpolated],
    "follow3_x": P_f3_traj_interpolated[0, :num_frames_interpolated],
    "follow3_y": P_f3_traj_interpolated[1, :num_frames_interpolated],
    "follow3_z": P_f3_traj_interpolated[2, :num_frames_interpolated],
    "follow4_x": P_f4_traj_interpolated[0, :num_frames_interpolated],
    "follow4_y": P_f4_traj_interpolated[1, :num_frames_interpolated],
    "follow4_z": P_f4_traj_interpolated[2, :num_frames_interpolated],
}

df = pd.DataFrame(leader_pose)
df.to_csv(traj_csv_path, index=False)

start_end_data = [[
    P_l_start[0], P_l_start[1], P_l_start[2],
    P_l_traj_interpolated[0, num_frames_interpolated - 1],
    P_l_traj_interpolated[1, num_frames_interpolated - 1],
    P_l_traj_interpolated[2, num_frames_interpolated - 1]
],
                  [
                      P_f1_start[0], P_f1_start[1], P_f1_start[2],
                      P_f1_traj_interpolated[0, num_frames_interpolated - 1],
                      P_f1_traj_interpolated[1, num_frames_interpolated - 1],
                      P_f1_traj_interpolated[2, num_frames_interpolated - 1]
                  ],
                  [
                      P_f2_start[0], P_f2_start[1], P_f2_start[2],
                      P_f2_traj_interpolated[0, num_frames_interpolated - 1],
                      P_f2_traj_interpolated[1, num_frames_interpolated - 1],
                      P_f2_traj_interpolated[2, num_frames_interpolated - 1]
                  ],
                  [
                      P_f3_start[0], P_f3_start[1], P_f3_start[2],
                      P_f3_traj_interpolated[0, num_frames_interpolated - 1],
                      P_f3_traj_interpolated[1, num_frames_interpolated - 1],
                      P_f3_traj_interpolated[2, num_frames_interpolated - 1]
                  ],
                  [
                      P_f4_start[0], P_f4_start[1], P_f4_start[2],
                      P_f4_traj_interpolated[0, num_frames_interpolated - 1],
                      P_f4_traj_interpolated[1, num_frames_interpolated - 1],
                      P_f4_traj_interpolated[2, num_frames_interpolated - 1]
                  ]]
start_end_data.insert(
    0, ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])
with open(l_f_loc_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in start_end_data:
        writer.writerow(row)  # 写入一行数据

obstacles_data = [[obs1_x, obs1_y, obs1_z, OBS_SAFE],
                  [obs2_x, obs2_y, obs2_z, OBS_SAFE]]
obstacles_data.insert(0, ["obs_x", "obs_y", "obs_z", "obs_r"])
with open(obs_loc_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in obstacles_data:
        writer.writerow(row)  # 写入一行数据

time_step_data.insert(0, ["time(s)", "final_step"])
with open(step_time_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in time_step_data:
        writer.writerow(row)  # 写入一行数据

vel_list.insert(0, [
    "leader_x",
    "leader_y",
    "leader_z",
    "leader_norm",
    "follow1_x",
    "follow1_y",
    "follow1_z",
    "follow1_norm",
    "follow2_x",
    "follow2_y",
    "follow2_z",
    "follow2_norm",
    "follow3_x",
    "follow3_y",
    "follow3_z",
    "follow3_norm",
    "follow4_x",
    "follow4_y",
    "follow4_z",
    "follow4_norm",
])
with open(vel_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in vel_list:
        writer.writerow(row)

dis_to_goal_list = [dis_to_goal_list]
dis_to_goal_list.insert(0,
                        ["leader", "follow1", "follow2", "follow3", "follow4"])
with open(dis_to_goal_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in dis_to_goal_list:
        writer.writerow(row)

# 创建球的参数
u = np.linspace(0, 2 * np.pi, 100)  # 从 0 到 2π
v = np.linspace(0, np.pi, 100)  # 从 0 到 π

# 使用参数方程计算球面上的点
x0_a = obstacles[0][0] + obstacles[0][-1] * np.outer(
    np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
y0_a = obstacles[0][1] + obstacles[0][-1] * np.outer(
    np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
z0_a = obstacles[0][2] + obstacles[0][-1] * np.outer(np.ones(
    np.size(u)), np.cos(v))  # z = z0 + r * cos(φ)

x1_a = obstacles[1][0] + obstacles[1][-1] * np.outer(
    np.cos(u), np.sin(v))  # x = x0 + r * cos(θ) * sin(φ)
y1_a = obstacles[1][1] + obstacles[1][-1] * np.outer(
    np.sin(u), np.sin(v))  # y = y0 + r * sin(θ) * sin(φ)
z1_a = obstacles[1][2] + obstacles[1][-1] * np.outer(np.ones(
    np.size(u)), np.cos(v))  # z = z0 + r * cos(φ)

# 設置動畫
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 2])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.0, 2.0])

# 绘制障碍物
ax.plot_surface(x0_a, y0_a, z0_a, color='b', alpha=0.6)  # 使用半透明的蓝色
ax.plot_surface(x1_a, y1_a, z1_a, color='b', alpha=0.6)  # 使用半透明的蓝色

# 初始化三條線條
leader_line, = ax.plot([], [], [], label="Leader Trajectory", color='red')
f1_line, = ax.plot([], [], [], label="Follower1 Trajectory", color='green')
f2_line, = ax.plot([], [], [], label="Follower2 Trajectory", color='orange')
f3_line, = ax.plot([], [], [], label="Follower3 Trajectory", color='cyan')
f4_line, = ax.plot([], [], [], label="Follower4 Trajectory", color='brown')

# 初始和目標點，以紅色標記並用黑色標注座標
ax.scatter(P_l_start[0],
           P_l_start[1],
           P_l_start[2],
           color='red',
           label='Start Point',
           s=100)

ax.scatter(P_l_goal[0],
           P_l_goal[1],
           P_l_goal[2],
           color='red',
           label='Goal Point',
           s=100)
ax.text(P_l_goal[0],
        P_l_goal[1],
        P_l_goal[2],
        f'({P_l_goal[0]:.2f}, {P_l_goal[1]:.2f}, {P_l_goal[2]:.2f})',
        color='black')

# Follower1 和 Follower2 的起始點
ax.scatter(P_f1_start[0],
           P_f1_start[1],
           P_f1_start[2],
           color='green',
           label='Follower1 Start',
           s=100)

ax.scatter(P_f2_start[0],
           P_f2_start[1],
           P_f2_start[2],
           color='orange',
           label='Follower2 Start',
           s=100)

ax.scatter(P_f3_start[0],
           P_f3_start[1],
           P_f3_start[2],
           color='cyan',
           label='Follower3 Start',
           s=100)

ax.scatter(P_f4_start[0],
           P_f4_start[1],
           P_f4_start[2],
           color='brown',
           label='Follower4 Start',
           s=100)


# 初始化函數
def init():
    leader_line.set_data([], [])
    leader_line.set_3d_properties([])

    f1_line.set_data([], [])
    f1_line.set_3d_properties([])

    f2_line.set_data([], [])
    f2_line.set_3d_properties([])

    f3_line.set_data([], [])
    f3_line.set_3d_properties([])

    f4_line.set_data([], [])
    f4_line.set_3d_properties([])

    return leader_line, f1_line, f2_line, f3_line, f4_line


# 更新每一幀的函數，以使用插值後的軌跡
def update_smooth(frame):
    leader_line.set_data(P_l_traj_interpolated[0, :frame],
                         P_l_traj_interpolated[1, :frame])
    leader_line.set_3d_properties(P_l_traj_interpolated[2, :frame])

    f1_line.set_data(P_f1_traj_interpolated[0, :frame],
                     P_f1_traj_interpolated[1, :frame])
    f1_line.set_3d_properties(P_f1_traj_interpolated[2, :frame])

    f2_line.set_data(P_f2_traj_interpolated[0, :frame],
                     P_f2_traj_interpolated[1, :frame])
    f2_line.set_3d_properties(P_f2_traj_interpolated[2, :frame])

    f3_line.set_data(P_f3_traj_interpolated[0, :frame],
                     P_f3_traj_interpolated[1, :frame])
    f3_line.set_3d_properties(P_f3_traj_interpolated[2, :frame])

    f4_line.set_data(P_f4_traj_interpolated[0, :frame],
                     P_f4_traj_interpolated[1, :frame])
    f4_line.set_3d_properties(P_f4_traj_interpolated[2, :frame])

    # 在每一帧更新leader和follower之间的虚线
    if frame < num_frames_interpolated - 1:
        # 连接leader和follower1的褐色虚线
        if not hasattr(update_smooth, 'line_f1'):
            update_smooth.line_f1, = ax.plot([], [], [],
                                             color='gray',
                                             linestyle='--')
        update_smooth.line_f1.set_data([
            P_l_traj_interpolated[0, frame], P_f1_traj_interpolated[0, frame]
        ], [P_l_traj_interpolated[1, frame], P_f1_traj_interpolated[1, frame]])
        update_smooth.line_f1.set_3d_properties([
            P_l_traj_interpolated[2, frame], P_f1_traj_interpolated[2, frame]
        ])

        # 连接leader和follower2的褐色虚线
        if not hasattr(update_smooth, 'line_f2'):
            update_smooth.line_f2, = ax.plot([], [], [],
                                             color='gray',
                                             linestyle='--')
        update_smooth.line_f2.set_data([
            P_l_traj_interpolated[0, frame], P_f2_traj_interpolated[0, frame]
        ], [P_l_traj_interpolated[1, frame], P_f2_traj_interpolated[1, frame]])
        update_smooth.line_f2.set_3d_properties([
            P_l_traj_interpolated[2, frame], P_f2_traj_interpolated[2, frame]
        ])

        # 连接leader和follower3的褐色虚线
        if not hasattr(update_smooth, 'line_f3'):
            update_smooth.line_f3, = ax.plot([], [], [],
                                             color='green',
                                             linestyle='--')
        update_smooth.line_f3.set_data([
            P_l_traj_interpolated[0, frame], P_f3_traj_interpolated[0, frame]
        ], [P_l_traj_interpolated[1, frame], P_f3_traj_interpolated[1, frame]])
        update_smooth.line_f3.set_3d_properties([
            P_l_traj_interpolated[2, frame], P_f3_traj_interpolated[2, frame]
        ])

        # 连接leader和follower4的褐色虚线
        if not hasattr(update_smooth, 'line_f4'):
            update_smooth.line_f4, = ax.plot([], [], [],
                                             color='green',
                                             linestyle='--')
        update_smooth.line_f4.set_data([
            P_l_traj_interpolated[0, frame], P_f4_traj_interpolated[0, frame]
        ], [P_l_traj_interpolated[1, frame], P_f4_traj_interpolated[1, frame]])
        update_smooth.line_f4.set_3d_properties([
            P_l_traj_interpolated[2, frame], P_f4_traj_interpolated[2, frame]
        ])

    # 在最後一幀展示Follower1和Follower2的終點
    if frame == num_frames_interpolated - 1:
        ax.scatter(P_f1_end[0],
                   P_f1_end[1],
                   P_f1_end[2],
                   color='green',
                   label='Follower1 End',
                   s=100)
        ax.text(P_f1_end[0],
                P_f1_end[1],
                P_f1_end[2],
                f'({P_f1_end[0]:.2f}, {P_f1_end[1]:.2f}, {P_f1_end[2]:.2f})',
                color='black')

        ax.scatter(P_f2_end[0],
                   P_f2_end[1],
                   P_f2_end[2],
                   color='orange',
                   label='Follower2 End',
                   s=100)
        ax.text(P_f2_end[0],
                P_f2_end[1],
                P_f2_end[2],
                f'({P_f2_end[0]:.2f}, {P_f2_end[1]:.2f}, {P_f2_end[2]:.2f})',
                color='black')

        ax.scatter(P_f3_end[0],
                   P_f3_end[1],
                   P_f3_end[2],
                   color='cyan',
                   label='Follower3 End',
                   s=100)
        ax.text(P_f3_end[0],
                P_f3_end[1],
                P_f3_end[2],
                f'({P_f3_end[0]:.2f}, {P_f3_end[1]:.2f}, {P_f3_end[2]:.2f})',
                color='black')

        ax.scatter(P_f4_end[0],
                   P_f4_end[1],
                   P_f4_end[2],
                   color='brown',
                   label='Follower4 End',
                   s=100)
        ax.text(P_f4_end[0],
                P_f4_end[1],
                P_f4_end[2],
                f'({P_f4_end[0]:.2f}, {P_f4_end[1]:.2f}, {P_f4_end[2]:.2f})',
                color='black')

        # plot l->f1->f2
        ax.plot([p_l_end[0], P_f1_end[0]], [p_l_end[1], P_f1_end[1]],
                [p_l_end[2], P_f1_end[2]],
                color='black')
        ax.plot([p_l_end[0], P_f2_end[0]], [p_l_end[1], P_f2_end[1]],
                [p_l_end[2], P_f2_end[2]],
                color='black')
        # plot l->f3->f4
        ax.plot([p_l_end[0], P_f3_end[0]], [p_l_end[1], P_f3_end[1]],
                [p_l_end[2], P_f3_end[2]],
                color='black')
        ax.plot([p_l_end[0], P_f4_end[0]], [p_l_end[1], P_f4_end[1]],
                [p_l_end[2], P_f4_end[2]],
                color='black')

        # plot f1->f2->f3->f4->f1
        ax.plot([P_f1_end[0], P_f2_end[0]], [P_f1_end[1], P_f2_end[1]],
                [P_f1_end[2], P_f2_end[2]],
                color='black')
        ax.plot([P_f2_end[0], P_f3_end[0]], [P_f2_end[1], P_f3_end[1]],
                [P_f2_end[2], P_f3_end[2]],
                color='black')
        ax.plot([P_f3_end[0], P_f4_end[0]], [P_f3_end[1], P_f4_end[1]],
                [P_f3_end[2], P_f4_end[2]],
                color='black')
        ax.plot([P_f4_end[0], P_f1_end[0]], [P_f4_end[1], P_f1_end[1]],
                [P_f4_end[2], P_f1_end[2]],
                color='black')

        # 删除虚线
        update_smooth.line_f1.set_data([], [])
        update_smooth.line_f1.set_3d_properties([])
        update_smooth.line_f2.set_data([], [])
        update_smooth.line_f2.set_3d_properties([])
        update_smooth.line_f3.set_data([], [])
        update_smooth.line_f3.set_3d_properties([])
        update_smooth.line_f4.set_data([], [])
        update_smooth.line_f4.set_3d_properties([])

        ani.event_source.stop()

    return leader_line, f1_line, f2_line


# 動畫設置，將動畫的更新與MPC的狀態更新同步
interval = 50  # 假設每次MPC更新狀態的時間間隔是100毫秒
ani = FuncAnimation(fig,
                    update_smooth,
                    frames=num_frames_interpolated,
                    init_func=init,
                    blit=False,
                    interval=interval)

ax.legend()

plt.show()
