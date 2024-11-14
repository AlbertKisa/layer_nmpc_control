import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
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
time_step_data = []

# safe parameter
NEIGHBOUR_SAFE = 0.5
OBS_SAFE = 1.0

# 設定Leader初始位置和目標位置
P_l_start = np.array([0, 0, 0])
P_l_goal = np.array([15, 15, 15])

# 隨機設置Follower的初始位置
f1_random = random.uniform(-2, -1)
f2_random = random.uniform(1, 2)
f1_f1_random = random.uniform(-4, -3)
f1_f2_random = random.uniform(-3, -2)
f2_f1_random = random.uniform(2, 3)
f2_f2_random = random.uniform(3, 4)
P_f1_start = np.array([f1_random, 0.0, 0])
P_f2_start = np.array([f2_random, 0.0, 0])
P_f1_f1_start = np.array([f1_f1_random, 0.0, 0])
P_f1_f2_start = np.array([f1_f2_random, 0.0, 0])
P_f2_f1_start = np.array([f2_f1_random, 0.0, 0])
P_f2_f2_start = np.array([f2_f2_random, 0.0, 0])

# 定義向量以形成編隊
d1 = np.array([-1.0, -1.0, -0.5])
d2 = np.array([1.0, -1.0, -0.5])

f_d1 = np.array([-0.5, -0.5, -0.25])
f_d2 = np.array([0.5, -0.5, -0.25])

obs1_x = random.uniform(4, 7)
obs1_y = obs1_x
obs1_z = obs1_x
obs2_x = random.uniform(9, 12)
obs2_y = obs2_x
obs2_z = obs2_x
# 设置球形障碍物的中心和半径, 下面这是两个障碍物的参数，前三位是x,y,z,第四位是r
obstacles = [[obs1_x, obs1_y, obs1_z, OBS_SAFE],
             [obs2_x, obs2_y, obs2_z, OBS_SAFE]]
obs = [[obs1_x, obs1_y, obs1_z], [obs2_x, obs2_y, obs2_z]]

# leader需要避开的障碍物体
obstacles_new = np.array(obs)

# 計算Leader的軌跡
print(f"P_l_start:{P_l_start} P_l_goal:{P_l_goal}")
time_start = time.time()
P_l_traj, step = NMPCLeader(P_l_start, P_l_goal, obstacles_new)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

# 延遲兩秒後計算mpc2，讓Follower跟隨Leader
time.sleep(0.5)

# 計算Follower1和Follower2的軌跡
print(f"P_f1_start:{P_f1_start} P_f1_goal:{P_l_goal + d1}")
time_start = time.time()
P_f1_traj, step = NMPCFollower(P_f1_start, P_l_goal + d1, P_l_traj, d1,
                         np.empty((3, 0)), obstacles_new, NEIGHBOUR_SAFE,
                         OBS_SAFE, True)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

print(f"P_f2_start:{P_f2_start} P_f2_goal:{P_l_goal + d2}")
time_start = time.time()
P_f2_traj, step = NMPCFollower(P_f2_start, P_l_goal + d2, P_l_traj, d2, P_f1_traj,
                         obstacles_new, NEIGHBOUR_SAFE, OBS_SAFE, True)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

# 计算F1_f1和F1_f2的轨迹
print(f"P_f1_f1_start:{P_f1_f1_start} P_f1_f1_goal:{P_l_goal + d1 + f_d1}")
time_start = time.time()
P_f1_f1_traj, step = NMPCFollower(P_f1_f1_start,
                            P_l_goal + d1 + f_d1, P_f1_traj, f_d1,
                            np.empty((3, 0)), obstacles_new, NEIGHBOUR_SAFE,
                            OBS_SAFE)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

print(f"P_f1_f2_start:{P_f1_f2_start} P_f1_f2_goal:{P_l_goal + d1 + f_d2}")
time_start = time.time()
P_f1_f2_traj, step = NMPCFollower(P_f1_f2_start, P_l_goal + d1 + f_d2, P_f1_traj,
                            f_d2, P_f1_f1_traj, obstacles_new, NEIGHBOUR_SAFE,
                            OBS_SAFE)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

# 计算F1_f1和F1_f2的轨迹
print(f"P_f2_f1_start:{P_f2_f1_start} P_f2_f1_goal:{P_l_goal + d2 + f_d1}")
time_start = time.time()
P_f2_f1_traj, step = NMPCFollower(P_f2_f1_start,
                            P_l_goal + d2 + f_d1, P_f2_traj, f_d1,
                            np.empty((3, 0)), obstacles_new, NEIGHBOUR_SAFE,
                            OBS_SAFE)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

print(f"P_f2_f2_start:{P_f2_f2_start} P_f2_f2_goal:{P_l_goal + d2 + f_d2}")
time_start = time.time()
P_f2_f2_traj, step = NMPCFollower(P_f2_f2_start, P_l_goal + d2 + f_d2, P_f2_traj,
                            f_d2, P_f2_f1_traj, obstacles_new, NEIGHBOUR_SAFE,
                            OBS_SAFE)
time_step_data.append(["{:3f}".format(time.time()-time_start), step])

# 計算Follower的終點，定義為全局變數
P_f1_end = P_f1_traj[:, -1]  # Follower1 的終點
P_f2_end = P_f2_traj[:, -1]  # Follower2 的終點
P_f1_f1_end = P_f1_f1_traj[:, -1]
P_f1_f2_end = P_f1_f2_traj[:, -1]
P_f2_f1_end = P_f2_f1_traj[:, -1]
P_f2_f2_end = P_f2_f2_traj[:, -1]

time.sleep(3)


# 插值以增加軌跡的點數，使動畫更加流暢
def interpolate_trajectory(trajectory, num_points):
    t_original = np.linspace(0, 1, trajectory.shape[1])
    t_new = np.linspace(0, 1, num_points)
    interp_func = interp1d(t_original, trajectory, axis=1, kind='linear')
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
P_f1_f1_traj_interpolated = interpolate_trajectory(P_f1_f1_traj,
                                                   num_frames_interpolated)
P_f1_f2_traj_interpolated = interpolate_trajectory(P_f1_f2_traj,
                                                   num_frames_interpolated)
P_f2_f1_traj_interpolated = interpolate_trajectory(P_f2_f1_traj,
                                                   num_frames_interpolated)
P_f2_f2_traj_interpolated = interpolate_trajectory(P_f2_f2_traj,
                                                   num_frames_interpolated)

#记录csv
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
    "follow1_1_x": P_f1_f1_traj_interpolated[0, :num_frames_interpolated],
    "follow1_1_y": P_f1_f1_traj_interpolated[1, :num_frames_interpolated],
    "follow1_1_z": P_f1_f1_traj_interpolated[2, :num_frames_interpolated],
    "follow1_2_x": P_f1_f2_traj_interpolated[0, :num_frames_interpolated],
    "follow1_2_y": P_f1_f2_traj_interpolated[1, :num_frames_interpolated],
    "follow1_2_z": P_f1_f2_traj_interpolated[2, :num_frames_interpolated],
    "follow2_1_x": P_f2_f1_traj_interpolated[0, :num_frames_interpolated],
    "follow2_1_y": P_f2_f1_traj_interpolated[1, :num_frames_interpolated],
    "follow2_1_z": P_f2_f1_traj_interpolated[2, :num_frames_interpolated],
    "follow2_2_x": P_f2_f2_traj_interpolated[0, :num_frames_interpolated],
    "follow2_2_y": P_f2_f2_traj_interpolated[1, :num_frames_interpolated],
    "follow2_2_z": P_f2_f2_traj_interpolated[2, :num_frames_interpolated],
}

df = pd.DataFrame(leader_pose)
df.to_csv(traj_csv_path, index=False)

start_end_data = [
    [
        P_l_start[0],P_l_start[1],P_l_start[2],
        P_l_traj_interpolated[0, num_frames_interpolated-1],
        P_l_traj_interpolated[1, num_frames_interpolated-1],
        P_l_traj_interpolated[2, num_frames_interpolated-1]
    ],
    [
        P_f1_start[0], P_f1_start[1], P_f1_start[2], 
        P_f1_traj_interpolated[0, num_frames_interpolated-1], 
        P_f1_traj_interpolated[1, num_frames_interpolated-1], 
        P_f1_traj_interpolated[2, num_frames_interpolated-1]
    ],
    [
        P_f2_start[0], P_f2_start[1], P_f2_start[2], 
        P_f2_traj_interpolated[0, num_frames_interpolated-1], 
        P_f2_traj_interpolated[1, num_frames_interpolated-1], 
        P_f2_traj_interpolated[2, num_frames_interpolated-1]
    ],
        [
        P_f1_f1_start[0], P_f1_f1_start[1], P_f1_f1_start[2], 
        P_f1_f1_traj_interpolated[0, num_frames_interpolated-1], 
        P_f1_f1_traj_interpolated[1, num_frames_interpolated-1], 
        P_f1_f1_traj_interpolated[2, num_frames_interpolated-1]
    ],
    [
        P_f1_f2_start[0], P_f1_f2_start[1], P_f1_f2_start[2], 
        P_f1_f2_traj_interpolated[0, num_frames_interpolated-1], 
        P_f1_f2_traj_interpolated[1, num_frames_interpolated-1], 
        P_f1_f2_traj_interpolated[2, num_frames_interpolated-1]
    ],
    [
        P_f2_f1_start[0], P_f2_f1_start[1], P_f2_f1_start[2], 
        P_f2_f1_traj_interpolated[0, num_frames_interpolated-1], 
        P_f2_f1_traj_interpolated[1, num_frames_interpolated-1], 
        P_f2_f1_traj_interpolated[2, num_frames_interpolated-1]
    ],
    [
        P_f2_f2_start[0], P_f2_f2_start[1], P_f2_f2_start[2], 
        P_f2_f2_traj_interpolated[0, num_frames_interpolated-1], 
        P_f2_f2_traj_interpolated[1, num_frames_interpolated-1], 
        P_f2_f2_traj_interpolated[2, num_frames_interpolated-1]
    ]
]
start_end_data.insert(0, ["start_x","start_y","start_z","end_x","end_y","end_z"])
with open(l_f_loc_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in start_end_data:
        writer.writerow(row)  # 写入一行数据

obstacles_data = [[obs1_x, obs1_y, obs1_z, OBS_SAFE],
             [obs2_x, obs2_y, obs2_z, OBS_SAFE]]
obstacles_data.insert(0, ["obs_x","obs_y","obs_z","obs_r"])
with open(obs_loc_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in obstacles_data:
        writer.writerow(row)  # 写入一行数据

time_step_data.insert(0, ["time(s)", "final_step"])
with open(step_time_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in time_step_data:
        writer.writerow(row)  # 写入一行数据

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
ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_zlim([-5, 15])

# 绘制障碍物
ax.plot_surface(x0_a, y0_a, z0_a, color='b', alpha=0.6)  # 使用半透明的蓝色
ax.plot_surface(x1_a, y1_a, z1_a, color='b', alpha=0.6)  # 使用半透明的蓝色

# 初始化三條線條
leader_line, = ax.plot([], [], [], label="Leader Trajectory", color='red')
f1_line, = ax.plot([], [], [], label="Follower1 Trajectory", color='green')
f2_line, = ax.plot([], [], [], label="Follower2 Trajectory", color='orange')
f1_f1_line, = ax.plot([], [], [], label="F1_F1 Trajectory", color='cyan')
f1_f2_line, = ax.plot([], [], [], label="F1_F2 Trajectory", color='brown')
f2_f1_line, = ax.plot([], [], [], label="F2_F1 Trajectory", color='pink')
f2_f2_line, = ax.plot([], [], [], label="F2_F2 Trajectory", color='purple')

# 初始和目標點，以紅色標記並用黑色標注座標
ax.scatter(P_l_start[0],
           P_l_start[1],
           P_l_start[2],
           color='red',
           label='Start Point',
           s=100)
ax.text(P_l_start[0],
        P_l_start[1],
        P_l_start[2],
        f'({P_l_start[0]:.2f}, {P_l_start[1]:.2f}, {P_l_start[2]:.2f})',
        color='black')

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
ax.text(P_f1_start[0],
        P_f1_start[1],
        P_f1_start[2],
        f'({P_f1_start[0]:.2f}, {P_f1_start[1]:.2f}, {P_f1_start[2]:.2f})',
        color='black')

ax.scatter(P_f2_start[0],
           P_f2_start[1],
           P_f2_start[2],
           color='orange',
           label='Follower2 Start',
           s=100)
ax.text(P_f2_start[0],
        P_f2_start[1],
        P_f2_start[2],
        f'({P_f2_start[0]:.2f}, {P_f2_start[1]:.2f}, {P_f2_start[2]:.2f})',
        color='black')

ax.scatter(P_f1_f1_start[0],
           P_f1_f1_start[1],
           P_f1_f1_start[2],
           color='cyan',
           label='Follower1_f1 Start',
           s=100)
ax.text(
    P_f1_f1_start[0],
    P_f1_f1_start[1],
    P_f1_f1_start[2],
    f'({P_f1_f1_start[0]:.2f}, {P_f1_f1_start[1]:.2f}, {P_f1_f1_start[2]:.2f})',
    color='black')

ax.scatter(P_f1_f2_start[0],
           P_f1_f2_start[1],
           P_f1_f2_start[2],
           color='brown',
           label='Follower1_f2 Start',
           s=100)
ax.text(
    P_f1_f2_start[0],
    P_f1_f2_start[1],
    P_f1_f2_start[2],
    f'({P_f1_f2_start[0]:.2f}, {P_f1_f2_start[1]:.2f}, {P_f1_f2_start[2]:.2f})',
    color='black')

ax.scatter(P_f2_f1_start[0],
           P_f2_f1_start[1],
           P_f2_f1_start[2],
           color='pink',
           label='Follower2_f1 Start',
           s=100)
ax.text(
    P_f2_f1_start[0],
    P_f2_f1_start[1],
    P_f2_f1_start[2],
    f'({P_f2_f1_start[0]:.2f}, {P_f2_f1_start[1]:.2f}, {P_f2_f1_start[2]:.2f})',
    color='black')

ax.scatter(P_f2_f2_start[0],
           P_f2_f2_start[1],
           P_f2_f2_start[2],
           color='purple',
           label='Follower2_f1 Start',
           s=100)
ax.text(
    P_f2_f2_start[0],
    P_f2_f2_start[1],
    P_f2_f2_start[2],
    f'({P_f2_f2_start[0]:.2f}, {P_f2_f2_start[1]:.2f}, {P_f2_f2_start[2]:.2f})',
    color='black')


# 初始化函數
def init():
    leader_line.set_data([], [])
    leader_line.set_3d_properties([])

    f1_line.set_data([], [])
    f1_line.set_3d_properties([])

    f2_line.set_data([], [])
    f2_line.set_3d_properties([])

    f1_f1_line.set_data([], [])
    f1_f1_line.set_3d_properties([])

    f1_f2_line.set_data([], [])
    f1_f2_line.set_3d_properties([])

    f2_f1_line.set_data([], [])
    f2_f1_line.set_3d_properties([])

    f2_f2_line.set_data([], [])
    f2_f2_line.set_3d_properties([])

    return leader_line, f1_line, f2_line, f1_f1_line, f1_f2_line, f2_f1_line, f2_f2_line


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

    f1_f1_line.set_data(P_f1_f1_traj_interpolated[0, :frame],
                        P_f1_f1_traj_interpolated[1, :frame])
    f1_f1_line.set_3d_properties(P_f1_f1_traj_interpolated[2, :frame])

    f1_f2_line.set_data(P_f1_f2_traj_interpolated[0, :frame],
                        P_f1_f2_traj_interpolated[1, :frame])
    f1_f2_line.set_3d_properties(P_f1_f2_traj_interpolated[2, :frame])

    f2_f1_line.set_data(P_f2_f1_traj_interpolated[0, :frame],
                        P_f2_f1_traj_interpolated[1, :frame])
    f2_f1_line.set_3d_properties(P_f2_f1_traj_interpolated[2, :frame])

    f2_f2_line.set_data(P_f2_f2_traj_interpolated[0, :frame],
                        P_f2_f2_traj_interpolated[1, :frame])
    f2_f2_line.set_3d_properties(P_f2_f2_traj_interpolated[2, :frame])

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

        # 连接follower1和follower1_1的褐色虚线
        if not hasattr(update_smooth, 'line_f1_f1'):
            update_smooth.line_f1_f1, = ax.plot([], [], [],
                                                color='green',
                                                linestyle='--')
        update_smooth.line_f1_f1.set_data([
            P_f1_traj_interpolated[0, frame], P_f1_f1_traj_interpolated[0,
                                                                        frame]
        ], [
            P_f1_traj_interpolated[1, frame], P_f1_f1_traj_interpolated[1,
                                                                        frame]
        ])
        update_smooth.line_f1_f1.set_3d_properties([
            P_f1_traj_interpolated[2, frame], P_f1_f1_traj_interpolated[2,
                                                                        frame]
        ])

        # 连接follower1和follower1_2的褐色虚线
        if not hasattr(update_smooth, 'line_f1_f2'):
            update_smooth.line_f1_f2, = ax.plot([], [], [],
                                                color='green',
                                                linestyle='--')
        update_smooth.line_f1_f2.set_data([
            P_f1_traj_interpolated[0, frame], P_f1_f2_traj_interpolated[0,
                                                                        frame]
        ], [
            P_f1_traj_interpolated[1, frame], P_f1_f2_traj_interpolated[1,
                                                                        frame]
        ])
        update_smooth.line_f1_f2.set_3d_properties([
            P_f1_traj_interpolated[2, frame], P_f1_f2_traj_interpolated[2,
                                                                        frame]
        ])

        # 连接follower2和follower2_1的褐色虚线
        if not hasattr(update_smooth, 'line_f2_f1'):
            update_smooth.line_f2_f1, = ax.plot([], [], [],
                                                color='orange',
                                                linestyle='--')
        update_smooth.line_f2_f1.set_data([
            P_f2_traj_interpolated[0, frame], P_f2_f1_traj_interpolated[0,
                                                                        frame]
        ], [
            P_f2_traj_interpolated[1, frame], P_f2_f1_traj_interpolated[1,
                                                                        frame]
        ])
        update_smooth.line_f2_f1.set_3d_properties([
            P_f2_traj_interpolated[2, frame], P_f2_f1_traj_interpolated[2,
                                                                        frame]
        ])

        # 连接follower2和follower2_2的褐色虚线
        if not hasattr(update_smooth, 'line_f2_f2'):
            update_smooth.line_f2_f2, = ax.plot([], [], [],
                                                color='orange',
                                                linestyle='--')
        update_smooth.line_f2_f2.set_data([
            P_f2_traj_interpolated[0, frame], P_f2_f2_traj_interpolated[0,
                                                                        frame]
        ], [
            P_f2_traj_interpolated[1, frame], P_f2_f2_traj_interpolated[1,
                                                                        frame]
        ])
        update_smooth.line_f2_f2.set_3d_properties([
            P_f2_traj_interpolated[2, frame], P_f2_f2_traj_interpolated[2,
                                                                        frame]
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

        ax.scatter(P_f1_f1_end[0],
                   P_f1_f1_end[1],
                   P_f1_f1_end[2],
                   color='cyan',
                   label='Follower1_1 End',
                   s=100)
        ax.text(
            P_f1_f1_end[0],
            P_f1_f1_end[1],
            P_f1_f1_end[2],
            f'({P_f1_f1_end[0]:.2f}, {P_f1_f1_end[1]:.2f}, {P_f1_f1_end[2]:.2f})',
            color='black')

        ax.scatter(P_f1_f2_end[0],
                   P_f1_f2_end[1],
                   P_f1_f2_end[2],
                   color='brown',
                   label='Follower1_2 End',
                   s=100)
        ax.text(
            P_f1_f2_end[0],
            P_f1_f2_end[1],
            P_f1_f2_end[2],
            f'({P_f1_f2_end[0]:.2f}, {P_f1_f2_end[1]:.2f}, {P_f1_f2_end[2]:.2f})',
            color='black')

        ax.scatter(P_f2_f1_end[0],
                   P_f2_f1_end[1],
                   P_f2_f1_end[2],
                   color='pink',
                   label='Follower2_1 End',
                   s=100)
        ax.text(
            P_f2_f1_end[0],
            P_f2_f1_end[1],
            P_f2_f1_end[2],
            f'({P_f2_f1_end[0]:.2f}, {P_f2_f1_end[1]:.2f}, {P_f2_f1_end[2]:.2f})',
            color='black')

        ax.scatter(P_f2_f2_end[0],
                   P_f2_f2_end[1],
                   P_f2_f2_end[2],
                   color='purple',
                   label='Follower2_2 End',
                   s=100)
        ax.text(
            P_f2_f2_end[0],
            P_f2_f2_end[1],
            P_f2_f2_end[2],
            f'({P_f2_f2_end[0]:.2f}, {P_f2_f2_end[1]:.2f}, {P_f2_f2_end[2]:.2f})',
            color='black')
        # plot l->f1->f2
        ax.plot([P_l_traj[0, -1], P_f1_end[0]], [P_l_traj[1, -1], P_f1_end[1]],
                [P_l_traj[2, -1], P_f1_end[2]],
                color='black')
        ax.plot([P_l_traj[0, -1], P_f2_end[0]], [P_l_traj[1, -1], P_f2_end[1]],
                [P_l_traj[2, -1], P_f2_end[2]],
                color='black')
        ax.plot([P_f1_end[0], P_f2_end[0]], [P_f1_end[1], P_f2_end[1]],
                [P_f1_end[2], P_f2_end[2]],
                color='black')
        # plot f1->f1_1->f1_2
        ax.plot([P_f1_end[0], P_f1_f1_end[0]], [P_f1_end[1], P_f1_f1_end[1]],
                [P_f1_end[2], P_f1_f1_end[2]],
                color='black')
        ax.plot([P_f1_end[0], P_f1_f2_end[0]], [P_f1_end[1], P_f1_f2_end[1]],
                [P_f1_end[2], P_f1_f2_end[2]],
                color='black')
        ax.plot([P_f1_f1_end[0], P_f1_f2_end[0]],
                [P_f1_f1_end[1], P_f1_f2_end[1]],
                [P_f1_f1_end[2], P_f1_f2_end[2]],
                color='black')

        # plot f2->f2_1->f2_2
        ax.plot([P_f2_end[0], P_f2_f1_end[0]], [P_f2_end[1], P_f2_f1_end[1]],
                [P_f2_end[2], P_f2_f1_end[2]],
                color='black')
        ax.plot([P_f2_end[0], P_f2_f2_end[0]], [P_f2_end[1], P_f2_f2_end[1]],
                [P_f2_end[2], P_f2_f2_end[2]],
                color='black')
        ax.plot([P_f2_f1_end[0], P_f2_f2_end[0]],
                [P_f2_f1_end[1], P_f2_f2_end[1]],
                [P_f2_f1_end[2], P_f2_f2_end[2]],
                color='black')

        # 删除虚线
        update_smooth.line_f1.set_data([], [])
        update_smooth.line_f1.set_3d_properties([])
        update_smooth.line_f2.set_data([], [])
        update_smooth.line_f2.set_3d_properties([])
        update_smooth.line_f1_f1.set_data([], [])
        update_smooth.line_f1_f1.set_3d_properties([])
        update_smooth.line_f1_f2.set_data([], [])
        update_smooth.line_f1_f2.set_3d_properties([])
        update_smooth.line_f2_f1.set_data([], [])
        update_smooth.line_f2_f1.set_3d_properties([])
        update_smooth.line_f2_f2.set_data([], [])
        update_smooth.line_f2_f2.set_3d_properties([])

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
