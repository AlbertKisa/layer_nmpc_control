import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import time

# 假設MPC控制器的函數已經定義（mpc1_leader, mpc2_follower, mpc3_distance）
from MPC1 import mpc1_leader
from MPC2 import mpc2_follower
from MPC3 import mpc3_distance

# 設定Leader初始位置和目標位置
P_l_start = np.array([0, 0, 0])
P_l_goal = np.array([10, 10, 10])

# 隨機設置Follower的初始位置
a = random.uniform(-5, 5)
P_f1_start = np.array([a, a, 0])
P_f2_start = np.array([a, a, 0])

# 定義向量以形成編隊
d1 = np.array([3, 3, 0])
d2 = np.array([-3, 3, 0])
d3 = np.array([6, 0, 0])

# 計算Leader的軌跡
P_l_traj, _ = mpc1_leader(P_l_start, P_l_goal)

# 延遲兩秒後計算mpc2，讓Follower跟隨Leader
time.sleep(2)

# 計算Follower1和Follower2的軌跡
P_f1_traj, _ = mpc2_follower(P_f1_start, P_l_traj, d1)
P_f2_traj, _ = mpc2_follower(P_f2_start, P_l_traj, d2)

# 保持Follower1和Follower2的距離
P_f12_traj = mpc3_distance(P_f1_start, P_f2_start, d3)

# 計算Follower的終點，定義為全局變數
P_f1_end = P_f1_traj[:, -1]  # Follower1 的終點
P_f2_end = P_f2_traj[:, -1]  # Follower2 的終點

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
P_l_traj_interpolated = interpolate_trajectory(P_l_traj, num_frames_interpolated)
P_f1_traj_interpolated = interpolate_trajectory(P_f1_traj, num_frames_interpolated)
P_f2_traj_interpolated = interpolate_trajectory(P_f2_traj, num_frames_interpolated)

# 設置動畫
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_zlim([-5, 15])

# 初始化三條線條
leader_line, = ax.plot([], [], [], label="Leader Trajectory", color='blue')
f1_line, = ax.plot([], [], [], label="Follower1 Trajectory", color='yellow')
f2_line, = ax.plot([], [], [], label="Follower2 Trajectory", color='green')

# 初始和目標點，以紅色標記並用黑色標注座標
ax.scatter(P_l_start[0], P_l_start[1], P_l_start[2], color='red', label='Start Point', s=100)
ax.text(P_l_start[0], P_l_start[1], P_l_start[2], f'({P_l_start[0]:.2f}, {P_l_start[1]:.2f}, {P_l_start[2]:.2f})', color='black')

ax.scatter(P_l_goal[0], P_l_goal[1], P_l_goal[2], color='red', label='Goal Point', s=100)
ax.text(P_l_goal[0], P_l_goal[1], P_l_goal[2], f'({P_l_goal[0]:.2f}, {P_l_goal[1]:.2f}, {P_l_goal[2]:.2f})', color='black')

# Follower1 和 Follower2 的起始點
ax.scatter(P_f1_start[0], P_f1_start[1], P_f1_start[2], color='yellow', label='Follower1 Start', s=100)
ax.text(P_f1_start[0], P_f1_start[1], P_f1_start[2], f'({P_f1_start[0]:.2f}, {P_f1_start[1]:.2f}, {P_f1_start[2]:.2f})', color='black')

ax.scatter(P_f2_start[0], P_f2_start[1], P_f2_start[2], color='green', label='Follower2 Start', s=100)
ax.text(P_f2_start[0], P_f2_start[1], P_f2_start[2], f'({P_f2_start[0]:.2f}, {P_f2_start[1]:.2f}, {P_f2_start[2]:.2f})', color='black')

# 初始化函數
def init():
    leader_line.set_data([], [])
    leader_line.set_3d_properties([])
    
    f1_line.set_data([], [])
    f1_line.set_3d_properties([])
    
    f2_line.set_data([], [])
    f2_line.set_3d_properties([])
    
    return leader_line, f1_line, f2_line

# 更新每一幀的函數，以使用插值後的軌跡
def update_smooth(frame):
    leader_line.set_data(P_l_traj_interpolated[0, :frame], P_l_traj_interpolated[1, :frame])
    leader_line.set_3d_properties(P_l_traj_interpolated[2, :frame])

    f1_line.set_data(P_f1_traj_interpolated[0, :frame], P_f1_traj_interpolated[1, :frame])
    f1_line.set_3d_properties(P_f1_traj_interpolated[2, :frame])

    f2_line.set_data(P_f2_traj_interpolated[0, :frame], P_f2_traj_interpolated[1, :frame])
    f2_line.set_3d_properties(P_f2_traj_interpolated[2, :frame])

    # 在最後一幀展示Follower1和Follower2的終點
    if frame == num_frames_interpolated - 1:
        ax.scatter(P_f1_end[0], P_f1_end[1], P_f1_end[2], color='yellow', label='Follower1 End', s=100)
        ax.text(P_f1_end[0], P_f1_end[1], P_f1_end[2], f'({P_f1_end[0]:.2f}, {P_f1_end[1]:.2f}, {P_f1_end[2]:.2f})', color='black')

        ax.scatter(P_f2_end[0], P_f2_end[1], P_f2_end[2], color='green', label='Follower2 End', s=100)
        ax.text(P_f2_end[0], P_f2_end[1], P_f2_end[2], f'({P_f2_end[0]:.2f}, {P_f2_end[1]:.2f}, {P_f2_end[2]:.2f})', color='black')

        ax.plot([P_l_traj[0, -1], P_f1_end[0]], [P_l_traj[1, -1], P_f1_end[1]], [P_l_traj[2, -1], P_f1_end[2]], color='black')
        ax.plot([P_l_traj[0, -1], P_f2_end[0]], [P_l_traj[1, -1], P_f2_end[1]], [P_l_traj[2, -1], P_f2_end[2]], color='black')
        ax.plot([P_f1_end[0], P_f2_end[0]], [P_f1_end[1], P_f2_end[1]], [P_f1_end[2], P_f2_end[2]], color='black')

    return leader_line, f1_line, f2_line

# 動畫設置，將動畫的更新與MPC的狀態更新同步
interval = 100  # 假設每次MPC更新狀態的時間間隔是100毫秒
ani = FuncAnimation(fig, update_smooth, frames=num_frames_interpolated, init_func=init, blit=False, interval=interval)

ax.legend()

plt.show()
