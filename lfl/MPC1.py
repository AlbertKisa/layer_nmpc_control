import cvxpy as cp
import numpy as np

A = np.block([[np.zeros((3, 3)), np.eye(3)], 
              [np.zeros((3, 3)), np.zeros((3, 3))]])

B = np.block([[np.zeros((3, 3))], 
              [np.eye(3)]])

C = np.eye(6)

def mpc1_leader(P_l_start, P_l_goal, Np=20, Nc=10):
    # 定義狀態
    u = cp.Variable((3, Np))  # 控制输入 (速度)
    P_l = cp.Variable((3, Np+1))  # Leader 位置 (x, y, z)

    # 最小化 Leader 到目標點的距離跟權重
    Q = np.eye(3)
    R = 0.1 * np.eye(3)

    # 目標函數
    cost = 0
    for k in range(Np):
        cost += cp.quad_form(P_l[:, k] - P_l_goal, Q) + cp.quad_form(u[:, k], R)

    # 約束條件
    constraints = [P_l[:, 0] == P_l_start]
    for k in range(Np):
        constraints += [P_l[:, k+1] == P_l[:, k] + u[:, k]]

    # 求解最佳化
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return P_l.value, u.value  # 回到位置和控制输入
