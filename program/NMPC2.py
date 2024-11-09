import cvxpy as cp
import numpy as np

A = np.block([[np.zeros((3, 3)), np.eye(3)],
              [np.zeros((3, 3)), np.zeros((3, 3))]])

B = np.block([[np.zeros((3, 3))], [np.eye(3)]])

C = np.eye(6)


def mpc2_follower(P_f_start, P_l, d, Np=80, Nc=10):
    # 定義狀態
    u = cp.Variable((3, Np))  # 控制输入 (速度)
    P_f = cp.Variable((3, Np + 1))  # Follower 位置

    # 最小化 Follower 跟 Leader 的距離 用d1 d2向量
    Q = np.eye(3)
    R = 0.1 * np.eye(3)

    # 目標函數
    cost = 0
    for k in range(Np):
        cost += cp.quad_form(P_f[:, k] -
                             (P_l[:, k] - d), Q) + cp.quad_form(u[:, k], R)

    # 約束條件
    constraints = [P_f[:, 0] == P_f_start]
    for k in range(Np):
        constraints += [P_f[:, k + 1] == P_f[:, k] + u[:, k]]

    # 求解最佳化
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return P_f.value, u.value  # 返回位置和控制输入
