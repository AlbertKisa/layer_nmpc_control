import cvxpy as cp
import numpy as np

A = np.block([[np.zeros((3, 3)), np.eye(3)], 
              [np.zeros((3, 3)), np.zeros((3, 3))]])

B = np.block([[np.zeros((3, 3))], 
              [np.eye(3)]])

C = np.eye(6)

def mpc3_distance(P_f1, P_f2, d3, Np=20, Nc=10):
    # 定義狀態
    u1 = cp.Variable((3, Np))  # Follower1 的控制输入
    u2 = cp.Variable((3, Np))  # Follower2 的控制输入
    P_f1_next = cp.Variable((3, Np+1))  # Follower1 位置
    P_f2_next = cp.Variable((3, Np+1))  # Follower2 位置

    # 固定f1 f2之間的距離為6
    Q = np.eye(3)
    R = 0.1 * np.eye(3)

    # 目標函數
    cost = 0
    for k in range(Np):
        cost += cp.quad_form((P_f1_next[:, k] - P_f2_next[:, k]) - d3, Q) + cp.quad_form(u1[:, k], R) + cp.quad_form(u2[:, k], R)

    # 約束條件
    constraints = [P_f1_next[:, 0] == P_f1]
    constraints += [P_f2_next[:, 0] == P_f2]
    for k in range(Np):
        constraints += [P_f1_next[:, k+1] == P_f1_next[:, k] + u1[:, k]]
        constraints += [P_f2_next[:, k+1] == P_f2_next[:, k] + u2[:, k]]

    # 求解最佳化
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return P_f1_next.value, P_f2_next.value, u1.value, u2.value
