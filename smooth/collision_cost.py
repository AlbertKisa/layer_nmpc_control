# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Define the CollisionCost function
def CollisionCost(p_robot, p_obs, safe_dis, Qc, kappa):
    d = np.linalg.norm(p_robot - p_obs)
    cost = Qc / (1 + np.exp(kappa * (d - 2 * safe_dis)))
    return cost


# Define parameters
p_robot = np.array([0, 0])
safe_dis = 0.1
Qc = 1.0
kappa = 1.0

# Generate obstacle positions along x-axis
x_obs = np.linspace(0, 5.0, 100)
p_obs_list = np.array([x_obs, np.zeros_like(x_obs)]).T

# Compute collision costs
collision_costs = [
    CollisionCost(p_robot, p_obs, safe_dis, Qc, kappa) for p_obs in p_obs_list
]

# Plot the results
# Redraw the plot with annotation
plt.figure(figsize=(8, 6))
plt.plot(x_obs, collision_costs, label="Collision Cost", color="b")
plt.axvline(2 * safe_dis, color="r", linestyle="--", label="2 * safe_dis")
x_ticks = np.linspace(0, 5.0, 21)  # 生成从 0 到 10 的 21 个刻度
plt.xticks(x_ticks)

plt.title("Collision Cost vs. Obstacle Distance")
plt.xlabel("Distance from Robot (x-axis)")
plt.ylabel("Collision Cost")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
