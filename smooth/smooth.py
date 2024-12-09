import numpy as np
import matplotlib.pyplot as plt

# 模擬飛機原始軌跡 (震盪和超衝)
t = np.linspace(0, 10, 100)
original_trajectory = np.sin(
    2 * np.pi * t) + 0.2 * np.random.normal(size=t.shape)

# 使用多項式擬合
degree = 5  # 多項式階數
coefficients = np.polyfit(t, original_trajectory, degree)
smooth_trajectory = np.polyval(coefficients, t)

# 繪製結果
plt.figure()
plt.plot(t,
         original_trajectory,
         label="Original Trajectory",
         linestyle="--",
         alpha=0.7)
plt.plot(t, smooth_trajectory, label="Smoothed Trajectory", linewidth=2)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Trajectory Smoothing with Polynomial Fitting")
plt.show()
