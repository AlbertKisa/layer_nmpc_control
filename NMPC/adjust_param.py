import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import euclidean
from NMPC1 import NMPCLeader

# 遗传算法相关参数
POPULATION_SIZE = 50  # 种群大小
NUM_GENERATIONS = 20  # 世代数
MUTATION_RATE = 0.2  # 突变率
CROSSOVER_RATE = 0.8  # 交叉率

# 初始范围
WEIGHT_BOUNDS = [(0.1, 10.0)] * 4  # 对应四个权重的范围


# 优化目标函数
def objective_function(weights):
    tracking_weight, collsion_weight, over_height_weight, input_change_weight = weights

    # 修改全局变量
    global tracking_weight_global, collsion_weight_global, over_height_weight_global, input_change_weight_global
    tracking_weight_global = tracking_weight
    collsion_weight_global = collsion_weight
    over_height_weight_global = over_height_weight
    input_change_weight_global = input_change_weight

    # 运行模拟，获取路径平滑性和避障性评分
    start_pose = np.array([0, 0, 1.0])
    goal_pose = np.array([2.0, 2.0, 1.0])
    obstacles = np.array([[0.4, 0.6, 1.0], [1.2, 0.8, 1.0]])
    z_limits = np.array([0.1, 1.7])
    obs_rad = 0.3

    path, final_step, vel_list, dis = NMPCLeader(start_pose, goal_pose, obstacles, obs_rad, z_limits)

    # 平滑性计算：路径点之间的角度变化总和
    smoothness_cost = 0.0
    for i in range(1, path.shape[1] - 1):
        v1 = path[:, i] - path[:, i - 1]
        v2 = path[:, i + 1] - path[:, i]
        angle = np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        )
        smoothness_cost += angle

    # 避障性计算：与障碍物的最近距离
    min_distance_to_obstacle = np.inf
    for i in range(path.shape[1]):
        for obs in obstacles:
            dist = euclidean(path[:, i], obs)
            min_distance_to_obstacle = min(min_distance_to_obstacle, dist)

    # 多目标优化，平滑性和避障性
    total_cost = smoothness_cost + 1.0 / (min_distance_to_obstacle + 1e-6)  # 避免分母为零

    return total_cost


# 遗传算法主流程
def genetic_algorithm():
    # 初始化种群
    population = np.random.uniform(
        low=[b[0] for b in WEIGHT_BOUNDS],
        high=[b[1] for b in WEIGHT_BOUNDS],
        size=(POPULATION_SIZE, len(WEIGHT_BOUNDS))
    )

    for generation in range(NUM_GENERATIONS):
        # 评估适应度
        fitness = np.array([objective_function(ind) for ind in population])
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # 输出当前最佳结果
        print(f"Generation {generation + 1}: Best Fitness = {fitness[sorted_indices[0]]}")
        print(f"Best Weights: {population[0]}")

        # 选择、交叉和突变
        new_population = population[:POPULATION_SIZE // 2].tolist()  # 保留最优一半
        while len(new_population) < POPULATION_SIZE:
            # 交叉
            if np.random.rand() < CROSSOVER_RATE:
                parents = population[np.random.choice(POPULATION_SIZE // 2, 2, replace=False)]
                crossover_point = np.random.randint(1, len(WEIGHT_BOUNDS))
                child1 = np.hstack((parents[0, :crossover_point], parents[1, crossover_point:]))
                child2 = np.hstack((parents[1, :crossover_point], parents[0, crossover_point:]))
                new_population.extend([child1, child2])

            # 突变
            if np.random.rand() < MUTATION_RATE:
                individual = population[np.random.randint(POPULATION_SIZE // 2)]
                mutation_index = np.random.randint(len(WEIGHT_BOUNDS))
                mutation_value = np.random.uniform(*WEIGHT_BOUNDS[mutation_index])
                individual[mutation_index] = mutation_value
                new_population.append(individual)

        population = np.array(new_population[:POPULATION_SIZE])

    # 返回最终的最佳个体
    fitness = np.array([objective_function(ind) for ind in population])
    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]


if __name__ == "__main__":
    # 使用遗传算法计算最佳权重
    best_weights, best_fitness = genetic_algorithm()
    print("Best Weights:", best_weights)
    print("Best Fitness:", best_fitness)
