import numpy as np
import matplotlib.pyplot as plt


def generate_triangle_vertices(start_vertex, direction, side_length):
    # 计算旋转矩阵
    theta = np.radians(direction)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # 正三角形顶点相对位置
    vertex_offsets = np.array([[0, 0], [side_length, 0],
                               [side_length / 2,
                                np.sqrt(3) * side_length / 2]])

    # 旋转和平移顶点
    vertices = start_vertex + np.dot(vertex_offsets, rotation_matrix.T)
    return vertices


def compute_midpoints(vertices):
    midpoints = (vertices + np.roll(vertices, -1, axis=0)) / 2
    return midpoints


def GenerateTrianglePoints(start_vertex, direction, side_length):
    # 计算旋转矩阵
    theta = np.radians(direction)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # 正三角形顶点相对位置
    vertex_offsets = np.array([[0, 0], [side_length, 0],
                               [side_length / 2,
                                np.sqrt(3) * side_length / 2]])

    # 旋转和平移顶点
    vertices = start_vertex + np.dot(vertex_offsets, rotation_matrix.T)

    midpoints = (vertices + np.roll(vertices, -1, axis=0)) / 2

    all_points = np.vstack((vertices[0], midpoints[0], vertices[1],
                            midpoints[1], vertices[2], midpoints[2]))
    all_points = np.hstack((all_points, np.zeros((6, 1))))

    return all_points


def plot_triangle(vertices, midpoints):
    plt.figure(figsize=(8, 6))
    # 绘制三角形
    triangle = np.vstack((vertices, vertices[0]))
    plt.plot(triangle[:, 0], triangle[:, 1], 'bo-', label="Vertices")

    # 标记顶点
    for i, (x, y) in enumerate(vertices):
        plt.text(x, y, f'V{i+1}', fontsize=12, ha='right')

    # 绘制边中点
    for i, (x, y) in enumerate(midpoints):
        plt.plot(x, y, 'ro', label="Midpoints" if i == 0 else "")
        plt.text(x, y, f'M{i+1}', fontsize=12, ha='left')

    plt.title("Equilateral Triangle")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


# 示例输入
start_vertex = np.array([0, 0, 0])
direction = 195  # 旋转角度
side_length = 0.4

# 计算顶点和边中点
vertices = generate_triangle_vertices(start_vertex[0:2], direction,
                                      side_length)
midpoints = compute_midpoints(vertices)
all_points = GenerateTrianglePoints(start_vertex[0:2], direction, side_length)
print(all_points)

# 绘制结果
plot_triangle(vertices, midpoints)
