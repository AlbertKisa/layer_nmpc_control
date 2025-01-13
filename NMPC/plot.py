import matplotlib.pyplot as plt
import numpy as np
from utils import GenerateHexagonVertices
from utils import GenerateTrianglePoints
from utils import GenerateRhombusFromFront
from utils import GeneratePyramid

font_size = 14


def plot_pyramid(start_vertex, side_length, figur_num):
    """
    绘制金字塔
    :param vertices: 金字塔的顶点坐标 (5x3)
    """
    vertices = GeneratePyramid(start_vertex, side_length)
    fig = plt.figure(figur_num)
    ax = fig.add_subplot(111, projection='3d')

    # 顶点
    ax.scatter(start_vertex[0], start_vertex[1], start_vertex[2], c='r')
    ax.scatter(vertices[1:, 0], vertices[1:, 1], vertices[1:, 2], c='g')

    # 边
    edges = [
        [vertices[0], vertices[1]],  # 顶点 -> 右前
        [vertices[0], vertices[2]],  # 顶点 -> 左前
        [vertices[0], vertices[3]],  # 顶点 -> 左后
        [vertices[0], vertices[4]],  # 顶点 -> 右后
        [vertices[1], vertices[2]],  # 底面边
        [vertices[2], vertices[3]],
        [vertices[3], vertices[4]],
        [vertices[4], vertices[1]],
    ]

    for edge in edges:
        ax.plot(*zip(*edge), color='b')

    # 添加标签
    ax.text(start_vertex[0] + 0.05,
            start_vertex[1],
            start_vertex[2],
            'Leader',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    for i in range(1, len(vertices)):
        ax.text(vertices[i, 0] - 0.05,
                vertices[i, 1] - 0.02,
                vertices[i, 2],
                f'F{i}',
                color='black',
                fontsize=font_size,
                ha='center',
                va='bottom')

    # 设置轴标签和范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_3d_rhombus(start_vertex, side_length, figur_num):
    """
    绘制立体菱形
    :param vertices: 菱形的顶点坐标
    """
    vertices = GenerateRhombusFromFront(start_vertex, side_length)
    fig = plt.figure(figur_num)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制菱形的顶点
    ax.scatter(start_vertex[0],
               start_vertex[1],
               start_vertex[2],
               color='r',
               s=50)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='g', s=50)

    # 绘制菱形的边
    edges = [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),  # 顶部连接
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),  # 底部连接
        (2, 4),
        (3, 4),
        (2, 5),
        (3, 5)  # 中间连接
    ]

    for edge in edges:
        ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                [vertices[edge[0], 1], vertices[edge[1], 1]],
                [vertices[edge[0], 2], vertices[edge[1], 2]],
                color='b')
    # 添加标签
    ax.text(start_vertex[0] + 0.05,
            start_vertex[1],
            start_vertex[2],
            'Leader',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.text(vertices[0, 0] - 0.05,
            vertices[0, 1] - 0.02,
            vertices[0, 2],
            'F1',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.text(vertices[1, 0] - 0.05,
            vertices[1, 1] - 0.02,
            vertices[1, 2],
            'F5',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.text(vertices[2, 0] - 0.05,
            vertices[2, 1] - 0.02,
            vertices[2, 2],
            'F4',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.text(vertices[3, 0] - 0.05,
            vertices[3, 1] - 0.02,
            vertices[3, 2],
            'F2',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.text(vertices[5, 0] - 0.05,
            vertices[5, 1] - 0.02,
            vertices[5, 2],
            'F3',
            color='black',
            fontsize=font_size,
            ha='center',
            va='bottom')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_triangle_points(start_vertex, direction, side_length, figur_num):
    points = GenerateTrianglePoints(start_vertex, direction, side_length)

    # 绘制三角形点
    plt.figure(figur_num, figsize=(6, 6))

    # 绘制顶点
    plt.scatter(points[0, 0], points[0, 1], color="r", marker="o")

    # 绘制各点
    plt.scatter(points[1:, 0], points[1:, 1], c='g', marker='o')

    # 绘制连线，连接顶点和中点
    plt.plot(points[:, 0], points[
        :,
        1,
    ], color="b", linestyle="-")
    plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]],
             color="b",
             linestyle="-")
    plt.plot([points[3, 0], points[1, 0]], [points[3, 1], points[1, 1]],
             color="b",
             linestyle="-")
    plt.plot([points[3, 0], points[5, 0]], [points[3, 1], points[5, 1]],
             color="b",
             linestyle="-")
    plt.plot([points[1, 0], points[5, 0]], [points[1, 1], points[5, 1]],
             color="b",
             linestyle="-")

    # 添加 X 和 Y 轴
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    # 添加标签
    plt.text(start_vertex[0] + 0.05,
             start_vertex[1],
             'Leader',
             color='black',
             fontsize=font_size,
             ha='center',
             va='bottom')

    for i in range(1, len(points)):
        plt.text(points[i, 0] - 0.05,
                 points[i, 1] - 0.02,
                 f'F{i}',
                 color='black',
                 fontsize=font_size,
                 ha='center',
                 va='bottom')

    plt.grid(True)
    plt.axis('equal')


def plot_hexagon_points(start_vertex, side_length, figur_num):
    x, y = GenerateHexagonVertices(side_length)
    plt.figure(figur_num, figsize=(6, 6))
    plt.plot(x, y, color="b", linestyle="-")
    plt.scatter(x, y, color="g", marker="o")
    plt.scatter(start_vertex[0], start_vertex[1], color="r", marker="o")
    # 添加 X 和 Y 轴
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    # 添加标签
    plt.text(start_vertex[0],
             start_vertex[1],
             'Leader',
             color='black',
             fontsize=font_size,
             ha='center',
             va='bottom')
    for i, (x_pos, y_pos) in enumerate(zip(x, y)):
        plt.text(x_pos + 0.02,
                 y_pos,
                 f'F{i+1}',
                 color='black',
                 fontsize=font_size,
                 ha='center',
                 va='bottom')

    plt.grid(True)
    plt.axis('equal')


if __name__ == "__main__":
    leader = np.array([0, 0])
    side_len = 0.3
    figur_num = 1
    plot_hexagon_points(leader, side_len, figur_num)

    side_len = 0.6
    heading_angle = 150.0
    figur_num = 2
    plot_triangle_points(leader, heading_angle, side_len, figur_num)

    leader_3d = np.array([0, 0, 1.0])
    side_len = 0.3
    figur_num = 3
    plot_3d_rhombus(leader_3d, side_len, figur_num)

    leader_3d = np.array([0, 0, 1.0])
    side_len = 0.3
    figur_num = 4
    plot_pyramid(leader_3d, side_len, figur_num)

    plt.show()
