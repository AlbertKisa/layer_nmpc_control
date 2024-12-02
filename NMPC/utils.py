import numpy as np
import matplotlib.pyplot as plt


def PointToSegmentDistance(p, seg_begin, seg_end):

    be = seg_end - seg_begin
    bp = p - seg_begin

    seg_len_squared = np.dot(be, be)
    if seg_len_squared == 0:
        return np.linalg.norm(bp)
    t = np.dot(be, bp) / seg_len_squared

    t = max(0.0, min(t, 1.0))
    D = seg_begin + t * be
    pd = D - p

    return np.linalg.norm(pd)


def GenerateHexagonVertices(radius=1.0):
    angles = np.linspace(0, 2 * np.pi, 7)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    return x, y


def GenerateRhombusFromFront(front, size):
    """
    根据前点坐标和尺寸生成立体菱形的顶点
    :param front: 菱形的前点坐标 (x, y, z)
    :param size: 菱形的尺寸（对角线长度的一半）
    :return: 菱形的所有顶点坐标
    """
    fx, fy, fz = front
    # 计算中心点
    center = np.array([fx, fy - size, fz])

    # 根据对称关系生成其余顶点
    vertices = np.array([
        center + [0, 0, size],  # 顶部
        center - [0, 0, size],  # 底部
        center + [size, 0, 0],  # 右
        center - [size, 0, 0],  # 左
        front,  # 前
        center - (front - center),  # 后
    ])

    return vertices


def plot_3d_rhombus(vertices):
    """
    绘制立体菱形
    :param vertices: 菱形的顶点坐标
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制菱形的顶点
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=50)

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('3D Rhombus')
    plt.show()


if __name__ == "__main__":
    p = (1, 2)
    a = (0, 0)
    b = (3, 3)
    d = PointToSegmentDistance(p, a, b)
    print("d:", d)
