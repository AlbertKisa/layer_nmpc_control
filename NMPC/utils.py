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


def GeneratePyramid(vertex, side_length):
    """
    根据顶点坐标和边长生成金字塔的5个点坐标
    :param vertex: 金字塔的顶点坐标 (x, y, z)
    :param side_length: 底面正方形的边长
    :return: 金字塔的所有顶点坐标
    """
    vx, vy, vz = vertex
    # 底面中心的坐标
    base_center = np.array([vx, vy, vz - side_length])

    # 计算底面四个顶点
    half_side = side_length / 2
    base_vertices = np.array([
        base_center + [half_side, half_side, 0],  # 右前
        base_center + [-half_side, half_side, 0],  # 左前
        base_center + [-half_side, -half_side, 0],  # 左后
        base_center + [half_side, -half_side, 0],  # 右后
    ])

    # 返回顶点和底面四点
    vertices = np.vstack([vertex, base_vertices])
    return vertices


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


def GenerateRandomFloats(count, start, end):
    return np.random.uniform(start, end, count).tolist()


if __name__ == "__main__":
    p = (1, 2)
    a = (0, 0)
    b = (3, 3)
    d = PointToSegmentDistance(p, a, b)
    print("d:", d)
