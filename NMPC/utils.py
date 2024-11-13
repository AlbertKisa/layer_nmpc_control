import numpy as np


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


if __name__ == "__main__":
    p = (1, 2)
    a = (0, 0)
    b = (3, 3)
    d = PointToSegmentDistance(p, a, b)
    print("d:", d)
