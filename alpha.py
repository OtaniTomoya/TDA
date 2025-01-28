import numpy as np
from itertools import combinations


def compute_circumcircle(a: np.ndarray, b, c):
    """3点の外接円の中心と半径を計算する関数"""
    mat = np.array([
        [a[0], a[1], 1],
        [b[0], b[1], 1],
        [c[0], c[1], 1]
    ])
    det = np.linalg.det(mat)
    if np.isclose(det, 0):
        return None, None  # 同一直線上の点

    A = np.array([
        [b[0] - a[0], b[1] - a[1]],
        [c[0] - a[0], c[1] - a[1]]
    ])
    b_vec = np.array([
        (b[0] ** 2 - a[0] ** 2 + b[1] ** 2 - a[1] ** 2) / 2,
        (c[0] ** 2 - a[0] ** 2 + c[1] ** 2 - a[1] ** 2) / 2
    ])
    try:
        center = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        return None, None
    radius = np.linalg.norm(center - a)
    return center, radius


def compute_delaunay_complex(points):
    """ドロネー複体を計算する関数"""
    n = points.shape[0]
    delaunay_triangles = []

    # 全ての3点の組み合わせをチェック
    for tri_indices in combinations(range(n), 3):
        a, b, c = points[list(tri_indices)]
        center, radius = compute_circumcircle(a, b, c)
        if center is None:
            continue

        # 他の点が外接円内にないかチェック
        is_delaunay = True
        for i in range(n):
            if i in tri_indices:
                continue
            p = points[i]
            dist = np.linalg.norm(p - center)
            if dist < radius - 1e-8:
                is_delaunay = False
                break

        if is_delaunay:
            sorted_tri = tuple(sorted(tri_indices))
            delaunay_triangles.append(sorted_tri)

    # 重複排除
    delaunay_triangles = list(set(delaunay_triangles))

    # 辺を収集
    edges = set()
    for tri in delaunay_triangles:
        edges.update([
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[0], tri[2])))
        ])

    return list(range(n)), sorted(edges), delaunay_triangles


def build_boundary_matrices(vertices, edges, triangles):
    """境界作用素を構築する関数"""
    num_vertices = len(vertices)
    num_edges = len(edges)
    num_triangles = len(triangles)

    # D1: 0次元→1次元
    D1 = np.zeros((num_vertices, num_edges), dtype=int)
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    for i, (v1, v2) in enumerate(edges):
        D1[v1, i] = 1
        D1[v2, i] = 1

    # D2: 1次元→2次元
    D2 = np.zeros((num_edges, num_triangles), dtype=int)
    for tri_idx, tri in enumerate(triangles):
        for e in [tuple(sorted((tri[0], tri[1]))),
                  tuple(sorted((tri[1], tri[2]))),
                  tuple(sorted((tri[0], tri[2])))]:
            if e in edge_to_idx:
                D2[edge_to_idx[e], tri_idx] = 1

    return D1, D2


def rank_mod2(matrix):
    """mod2での行列のランクを計算する関数"""
    m = matrix.copy()
    rows, cols = m.shape
    rank = 0

    for col in range(cols):
        pivot = -1
        for row in range(rank, rows):
            if m[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        m[[rank, pivot]] = m[[pivot, rank]]

        for row in range(rows):
            if row != rank and m[row, col] % 2 == 1:
                m[row] = (m[row] + m[rank]) % 2

        rank += 1

    return rank


def calculate_betti_numbers(D1, D2):
    """ベッチ数を計算する関数"""
    rank_D1 = rank_mod2(D1)
    rank_D2 = rank_mod2(D2) if D2.size > 0 else 0

    betti0 = D1.shape[0] - rank_D1
    betti1 = (D1.shape[1] - rank_D1) - rank_D2
    betti2 = D2.shape[1] - rank_D2 if D2.size > 0 else 0

    return betti0, betti1, betti2


def main(points):
    """メイン関数"""
    vertices, edges, triangles = compute_delaunay_complex(points)

    if len(triangles) == 0:
        D2 = np.zeros((len(edges), 0), dtype=int)
    else:
        D1, D2 = build_boundary_matrices(vertices, edges, triangles)

    D1, _ = build_boundary_matrices(vertices, edges, triangles)
    betti0, betti1, betti2 = calculate_betti_numbers(D1, D2)

    print(f"Betti numbers:")
    print(f"H0: {betti0}")
    print(f"H1: {betti1}")
    print(f"H2: {betti2}")


# 使用例
if __name__ == "__main__":
    # 正方形の4点
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    main(points)
