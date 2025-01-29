"""
2次元可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import defaultdict


def triangle_circumradius(points):
    """三角形の外接円の半径を計算する関数"""
    a = np.linalg.norm(points[1] - points[2])
    b = np.linalg.norm(points[0] - points[2])
    c = np.linalg.norm(points[0] - points[1])

    # ヘロンの公式で面積を計算 https://manabitimes.jp/math/579
    area = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    if area == 0:
        return np.inf
    return (a * b * c) / (4 * area)     # https://manabitimes.jp/math/577


# ランダムな点群の生成
np.random.seed(42)
points = np.random.rand(100, 2)

# Delaunay三角形分割の計算
tri = Delaunay(points)

# Alpha値の設定
alpha = 0.2

# 各三角形の外接円半径を計算
tri_radii = []
for simplex in tri.simplices:
    tri_points = points[simplex]
    tri_radii.append(triangle_circumradius(tri_points))

# Alpha以下の三角形を抽出
valid_triangles = tri.simplices[np.array(tri_radii) <= alpha]

# 各辺の最小外接円半径を計算
edge_radii = defaultdict(list)  # 辞書を初期化
for i, simplex in enumerate(tri.simplices):
    radius = tri_radii[i]
    # 三角形なのでどの辺からやってもOK
    edges = [
        tuple(sorted((simplex[0], simplex[1]))),
        tuple(sorted((simplex[1], simplex[2]))),
        tuple(sorted((simplex[2], simplex[0])))
    ]
    for edge in edges:
        edge_radii[edge].append(radius)

# Alpha以下の辺を抽出
valid_edges = [edge for edge, radii in edge_radii.items()
               if min(radii) <= alpha]

# 可視化
plt.figure(figsize=(10, 6))
plt.gca().set_aspect('equal')

# 元のDelaunay三角形を薄いグレーで表示
plt.triplot(points[:, 0], points[:, 1], tri.simplices,
            color='gray', lw=0.5, alpha=0.3)

# Alpha複体の辺を青で強調表示
for edge in valid_edges:
    p1, p2 = points[edge[0]], points[edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)

# Alpha複体の三角形を水色で塗りつぶし
for triangle in valid_triangles:
    pts = points[triangle]
    plt.fill(pts[:, 0], pts[:, 1], edgecolor='none', alpha=0.3, color='cyan')

# 点を赤でプロット
plt.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=5)

plt.title(f'Alpha Complex (α={alpha})')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
