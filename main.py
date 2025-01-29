import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from shapely.geometry import Polygon, Point
from collections import defaultdict


def triangle_circumradius(points):
    """三角形の外接円半径を計算"""
    a = np.linalg.norm(points[1] - points[2])
    b = np.linalg.norm(points[0] - points[2])
    c = np.linalg.norm(points[0] - points[1])
    area = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    return (a * b * c) / (4 * area) if area != 0 else np.inf


# ランダム点群の生成（可視化しやすいよう20点に減らす）
np.random.seed(42)
points = np.random.rand(3, 2)
alpha = 0.1

# 1. ランダムな点群の可視化
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], c='red', s=30)
plt.title("1. Random Point Cloud")
plt.show()

# 2. ボロノイ図の可視化
vor = Voronoi(points)
fig, ax = plt.subplots(figsize=(6, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='gray')
plt.scatter(points[:, 0], points[:, 1], c='red', s=30)
plt.title("2. Voronoi Regions")
plt.show()

# 3. ボロノイ領域とαボールの重なり部分の可視化
fig, ax = plt.subplots(figsize=(6, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='gray')
plt.scatter(points[:, 0], points[:, 1], c='red', s=30)

# αボールの描画
for point in points:
    ax.add_artist(plt.Circle(point, alpha, facecolor='blue', alpha=0.1, edgecolor='none'))

# ボロノイ領域とαボールの重なり部分を計算
for i in range(len(points)):
    region = vor.regions[vor.point_region[i]]
    if not region or -1 in region:
        continue  # 無限遠点を含む領域はスキップ

    vor_poly = Polygon(vor.vertices[region])
    circle = Point(points[i]).buffer(alpha)
    intersection = vor_poly.intersection(circle)

    if not intersection.is_empty:
        if intersection.geom_type == 'Polygon':
            x, y = intersection.exterior.xy
            ax.fill(x, y, color='green', alpha=0.3)
        elif intersection.geom_type == 'MultiPolygon':
            for poly in intersection.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color='green', alpha=0.3)

plt.title("3. Voronoi-Alpha Ball Intersections")
plt.show()

# 4. アルファ複体の構築
tri = Delaunay(points)
tri_radii = [triangle_circumradius(points[s]) for s in tri.simplices]
valid_triangles = tri.simplices[np.array(tri_radii) <= alpha]

edge_radii = defaultdict(list)
for i, simplex in enumerate(tri.simplices):
    radius = tri_radii[i]
    for edge in (sorted((simplex[0], simplex[1])),
                 sorted((simplex[1], simplex[2])),
                 sorted((simplex[2], simplex[0]))):
        edge_radii[tuple(edge)].append(radius)

valid_edges = [edge for edge, radii in edge_radii.items() if min(radii) <= alpha]

# アルファ複体の描画
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal')

# デロネー三角形を背景に表示
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='gray', lw=0.5, alpha=0.3)

# アルファ複体の辺を表示
for edge in valid_edges:
    p1, p2 = points[edge[0]], points[edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)

# アルファ複体の三角形を表示
for triangle in valid_triangles:
    pts = points[triangle]
    plt.fill(pts[:, 0], pts[:, 1], edgecolor='none', alpha=0.3, color='cyan')

plt.scatter(points[:, 0], points[:, 1], c='red', s=30)
plt.title(f'4. Alpha Complex (α={alpha})')
plt.show()