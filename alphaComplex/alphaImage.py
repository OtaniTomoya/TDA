import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from shapely.geometry import Polygon, Point


def triangle_circumradius(points):
    """三角形の外接円半径を計算"""
    a = np.linalg.norm(points[1] - points[2])
    b = np.linalg.norm(points[0] - points[2])
    c = np.linalg.norm(points[0] - points[1])
    area = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    return (a * b * c) / (4 * area) if area != 0 else np.inf


# ランダム点群の生成
np.random.seed(42)
points = np.random.rand(20, 2)
for alpha in range(1, 50):
    alpha = alpha * 0.01
    # FigureとAxesオブジェクトを作成
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # 2x2のレイアウトで、Figureサイズを12x12に設定

    # 1. ランダムな点群の可視化
    axes[0, 0].scatter(points[:, 0], points[:, 1], c='red', s=30)
    axes[0, 0].set_title("1. Random Point Cloud")

    # 2. ボロノイ図の可視化
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, show_points=False, line_colors='gray')
    axes[0, 1].scatter(points[:, 0], points[:, 1], c='red', s=30)
    axes[0, 1].set_title("2. Voronoi Regions")

    # 3. ボロノイ領域とαボールの重なり部分の可視化
    voronoi_plot_2d(vor, ax=axes[1, 0], show_vertices=False, show_points=False, line_colors='gray')
    axes[1, 0].scatter(points[:, 0], points[:, 1], c='red', s=30)

    for point in points:
        axes[1, 0].add_artist(plt.Circle(point, alpha, facecolor='blue', alpha=0.1, edgecolor='none'))  # 修正済み

    for i in range(len(points)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            continue

        vor_poly = Polygon(vor.vertices[region])
        circle = Point(points[i]).buffer(alpha)
        intersection = vor_poly.intersection(circle)

    axes[1, 0].set_title("3. Voronoi-Alpha Ball Intersections")

    # 4. アルファ複体の構築
    tri = Delaunay(points)
    tri_radii = [triangle_circumradius(points[s]) for s in tri.simplices]
    valid_triangles = tri.simplices[np.array(tri_radii) <= alpha]

    voronoi_plot_2d(vor, ax=axes[1, 1], show_vertices=False, show_points=False, line_colors='gray')
    for point in points:
        axes[1, 1].add_artist(plt.Circle(point, alpha, facecolor='blue', alpha=0.1, edgecolor='none'))  # 修正済み

    valid_edges = []
    for edge in tri.simplices:
        for i in range(3):
            p1, p2 = edge[i], edge[(i + 1) % 3]
            dist = np.linalg.norm(points[p1] - points[p2])
            if dist <= 2 * alpha:
                valid_edges.append((p1, p2))
    valid_edges = list(set(valid_edges))  # 重複を削除

    # アルファ複体の描画
    axes[1, 1].set_aspect('equal')

    # アルファ複体の辺を表示
    for edge in valid_edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        axes[1, 1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)

    # アルファ複体の三角形を表示
    for triangle in valid_triangles:
        pts = points[triangle]
        axes[1, 1].fill(pts[:, 0], pts[:, 1], edgecolor='none', alpha=0.3, color='cyan')

    axes[1, 1].scatter(points[:, 0], points[:, 1], c='red', s=30)
    axes[1, 1].set_title(f'4. Alpha Complex (α={alpha})')

    # グラフ間のスペース調整
    plt.tight_layout()

    # 画像として保存
    plt.savefig(f"../image/alpha_complex{alpha}.png")
    plt.show()
