import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from matplotlib.animation import FuncAnimation

def triangle_circumradius(points):
    """三角形の外接円半径を計算"""
    a = np.linalg.norm(points[1] - points[2])
    b = np.linalg.norm(points[0] - points[2])
    c = np.linalg.norm(points[0] - points[1])
    area = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    return (a * b * c) / (4 * area) if area != 0 else np.inf

# ランダム点群の生成
np.random.seed(42)
points = np.random.rand(100, 2)

# FigureとAxesオブジェクトを作成
fig, axes = plt.subplots(1,2, figsize=(24, 12))

# ボロノイ図
vor = Voronoi(points)

# Delaunay三角形分割
tri = Delaunay(points)

def animate(i):
    alpha = (i + 1) * 0.001

    # 1. ランダムな点群の可視化
    # axes[0, 0].clear()
    # axes[0, 0].scatter(points[:, 0], points[:, 1], c='red', s=30)
    # axes[0, 0].set_title("1. Random Point Cloud")

    # 2. ボロノイ図の可視化
    # axes[0, 1].clear()
    # voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, show_points=False, line_colors='gray')
    # axes[0, 1].scatter(points[:, 0], points[:, 1], c='red', s=30)
    # axes[0, 1].set_title("2. Voronoi Regions")

    # 3. ボロノイ領域とαボールの重なり部分の可視化
    axes[0].clear()
    voronoi_plot_2d(vor, ax=axes[0], show_vertices=False, show_points=False, line_colors='gray')
    axes[0].scatter(points[:, 0], points[:, 1], c='red', s=30)
    for point in points:
        axes[0].add_artist(plt.Circle(point, alpha, facecolor='blue', alpha=0.1, edgecolor='none'))
    axes[0].set_title("3. Voronoi-Alpha Ball Intersections")

    # 4. アルファ複体の構築
    axes[1].clear()
    voronoi_plot_2d(vor, ax=axes[1], show_vertices=False, show_points=False, line_colors='gray')
    for point in points:
        axes[1].add_artist(plt.Circle(point, alpha, facecolor='blue', alpha=0.1, edgecolor='none'))

    # アルファ複体の辺を表示
    valid_edges = []
    for edge in tri.simplices:
        for i in range(3):
            p1, p2 = edge[i], edge[(i + 1) % 3]
            dist = np.linalg.norm(points[p1] - points[p2])
            if dist <= 2 * alpha:
                valid_edges.append((p1, p2))
    valid_edges = list(set(valid_edges))  # 重複を削除

    for edge in valid_edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)

    # アルファ複体の三角形を表示
    tri_radii = [triangle_circumradius(points[s]) for s in tri.simplices]
    valid_triangles = tri.simplices[np.array(tri_radii) <= alpha]
    for triangle in valid_triangles:
        pts = points[triangle]
        axes[1].fill(pts[:, 0], pts[:, 1], edgecolor='none', alpha=0.3, color='cyan')

    axes[1].scatter(points[:, 0], points[:, 1], c='red', s=30)
    axes[1].set_title(f'4. Alpha Complex (α={alpha:.2f})')

    # グラフ間のスペース調整
    plt.tight_layout()

# アニメーションの作成
ani = FuncAnimation(fig, animate, frames=200, interval=50, repeat=False)

# 動画として保存
ani.save('alpha_complex_animation.mp4', writer='ffmpeg')

plt.show()