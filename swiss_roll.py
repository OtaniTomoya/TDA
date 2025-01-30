import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# スイスロールデータ生成
X, color = make_swiss_roll(n_samples=1000, noise=0.1)
X = X[:, [0, 2, 1]]  # 視覚化のため軸を調整

# 3D Delaunay四面体分割
tri = Delaunay(X)

# 図の準備
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# アニメーションパラメータ
alpha_values = np.linspace(0.1, 5, 100)

# 単体のプロパティを格納する辞書
simplex_properties = {}

def calculate_simplex_properties(points):
    """単体（辺、三角形、四面体）の外接球半径を計算"""
    key = frozenset(tuple(p) for p in points)
    if key in simplex_properties:
        return simplex_properties[key]

    dim = points.shape[1]
    if dim == 1:  # 辺
        radius = np.linalg.norm(points[1] - points[0]) / 2
    else:
        A = np.hstack([points, np.ones((points.shape[0], 1))])
        b = np.sum(points ** 2, axis=1)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        center = x[:-1] / 2
        radius = np.sqrt(x[-1] + np.sum(center ** 2))

    simplex_properties[key] = radius  # 計算結果を保存
    return radius


def animate(i):
    print(i, end="")
    alpha = alpha_values[i]

    # 前のフレームをクリア
    ax2.clear()

    # 全単体を収集
    edges = set()
    triangles = set()
    tetrahedrons = set()

    # 四面体をチェック
    for tetra in tri.simplices:
        pts = X[tetra]
        radius = calculate_simplex_properties(pts)
        if radius <= alpha:
            tetrahedrons.add(frozenset(tetra))

            # 四面体の面を三角形として追加
            for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
                triangles.add(frozenset(tetra[face]))

            # 四面体の辺を追加
            for edge in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
                edges.add(frozenset(tetra[edge]))

    # 三角形をチェック（四面体に含まれないもの）
    for simplex in tri.simplices:
        for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
            triangle = frozenset(simplex[face])
            if triangle not in triangles:
                pts = X[list(triangle)]
                radius = calculate_simplex_properties(pts)
                if radius <= alpha:
                    triangles.add(triangle)

                    # 辺を追加 (修正)
                    for edge in [[0, 1], [1, 2], [2, 0]]:
                        edges.add(frozenset(simplex[face][edge]))  # simplex から直接辺を取得

    # プロット
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=20, alpha=0.5)
    # ax1.view_init(elev=i * 2, azim=i * 2)
    ax1.view_init(elev=75)
    ax1.set_title('Swiss Roll')

    # アルファ複体のプロット
    for edge in edges:
        line = X[list(edge)]
        ax2.plot(*line.T, color='blue', lw=1)

    for triangle in triangles:
        verts = X[list(triangle)]
        tri_obj = Poly3DCollection([verts], alpha=0.3, edgecolor='k')
        tri_obj.set_facecolor('cyan')
        ax2.add_collection3d(tri_obj)

    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=20)
    # ax2.view_init(elev=i*2, azim=i * 2)
    ax2.view_init(elev=75)
    ax2.set_title(f'Alpha Complex (α={alpha:.1f})')

    ax2.set_xlim(X[:, 0].min(), X[:, 0].max())
    ax2.set_ylim(X[:, 1].min(), X[:, 1].max())
    ax2.set_zlim(X[:, 2].min(), X[:, 2].max())


# アニメーション作成
ani = FuncAnimation(fig, animate, frames=len(alpha_values), interval=100)
import time

t = time.time()
# 動画保存
ani.save(f'alpha_complex_3d{t}.mp4', writer='ffmpeg', dpi=100)

plt.show()