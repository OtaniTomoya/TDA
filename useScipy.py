"""
ランダム点群→アルファ複体
可視化
"""


import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def nsphere_radius(simplex_points):
    """n次元単体の外接超球の半径を計算"""
    A = simplex_points[1:] - simplex_points[0]
    A = 2 * A
    b = np.sum(simplex_points[1:] ** 2 - simplex_points[0] ** 2, axis=1)
    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        return np.inf
    radius = np.sqrt(np.sum((simplex_points[0] - x) ** 2))
    return radius


def calculate_alpha_complex(points, alpha=2.5, max_dim=5):
    """Alpha複体の主要計算関数"""
    tri = Delaunay(points)
    filtration = defaultdict(lambda: np.inf)

    # 全単体の外接球半径を計算
    for simplex in tri.simplices:
        simplex_points = points[simplex]
        radius = nsphere_radius(simplex_points)

        # 全ての部分単体を記録
        for dim in range(1, max_dim + 1):
            for subsimplex in combinations(simplex, dim + 1):
                key = tuple(sorted(subsimplex))
                filtration[key] = min(filtration[key], radius)

    # Alphaでフィルタリング
    valid_structures = [k for k, v in filtration.items() if v <= alpha]
    return valid_structures, tri


def visualize_projection(points, structures, dimensions=2):

    plt.figure(figsize=(10, 6))
    if dimensions == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(*points.T, c='red')
    else:
        plt.scatter(*points.T, c='red')

    # 辺の描画
    for structure in structures:
        if len(structure) == 2:  # 1-単体のみ描画
            line = points[list(structure)]
            if dimensions == 3:
                ax.plot(*line.T, color='blue', alpha=0.3)
            else:
                plt.plot(*line.T, color='blue', alpha=0.3)

    title = f'{dimensions} Alpha Complex'
    plt.title(title)
    plt.show()


def main():
    """メイン関数"""
    # パラメータ設定
    np.random.seed(42)
    n_points = 20
    alpha = 1
    max_dim = 10

    # データ生成
    points = StandardScaler().fit_transform(np.random.rand(n_points, max_dim))

    # Alpha複体計算
    structures, tri = calculate_alpha_complex(points, alpha, max_dim)

    # 結果表示
    dim_counts = defaultdict(int)
    for s in structures:
        dim_counts[len(s) - 1] += 1

    print(f"=== Alpha Complex Statistics (α={alpha}) ===")
    print(f"Total structures: {len(structures)}")
    print("Dimension distribution:")
    for dim, count in sorted(dim_counts.items()):
        print(f" {dim}-simplices: {count}")

    # 可視化
    # visualize_projection(points, structures, dimensions=3)


if __name__ == "__main__":
    main()