import numpy as np


def generate_nd_archimedean_spiral(
        n_dim: int,
        theta_max: float = 8 * np.pi,
        num_points: int = 1000,
        a: float = 0.0,
        b: float = 1.0,
        omega: float = 1.0,
        c: float = 1.0
) -> np.ndarray:
    """
    n次元アルキメデス螺旋を生成する関数

    Args:
        n_dim (int): 次元数
        theta_max (float): θの最大値 (default: 8π)
        num_points (int): 生成する点の数 (default: 1000)
        a (float): 半径のオフセット (default: 0)
        b (float): 半径の増加率 (default: 1)
        omega (float): 角周波数 (default: 1)
        c (float): 奇数次元の線形成分の係数 (default: 1)

    Returns:
        np.ndarray: 螺旋の座標配列 (shape: [num_points, n_dim])
    """
    theta = np.linspace(0, theta_max, num_points)
    coordinates = []

    # 2次元ペアごとに螺旋を生成
    for _ in range(n_dim // 2):
        r = a + b * theta
        x = r * np.cos(omega * theta)
        y = r * np.sin(omega * theta)
        coordinates.extend([x, y])

    # 奇数次元の場合、線形成分を追加
    if n_dim % 2 != 0:
        z = c * theta
        coordinates.append(z)

    return np.array(coordinates).T


# 使用例と可視化（3次元の場合）
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 3次元螺旋の生成
    n = 3
    spiral_3d = generate_nd_archimedean_spiral(n, theta_max=6 * np.pi, c=0.5)

    # 3Dプロット
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(spiral_3d[:, 0], spiral_3d[:, 1], spiral_3d[:, 2])

    ax.set_title(f"{n}D Archimedean Spiral")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()