import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform


def generate_alpha_complex_1_skeleton(points, alpha):
    """
    与えられた点集合に対し、Delaunay 三角形分割を行い、
    各三角形の辺の長さが alpha 以下の場合にその辺を 1-骨格に含める。
    ここでは簡略のため「辺が alpha 以下」かどうかのみで採用を判断する。

    Parameters:
    -----------
    points : ndarray of shape (n, d)
        入力の座標データ
    alpha : float
        辺を含めるかどうかの閾値 (簡易的なアルファ複体用)

    Returns:
    --------
    adjacency_matrix : ndarray of shape (n, n)
        アルファ複体の 1-骨格を表す隣接行列（非対称成分は持たない）
        辺があればその距離を、無ければ np.inf を格納
    """
    n = len(points)
    # 初期状態は全て無限大距離とする（対角要素は0）
    adjacency_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(adjacency_matrix, 0.0)

    # Delaunay 三角形分割
    tri = Delaunay(points)

    # Delaunay で得られた単体（simplices）から辺を取り出す
    # 三角形の頂点インデックス (i, j, k) に対し (i, j), (j, k), (k, i) の辺を確認
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                i_idx = simplex[i]
                j_idx = simplex[j]
                # 距離を計算
                dist_ij = np.linalg.norm(points[i_idx] - points[j_idx])
                # alpha 以下なら採用
                if dist_ij <= alpha:
                    adjacency_matrix[i_idx, j_idx] = dist_ij
                    adjacency_matrix[j_idx, i_idx] = dist_ij

    return adjacency_matrix


def floyd_warshall_distances(adjacency_matrix):
    """
    フロイド–ウォーシャル法により、グラフの全点対最短路長を計算する。

    Parameters:
    -----------
    adjacency_matrix : ndarray of shape (n, n)
        グラフの隣接行列。存在しない辺は np.inf で表されていることを想定。

    Returns:
    --------
    dist_matrix : ndarray of shape (n, n)
        全点対最短路長
    """
    n = adjacency_matrix.shape[0]
    dist_matrix = adjacency_matrix.copy()

    for k in range(n):
        for i in range(n):
            # dist_matrix[i] の更新のほうが高速実装になるが説明のため3重ループ
            for j in range(n):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
    return dist_matrix


def compute_perplexity(p_row):
    """
    ある i 行に対する p_{j|i} の列ベクトルに対して、
    パープレキシティを計算する。

    Parameters:
    -----------
    p_row : ndarray of shape (n,)
        p_{j|i} を並べたベクトル

    Returns:
    --------
    perp : float
        パープレキシティ (2^(エントロピー))
    """
    # 0 にならないようクリップ
    p = np.clip(p_row, 1e-12, None)
    entropy = -np.sum(p * np.log2(p))
    perp = 2 ** entropy
    return perp


def binary_search_sigma(dist_row, target_perplexity):
    """
    1点 i に対して、与えられた距離ベクトル dist_row をもとに
    p_{j|i} のパープレキシティが target_perplexity になるような
    sigma_i を2分探索で求める。

    Parameters:
    -----------
    dist_row : ndarray of shape (n,)
        点 i から各点 j への距離
    target_perplexity : float
        目標とするパープレキシティ

    Returns:
    --------
    sigma : float
        2分探索で得られた適当な sigma_i
    p_row : ndarray of shape (n,)
        得られた sigma_i で計算した p_{j|i}
    """
    # 2分探索の範囲を適当に設定
    sigma_min = 1e-10
    sigma_max = 1e5
    epsilon = 1e-5

    # i 番目自身への距離は無視 or 0 とするのでマスク
    # dist_row[i] = 0 のはずだが、p_{i|i} = 0 になるので実質無視される

    for _ in range(100):  # 反復回数は適当
        sigma_mid = 0.5 * (sigma_min + sigma_max)

        # p_{j|i} の計算: p_{j|i} = exp(-dist^2 / 2 sigma^2) / 正規化
        # ※ここではオリジナル SNE 論文の式に沿って (distance^2) を用いる
        #  「パス長」を使うのに dist^2 で良いか...？
        # dist_sq = dist_row ** 2
        dist_sq = dist_row
        numerators = np.exp(-dist_sq / (2.0 * (sigma_mid ** 2)))
        numerators[dist_sq == 0] = 0  # 自分自身は確率0にする
        denom = np.sum(numerators)
        if denom < 1e-12:
            # 計算が極端に小さくなる場合は、sigma を増大
            sigma_min = sigma_mid
            continue

        p_row = numerators / denom
        perp = compute_perplexity(p_row)

        if abs(perp - target_perplexity) < epsilon:
            break

        if perp > target_perplexity:
            # パープレキシティが大きい => エントロピーが大きい => 分散を大きくしすぎ
            sigma_max = sigma_mid
        else:
            sigma_min = sigma_mid

    return sigma_mid, p_row


def compute_p_matrix(dist_matrix, perplexity=30.0):
    """
    グラフ上の距離行列 dist_matrix を用いて、
    オリジナル SNE における確率分布 P を計算する (対称化は後で行う)。

    Parameters:
    -----------
    dist_matrix : ndarray of shape (n, n)
        全点対パス長
    perplexity : float
        ターゲットとなるパープレキシティ

    Returns:
    --------
    P : ndarray of shape (n, n)
        p_{j|i} を行列として格納したもの (対角成分は 0)
    """
    n = dist_matrix.shape[0]
    P = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        sigma_i, p_row = binary_search_sigma(dist_matrix[i], perplexity)
        P[i] = p_row

    # 対角は 0 にしておく（ただし既に 0 になっているはず）
    np.fill_diagonal(P, 0.0)
    return P


def symmetrize_p_matrix(P):
    """
    p_{j|i} を対称化し、最終的に p_{ij} を返す。
    p_{ij} = (p_{j|i} + p_{i|j}) / (2n) にする実装例。

    Parameters:
    -----------
    P : ndarray of shape (n, n)
        p_{j|i} をまとめた行列

    Returns:
    --------
    P_sym : ndarray of shape (n, n)
        対称化後の行列
    """
    n = P.shape[0]
    P_sym = (P + P.T) / (2.0 * n)
    return P_sym


def sne(dist_matrix, dim=2, perplexity=30.0, learning_rate=50.0, max_iter=300, seed=42):
    """
    オリジナルの SNE (Stochastic Neighbor Embedding) を実装する。
    距離行列 dist_matrix を用いて P を求め、低次元埋め込み Y を勾配降下で更新する。

    Parameters:
    -----------
    dist_matrix : ndarray of shape (n, n)
        パス長に基づく全点対距離行列
    dim : int
        埋め込みの次元数
    perplexity : float
        パープレキシティ (p_{j|i} 計算用)
    learning_rate : float
        学習率
    max_iter : int
        最大イテレーション回数
    seed : int
        乱数シード

    Returns:
    --------
    Y : ndarray of shape (n, dim)
        低次元に埋め込んだ座標
    """
    np.random.seed(seed)

    n = dist_matrix.shape[0]
    # 確率行列 P を計算
    P_cond = compute_p_matrix(dist_matrix, perplexity=perplexity)
    P = symmetrize_p_matrix(P_cond)

    # 初期埋め込みをランダムに生成
    Y = np.random.randn(n, dim) * 1e-4

    for iteration in range(max_iter):
        # Q の計算 (対称化)
        # 距離の二乗 ||y_i - y_j||^2
        y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # shape (n, n, dim)
        dist_sq_y = np.sum(y_diff ** 2, axis=2)  # shape (n, n)
        # q_ij (非正規化) = exp(-dist_sq_y)
        # i == j の場合は 0 にしておく
        np.fill_diagonal(dist_sq_y, np.inf)
        num_Q = np.exp(-dist_sq_y)
        # ゼロ割り回避
        denom_Q = np.sum(num_Q)
        Q = num_Q / (denom_Q + 1e-12)

        PQ_diff = (P - Q)  # shape (n, n)
        # グラディエントを初期化
        grad = np.zeros_like(Y)

        # (y_i - y_j) をまとめて計算するための一例
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # 係数(スカラー)
                coeff = (P[i, j] - Q[i, j]) * num_Q[i, j] * (-2.0)
                grad[i] += coeff * (Y[i] - Y[j])

        # 勾配降下法で Y を更新
        Y = Y - learning_rate * grad

        # 進捗表示 (任意)
        if (iteration + 1) % 50 == 0:
            PQ_ratio = np.clip(P / np.clip(Q, 1e-12, None), 1e-12, None)
            cost = np.sum(P * np.log(PQ_ratio))
            print(f"Iteration {iteration + 1}, KL divergence = {cost:.4f}")

    return Y


def main():
    np.random.seed(0)
    n = 10
    points = np.random.randn(n, 2)
    dists = pdist(points)
    mean_nn_dist = np.mean(np.sort(dists)[:n])  # 適当
    alpha = 1.5 * mean_nn_dist

    adjacency_matrix = generate_alpha_complex_1_skeleton(points, alpha)

    dist_matrix = floyd_warshall_distances(adjacency_matrix)

    Y = sne(dist_matrix, dim=2, perplexity=5.0, learning_rate=10.0, max_iter=200)

    print("Embedded coordinates (Y):")
    print(Y)


if __name__ == "__main__":
    main()
