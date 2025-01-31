import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import gudhi
from sklearn.manifold import Isomap
# スイスロールの生成
X, color = make_swiss_roll(n_samples=1000, noise=0.1)

# アルファ複体の構築とパーシステント図の計算
alpha_complex = gudhi.AlphaComplex(points=X)
st = alpha_complex.create_simplex_tree()
st.compute_persistence()

# 近傍半径をパーシステント図 (H0) から推定
d0_intervals = st.persistence_intervals_in_dimension(0)

# deathが無限大のバーは無視し，有限deathだけ取り出す
finite_d0 = d0_intervals[np.isfinite(d0_intervals[:, 1])]

# その中で最大のdeath値を閾値とする
radius = np.max(finite_d0[:, 1])

print("推定した近傍半径:", radius)

# ちょうど二つの円が交わるところなはずなので一つの円では2倍の半径がいるはず
isomap = Isomap(radius=radius*2, n_neighbors=None)
X_2d = isomap.fit_transform(X)

# 結果の可視化
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color)
plt.colorbar(label="color")
plt.title("Isomap Embedding")
plt.show()