import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

def generate_object(center, cov, n_points):
    return np.random.multivariate_normal(center, cov, n_points)

# ---- 3D点群 ----
points = np.vstack([
    generate_object([2, 2, 2],
                    [[0.1, 0, 0],
                     [0, 0.1, 0],
                     [0, 0, 0.1]], 150),

    generate_object([5, 3, 1],
                    [[0.2, 0.05, 0],
                     [0.05, 0.1, 0],
                     [0, 0, 0.1]], 150),

    generate_object([7, 7, 3],
                    [[0.1, -0.03, 0],
                     [-0.03, 0.2, 0],
                     [0, 0, 0.1]], 150)
])

N, dimension = points.shape   # dimension = 3
object = 100

#EM法
mu = points[np.random.choice(N, object, replace=False)]
Sigma = np.array([np.eye(dimension) for _ in range(object)])
pi = np.ones(object) / object

def gaussian(x, mu, Sigma):
    d = x.shape[1]
    diff = x - mu
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    norm = 1.0 / np.sqrt((2 * np.pi)**d * det)
    return norm * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))

for _ in range(100):
    # E-step
    gamma = np.zeros((N, object))
    for k in range(object):
        gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-step
    Nk = gamma.sum(axis=0)
    for k in range(object):
        mu[k] = np.sum(gamma[:, k, None] * points, axis=0) / Nk[k]
        diff = points - mu[k]
        Sigma[k] = (gamma[:, k, None, None] *
                    np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Nk[k]
        pi[k] = Nk[k] / N

    threshold = 0.01  # 全体の1%未満は削除
    valid = pi > threshold
    mu = mu[valid]
    Sigma = Sigma[valid]
    pi = pi[valid]

    # 正規化（合計を1に戻す）
    pi = pi / pi.sum()

    object = len(pi)  # クラスタ数を更新

labels = np.argmax(gamma, axis=1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

for k in range(object):
    cluster = points[labels == k]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
               s=10, label=f"Object {k+1}")

    ax.scatter(mu[k][0], mu[k][1], mu[k][2],
               c='black', marker='x', s=100)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Object Detection using EM (GMM)")
ax.legend()
plt.show()
