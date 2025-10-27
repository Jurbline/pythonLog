# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月27日14时28分56秒
# 下午2:28

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from collections import Counter

# 读取数据
path = r'.\iris.csv'
iris = pd.read_csv(path)

x = iris.iloc[:, 1:-1].values
y_raw = iris.iloc[:, -1].values

label_to_num = {"setosa": 0,
                "versicolor": 1,
                "virginica": 2
                }
num_to_label = {v: k for k, v in label_to_num.items()}
y = np.array([label_to_num[label] for label in y_raw])

# 标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(x)


# 手动实现KNN
def knn_predict(X_train, y_train, X_test, k):
    tree = KDTree(X_train)
    y_pred = []
    for x in X_test:
        # 查找k个最近邻
        dist, idx = tree.query(x, k)
        if k == 1:  # k=1返回单个索引的情况
            idx = [idx]
        neighbors = y_train[idx]
        most_common = Counter(neighbors).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)


# 5折交叉验证评估不同的k
def cross_validate_knn(X, y, k_values, n_splits=5):
    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k in k_values:
        acc_list = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_pred = knn_predict(X_train, y_train, X_test, k)
            acc = np.mean(y_pred == y_test)
            acc_list.append(acc)
        results[k] = np.mean(acc_list)
    return results

# 不同K值
k_values = [1, 3, 5, 7, 9, 11]
results_original = cross_validate_knn(X_std, y, k_values)
print("Origin Visualization 5-fold cross-validation accuracy")
for k, acc in results_original.items():
    print(f"k={k:<2} | Accuracy={acc:.4f}")

# PCA降维到2D
pca_2d = PCA(n_components=2)
X_pca2 = pca_2d.fit_transform(X_std)
results_pca2 = cross_validate_knn(X_pca2, y, k_values)
print("\nPCA 2D Visualization 5-fold cross-validation accuracy")
for k, acc in results_pca2.items():
    print(f"k={k:<2} | Accuracy={acc:.4f}")

# 绘制PCA 2D散点图
plt.figure(figsize=(7, 6))
for label in np.unique(y):
    plt.scatter(X_pca2[y == label, 0], X_pca2[y == label, 1],
                label=num_to_label[label], alpha=0.7)
plt.title("PCA 2D Visualization of Iris Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# PCA降维到3D
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_std)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    ax.scatter(X_pca3[y == label, 0], X_pca3[y == label, 1], X_pca3[y == label, 2],
               label=num_to_label[label], alpha=0.7)
ax.set_title("PCA 3D Visualization of Iris Dataset")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()

# 对比结果可视化
plt.figure(figsize=(8, 5))
plt.plot(k_values, [results_original[k] for k in k_values], 'o-', label='Original Features')
plt.plot(k_values, [results_pca2[k] for k in k_values], 's-', label='PCA 2D Features')
plt.title("KNN Accuracy Comparison (5-fold CV)")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
