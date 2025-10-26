# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月18日22时38分58秒
# 下午10:38

import numpy as np
import struct
import os
import matplotlib.pyplot as plt

# configs
MNIST_DIR = r"D:\PyStudy\day6\data\MNIST\raw"
alpha = 0.001
epochs = 1000
batch_size = 128
hidden_size = 128
input_size = 784  # MNIST图像28x28=784
output_size = 10  # 数字0-9对应10个神经元

# Adam参数 一阶矩和二阶矩的衰减率，以及防止除0的eps
beta1, beta2, eps = 0.9, 0.999, 1e-8  # 通常设为0.9和0.999和1e-8

# 初始化权重
theta1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
theta2 = np.zeros((1, hidden_size))
theta3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
theta4 = np.zeros((1, output_size))

# Adam状态
m = {k: np.zeros_like(v) for k, v in zip(["theta1", "theta2", "theta3", "theta4"], [theta1, theta2, theta3, theta4])}
v = {k: np.zeros_like(v) for k, v in zip(["theta1", "theta2", "theta3", "theta4"], [theta1, theta2, theta3, theta4])}
mid = None


# 加载数据（MNIUS的数据里已经下载到本地了，所以直接通过struct从本地加载）
def load_idx_images_labels(img_path, lbl_path):
    with open(img_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with open(lbl_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def prepare_dataset():
    train_img = os.path.join(MNIST_DIR, "train-images-idx3-ubyte")
    train_lbl = os.path.join(MNIST_DIR, "train-labels-idx1-ubyte")
    test_img = os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte")
    test_lbl = os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte")

    X_train, y_train = load_idx_images_labels(train_img, train_lbl)
    X_test, y_test = load_idx_images_labels(test_img, test_lbl)
    # 数据通过astype函数转换为浮点数，并归一化
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    # y_train 和 y_test 转换为独热编码，解决多分类问题
    return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test


def activation_f(x):
    return np.maximum(0, x)  # ReLU激活函数，将负值置为0，正值保持不变


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 前向传播模型
def model(x):
    global mid
    z1 = np.dot(x, theta1) + theta2
    mid = activation_f(z1)
    z2 = np.dot(mid, theta3) + theta4
    y = softmax(z2)
    return y


# 损失函数
def loss_function(y_true, y_pred):
    eps = 1e-12
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + eps)) / N
    return loss


# 更新参数
def update(y_pred, y_true, x, epoch, optimizer="sgd"):
    global theta1, theta2, theta3, theta4, m, v

    N = x.shape[0]
    dz2 = (y_pred - y_true) / N
    dtheta3 = np.dot(mid.T, dz2)
    dtheta4 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, theta3.T)
    dz1 = da1 * (mid > 0)
    dtheta1 = np.dot(x.T, dz1)
    dtheta2 = np.sum(dz1, axis=0, keepdims=True)
    # 这边先传入sgd，再跑adam（因为要做对比）
    if optimizer == "sgd":
        theta1 -= alpha * dtheta1
        theta2 -= alpha * dtheta2
        theta3 -= alpha * dtheta3
        theta4 -= alpha * dtheta4
    elif optimizer == "adam":
        t = epoch + 1
        for name, grad in zip(
                ["theta1", "theta2", "theta3", "theta4"],
                [dtheta1, dtheta2, dtheta3, dtheta4],
        ):
            m[name] = beta1 * m[name] + (1 - beta1) * grad
            v[name] = beta2 * v[name] + (1 - beta2) * (grad * grad)
            m_hat = m[name] / (1 - beta1 ** t)
            v_hat = v[name] / (1 - beta2 ** t)
            update_val = alpha * m_hat / (np.sqrt(v_hat) + eps)
            globals()[name] -= update_val


def accuracy(y_pred, y_true_onehot):
    pred_label = np.argmax(y_pred, axis=1)
    true_label = np.argmax(y_true_onehot, axis=1)
    return np.mean(pred_label == true_label)


def main(opt_name="sgd"):
    print(f"正在训练（优化器={opt_name}）...")
    X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test = prepare_dataset()
    loss_list, acc_list, loss_variation = [], [], []

    # 训练
    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_train, y_train_onehot = X_train[idx], y_train_onehot[idx]

        batch_losses, batch_accs = [], []
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train_onehot[i:i + batch_size]
            y_pred = model(x_batch)  # 模型预测
            loss_val = loss_function(y_batch, y_pred)  # 计算损失
            update(y_pred, y_batch, x_batch, epoch, optimizer=opt_name)  # 更新参数
            batch_losses.append(loss_val)
            batch_accs.append(accuracy(y_pred, y_batch))

        loss_list.append(np.mean(batch_losses))
        acc_list.append(np.mean(batch_accs))
        loss_variation.append(batch_losses[-1])

        print(f"Epoch [{epoch + 1}/{epochs}] | Loss={loss_list[-1]:.4f} | Acc={acc_list[-1]:.4f}")

    y_test_pred = model(X_test)
    test_acc = accuracy(y_test_pred, y_test_onehot)
    print(f"最终测试集平均准确率：{test_acc:.4f}")

    return loss_list, acc_list, loss_variation


if __name__ == "__main__":
    np.random.seed(42)

    # SGD
    loss_sgd, acc_sgd, variation_sgd = main(opt_name="sgd")

    # 重置参数
    theta1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
    theta2 = np.zeros((1, hidden_size))
    theta3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
    theta4 = np.zeros((1, output_size))
    m = {k: np.zeros_like(v) for k, v in m.items()}
    v = {k: np.zeros_like(v) for k, v in v.items()}

    # Adam
    loss_adam, acc_adam, variation_adam = main(opt_name="adam")

    # 绘制损失函数平均变化图、通过率变化图、损失函数变化图
    plt.figure(figsize=(12, 4))

    # 平均 Loss
    plt.subplot(1, 3, 1)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.plot(loss_sgd, label="SGD loss")
    plt.plot(loss_adam, label="Adam loss")
    plt.title("Loss Comparison (mean)")
    plt.legend();
    plt.grid(True, alpha=0.3)

    # 中：Accuracy
    plt.subplot(1, 3, 2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(acc_sgd, label="SGD train acc")
    plt.plot(acc_adam, label="Adam train acc")
    plt.title("Accuracy Comparison")
    plt.legend();
    plt.grid(True, alpha=0.3)

    # Loss Variation(Last batch loss in per epoch)
    plt.subplot(1, 3, 3)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.plot(variation_sgd, label="SGD loss variation", alpha=0.8)
    plt.plot(variation_adam, label="Adam loss variation", alpha=0.8)
    plt.title("Loss Comparison(Last batch per epoch)")
    plt.legend();
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
