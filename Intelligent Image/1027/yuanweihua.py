# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月27日14时28分56秒
# 下午2:28
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
x_test_std = scaler.transform(X_test)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(x_test_std)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classifcation Report:\n", classification_report(y_test, y_pred, target_names=[num_to_label[i] for i in range(3)]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

x_new = np.array([[5.8, 2.8, 5.1, 2.4]])
x_new_std = scaler.transform(x_new)
pred_label = num_to_label[knn.predict(x_new_std)[0]]
print("Prediction:", pred_label)