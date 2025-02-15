import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm mất mát (Loss Function)
def loss_function(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# Gradient Descent
def gradient_descent(X, y, learning_rate, n_iterations):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    y = y.reshape((-1, 1))
    for _ in range(n_iterations):
        z = np.dot(X, w)
        p = sigmoid(z)
        gradient = np.dot(X.T, (p - y)) / n_samples
        w -= learning_rate * gradient
    return w

# Hàm dự đoán
def predict(X, w):
    p = sigmoid(np.dot(X, w))
    return (p > 0.5).astype(int)

# Load dữ liệu và tiền xử lý
data = pd.read_csv('./stroke.csv')

# Chọn đặc trưng và biến mục tiêu
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']].values
y = data['stroke'].values  # Cột dự đoán là "stroke"

# Tách tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
w = gradient_descent(X_train, y_train, learning_rate=0.01, n_iterations=1000)
print(w)
# Dự đoán và đánh giá mô hình
y_pred = predict(X_test, w)

# Tính toán và in ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
print("=== Confusion Matrix ===")
print(f"   a   b   <-- classified as")
print(f"{conf_matrix[0, 0]:>4} {conf_matrix[0, 1]:>4} |   a = 0")
print(f"{conf_matrix[1, 0]:>4} {conf_matrix[1, 1]:>4} |   b = 1")

# Tính độ chính xác
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("Accuracy score:", accuracy(y_test, y_pred))

# Vẽ đồ thị ranh giới quyết định
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 1], X_test[:, 7], c=y_test, cmap='Reds_r', marker='o', label='Actual')
plt.scatter(X_test[:, 1], X_test[:, 7], c=y_pred, cmap='Blues_r', marker='x', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Avg Glucose Level')
plt.title('Decision Boundary')
plt.legend()
plt.show()