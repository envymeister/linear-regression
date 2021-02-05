import numpy as np
from matplotlib import pyplot as plt

# y = О0 * x0 + О1 * x1 + О2 * x2

def compute_hypothesis(X, theta):
    return X @ theta 


def compute_cost(X, y, theta):
    m = X.shape[0]  # m - количество строк, n - количество параметров
    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)


def gradient_descend(X, y, theta, alpha, num_iter): # theta - вектор коэффициентов
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным
    for i in range(num_iter):
        theta_temp = theta
        for i in range(n):
            theta_temp[i] = theta_temp[i] - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, i]) / m

        theta = theta_temp
        history.append(compute_cost(X, y, theta))
    return history, theta


def scale_features(X):
    for i in range(1,3):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        X[:,i] = (X[:,i] - mean)/std
    
    X[:,0] = 1
    print(X)
    return X

def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data('data.txt')
history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации до нормализации')
plt.plot(range(len(history)), history)
plt.show()

X = scale_features(X)

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации после нормализации')
plt.plot(range(len(history)), history)
plt.show()

theta_solution = normal_equation(X, y)
print(f'theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}')
