import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import find_peaks

def gauss(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

x_values = np.arange(0.1, 1 + 1/22, 1/22)

y_original = (1 + 0.6 * np.sin(2 * np.pi * x_values / 0.7) + 0.3 * np.sin(2 * np.pi * x_values)) / 2

peaks, _ = find_peaks(y_original)
peak_x = x_values[peaks]
peak_y = y_original[peaks]

w1 = random.uniform(0, 1)
w2 = random.uniform(0, 1)
w0 = random.uniform(0, 1)

learning_rate = 0.01
print(peak_x)

# Example usage
c1 = peak_x[0]  # center
r1 = 0.15  # radius
c2 = peak_x[1]  # center
r2 = 0.185  # radius

# y_gauss1 = peak_y[0]*gauss(x_values, c1, r1)
# y_gauss2 = peak_y[1]*gauss(x_values, c2, r2)
# plt.plot(x_values, y_original, label='y = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2')
# plt.plot(x_values, y_gauss1, label='Gaussian Function 1')
# plt.plot(x_values, y_gauss2, label='Gaussian Function 2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)
# # plt.legend()
# plt.show()

number_of_epochs = 20000
# Training
for x, y in zip(x_values, y_original):
    y_pred = w0 + w1 * gauss(x,c1,r1) + w2 * gauss(x,c2,r2)
    error = y - y_pred
    for _ in range(number_of_epochs):
        w0 += learning_rate * error
        w1 += learning_rate * error * gauss(x,c1,r1)
        w2 += learning_rate * error * gauss(x,c2,r2)
        y_pred = w0 + w1 * gauss(x,c1,r1) + w2 * gauss(x,c2,r2)
        error = y - y_pred

print(f'w0: {w0}, w1: {w1}, w2: {w2}')

# Prediction
new_x_values = np.arange(0.1, 1 + 1/30, 1/30)

y_pred = w0 + w1 * gauss(new_x_values,c1,r1) + w2 * gauss(new_x_values,c2,r2)
y_original = (1 + 0.6 * np.sin(2 * np.pi * new_x_values / 0.7) + 0.3 * np.sin(2 * np.pi * new_x_values)) / 2
plt.plot(new_x_values, y_original, label='Original y')
plt.plot(new_x_values, y_pred, label='Predicted y')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

