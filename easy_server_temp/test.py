import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [1.5, 20, 7],
    [14.5, 20, 5],
    [15, 20, 9.5],
    [4.5, 20, 1.5],
    [18.5, 20, 16],
    [1.5, 20, 10.5],
    [10.5, 20, 8.5],
    [10.5, 20, 9],
    [7, 20, 15.5],
    [1.5, 20, 18.5],
    [13.5, 20, 3],
    [10.5, 20, 10.5],
    [6.5, 20, 12],
    [5.5, 20, 3.5]
])

sub_data = data[data[:, 2] > 7.5]
sub_data = sub_data[sub_data[:, 2] < 10.0]
sub_data = sub_data[sub_data[:, 0] > 10.0]

# print(sub_data)

# plt.scatter(data[:, 0], data[:, 2])
# plt.show()

a = np.array([18.5, 20, 2.5])
b = np.array([18, 20, 3])
c = np.linalg.norm(a - b)
print(c)