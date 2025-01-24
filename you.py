import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
wx = np.linspace(-20, 20, 100)
wy = 2 * wx + 13 + np.random.normal(0, 2, (100, 1))

plt.scatter(wx, wy)
plt.show()