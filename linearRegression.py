import numpy as np
import matplotlib.pyplot as plt
NUM_POINTS = 1000
vectors = []
for i in range(NUM_POINTS):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1 * 0.1 + 0.3 * np.random.normal(0.0,0.05)
    vectors.append([x1,y1])
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

plt.plot(x_data,y_data,'ro',label='original data')
plt.legend()
plt.show()


