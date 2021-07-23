import numpy as np

W = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1/4, 1/4, 0, 1/2],
    [1/5, 2/5, 2/5, 0]]
)

# No D, W is already normalized
# Assumption is v_0 = 1, v_1 = 0

v = -np.linalg.inv(W - np.eye(len(W)))[:,0]
