import numpy as np
import scipy.linalg


A = np.random.random((3,3))
AAt = A @ A.T
print(A, AAt)

# print(scipy.linalg.orth(A))
# print(scipy.linalg.orth(AAt))

