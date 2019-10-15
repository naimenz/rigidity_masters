import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

points = np.array([
                [1,2],
                [2,3],
                [1,1],
                [4,6],
                [2,3]
                ])

tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.show()
