from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

# initialise the seed for reproducibility 
np.random.seed(102)

triangle = create_framework([0,1,2], [(0,1), (0,2), (1,2)], [(1,0), (3,0), (2,2)])
# draw_framework(triangle)
f = np.zeros(2*len(triangle.nodes))
f[0] = 1
f[2] = -1

Rold = create_rigidity_matrix(triangle,2)
R = create_augmented_rigidity_matrix(triangle,2)
print(R)
print("NEW:",np.array([-1/2, 0, 0,0,0,0]).dot(R))
print("OLD:",np.array([-1/2, 0, 0]).dot(Rold))
print(R.dot(R.T))

# draw_stresses(triangle, f)

# shape = create_framework([0,1,2,3,4], [(0,1), (0,2), (1,2),(3,4), (2,3), (2,4),(1,4)], [(1,0), (3,0), (2,2),(1,3),(3,3)])
# draw_framework(shape)
# f = np.zeros(2*len(shape.nodes))
# f[0] = 0
# f[1] = 0
# f[2] = 3
# f[4] = -1
# f[5] = 1.5

# draw_stresses(shape, f)

square = create_framework([0,1,2,3], [(0,1), (0,2), (3,2),(3,1)], [(1,0), (3,0), (1,3),(3,3)])
draw_framework(square)
f = np.zeros(2*len(square.nodes))
f[0] = 0.1
f[1] = 0.1
f[2] = -0.1
f[3] = -0.1

# f[4] = 
# f[5] = 1.5

draw_stresses(square, f)
