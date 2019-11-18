from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

# initialise the seed for reproducibility 
np.random.seed(102)

triangle = create_framework([0,1,2,3,4], [(0,1), (0,2), (1,2),(3,4), (2,3), (2,4),(1,4)], [(1,0), (3,0), (2,2),(1,3),(3,3)])
draw_framework(triangle)
f = np.zeros(2*len(triangle.nodes))
f[0] = 1
f[2] = -1

draw_stresses(triangle, f)



