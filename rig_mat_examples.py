from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
# understanding how the rigidity matrix relates tension and extension
# fw = create_framework([0,1,2,3,4], [(0,1), (1,2), (0,2), (2,3), (3,4), (4,2),(3,1)], [(0,0), (2,0), (1,1), (2.5, 1), (2.7,2)])
fw = create_framework([0,1, 2, 3], [(0,1), (2,3)], [(1,2), (2,1), (1, 4), (2,3)])
u = np.array([1,1,-1,-1])
t = [1,0]
draw_framework(fw)

R = rig_mat(fw)
Rt = R.T
print(Rt,t)
print(Rt @ t)

