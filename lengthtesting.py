from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import scipy 


positions = np.array([(0,0), (1,0), (1,1), (0,1)])
fw = create_framework([0,1,2,3], [(0,1), (1,2), (2,3), (0,3)], positions )

tension = np.array([0] * len(fw.edges))
tension[1] = 1
disps = np.array([-1,-1, 0, 0, 0, 0, 0, 0])

exts = extensions(fw, None, disps)
# edge_list = list(fw.edges)
# for i in range(len(edge_list)):
    # print(fw.edges[edge_list[i]]["length"])
    # exts[i] = exts[i] / fw.edges[edge_list[i]]["length"]
print(strains(fw, None, exts))
draw_framework(fw)

