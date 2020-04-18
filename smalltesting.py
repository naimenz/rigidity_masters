from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

import sys

# initialise the seed for reproducibility 
np.random.seed(100)

# fw = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])
fw = make_nice_fw(80,1)

fw.add_nodes_from([80])
fw.nodes[80]["position"] = (-6,6)
fw1 = fw.copy()
fw.add_edge(79,80)
fw.add_edge(78,80)
fw = add_lengths_and_stiffs(fw)
# fw.edges[(79,80)]["lam"] = 0 
draw_framework(fw, ghost=True)
draw_framework(fw1, ghost=True)

tensions = [0] *len(fw.edges)
tensions[0] = 1
tensions1 = [0] *len(fw1.edges)
tensions1[0] = 1

# print("Forces:",forces(fw, tensions))
# forces = [-1, 0, 1, 0, 0, 0, 0, 0]

np.set_printoptions(suppress=True)
u = displacements(fw, tensions)
u1 = displacements(fw1, tensions1)
print("WITH GHOST\n",np.around(u,2),"WITHOUT GHOST\n",np.around(u1,2))

print("DIFFERENCE:",np.around(u-u1, 2).astype(np.float32))
# strs = strains(fw1, tensions)
# print("Displacements WITH edge:\n",u1)
# print("Displacements WITHOUT edge:\n",u)

# print("STRAINS with ghost edge:",np.around(strs,2))

# fw.edges[(0,1)]["lam"] = 0
# fw.remove_edge(0,1)


# tensions = [0] *len(fw.edges)
# tensions[0] = 1

# print("Forces:",forces(fw, tensions))
# # fs = [-1, 0, 1, 0, 0, 0, 0, 0]

# u = displacements(fw, None, fs=fs) 

# fw1 = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])
