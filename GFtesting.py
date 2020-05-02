from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import scipy 

import sys

# initialise the seed for reproducibility 
np.random.seed(100)

Hfw = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])
fw = make_nice_fw(50,1)
# add_edges = [(17,18), (1,10), (10,21)]
# fw.add_edges_from(add_edges)
# fw = add_lengths_and_stiffs(fw)
# for edge in add_edges:
#     fw.edges[edge]["lam"] = 0

# draw_framework(fw,ghost=True)

source = (39,40)
target = (31,42)

     return fw, min_cost, edge_to_remove

fwc = SM_tune_network(fw, source, target, cost_thresh=0.0001)
min_cost = np.inf
medges = []
it = 0
while min_cost > 0.00001:
    it += 1
    print("============== ITERATION",it, "===============")
    fw, min_cost, medge = GF_one_step(fw, source, target, 1, [1.0])
    medges.append(medge)



# fw = SM_tune_network(fw, source, target, 1, nstars)
