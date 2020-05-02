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

fwc = SM_tune_network(fw, source, target) 
fw = GF_tune_network(fw, source, target)
