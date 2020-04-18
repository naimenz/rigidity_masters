from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize

# initialise the seed for reproducibility 
np.random.seed(103)

fw = make_nice_fw(300, 1)
flag, comps = pebble_game(fw)
draw_comps(fw, comps)

# returns a tuned network
source = (5,15)
target = (7,8)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)
# test of creating a dictionary to keep track of edges
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

# COPYING BEFORE TUNING
fwc = fw.copy()


import time
tic = time.perf_counter()
fw_sm = SM_tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
toc = time.perf_counter()
print(f"SM took {toc - tic:0.4f} seconds")

tic = time.perf_counter()
fw_bf = tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
toc = time.perf_counter()
print(f"brute force took {toc - tic:0.4f} seconds")

# tensions = [0]*len(fw.edges)
# tensions[edge_dict[source]] = 1
# print("AFTER TUNING:\n===============")
# # exts = extensions(fw, tensions)
# # strains = exts_to_strains(fw, exts)
# strs = strains(fw,tensions)
# print("strains on source, target resp.",strs[edge_dict[source]], strs[edge_dict[target]])
# draw_strains(fw, strs, source, target, ghost=True)

# ratios = animate(fw, source, target, "images/anim11/", nstars, s_max=1, tensions=1)
