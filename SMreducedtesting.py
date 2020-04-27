from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize
import time
import os

# initialise the seed for reproducibility 
np.random.seed(110)
fileroot = "images/anim"+str(int(time.time())) + "/"

fw = make_nice_fw(200, 1)
flag, comps = pebble_game(fw)
draw_comps(fw, comps)
print(len(fw.nodes), len(fw.edges))

# returns a tuned network
source = (168, 171)
target = (192,199)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)
# test of creating a dictionary to keep track of edges
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

# tic = time.perf_counter()
# fw = SM_tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
# toc = time.perf_counter()
# print(f"SM took {toc - tic:0.4f} seconds")

tic = time.perf_counter()
fw = tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
toc = time.perf_counter()
print(f"brute force took {toc - tic:0.4f} seconds")

tensions = [0]*len(fw.edges)
tensions[edge_dict[source]] = 1
print("AFTER TUNING:\n===============")
# exts = extensions(fw, tensions)
# strains = exts_to_strains(fw, exts)
strs = strains(fw,tensions)
print("strains on source, target resp.",strs[edge_dict[source]], strs[edge_dict[target]])
draw_strains(fw, strs, source, target, ghost=True)

os.mkdir(fileroot)
ratios = animate(fw, source, target, fileroot, nstars, s_max=1, tensions=1)
