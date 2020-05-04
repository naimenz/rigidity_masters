from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize
import datetime
import time
import os

# initialise the seed for reproducibility 
np.random.seed(505)
# fileroot = "images/"+str(datetime.date.today()) +"/"+ str(int(time.time())) + "/"

fw = make_nice_fw(200, 1e-2)
flag, comps = pebble_game(fw)
draw_comps(fw, comps)

# returns a tuned network
source = (152,153)
target = (190,194)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)
# test of creating a dictionary to keep track of edges
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

fw = SM_tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, cost_thresh=0.01, verbose=True)


# fw = GF_tune_network(fw, source, target, tension=1, nstars=nstars)
# print(np.linalg.svd(rig_mat(fw)))
tensions = [0]*len(fw.edges)
tensions[edge_dict[source]] = 1
# trying tension on neighbouring bonds
# tensions[edge_dict[(195,199)]] = 1
print("AFTER TUNING:\n===============")
# exts = extensions(fw, tensions)
# strains = exts_to_strains(fw, exts)
# fw.edges[(18,22)]["lam"] = 0
strs = strains(fw,tensions)
print("strains on source, target resp.",strs[edge_dict[source]], strs[edge_dict[target]])
draw_strains(fw, strs, source, target, ghost=True)

ratios = animate(fw, source, target, nstars, s_max=1, tensions=1)
