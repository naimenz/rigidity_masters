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
np.random.seed(103)
fileroot = "images/"+str(datetime.date.today()) +"/"+ str(int(time.time())) + "/"

fw = make_nice_fw(40, 1)
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
R = rig_mat(fw)
R_bar = Qbar_mat(fw).T
print("Singular values of R:\n",np.linalg.svd(R, compute_uv=False))
print("Singular values of R_bar:\n",np.linalg.svd(R_bar, compute_uv=False))

fw = SM_tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, cost_thresh=0.000001, verbose=True)

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

R = rig_mat(fw)
R_bar = Qbar_mat(fw).T

print("Singular values of R:\n",np.linalg.svd(R, compute_uv=False))
print("Singular values of R_bar:\n",np.linalg.svd(R_bar, compute_uv=False))

os.makedirs(fileroot)
ratios = animate(fw, source, target, fileroot, nstars, s_max=1, tensions=1)
