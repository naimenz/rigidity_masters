from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

# initialise the seed for reproducibility 
np.random.seed(102)

# fw = make_nice_fw(20, 1)
# # fw = create_reduced_fw(20, 1)
# flag, comps = pebble_game(fw)
# draw_comps(fw, comps)


start = 0
stop = 16
xs1 = np.arange(start, stop, 1.0)
ys1 = np.arange(start, stop, 2.0)
ys2 = np.arange(start+1,stop, 2.0)
xs2 = np.arange(start+0.5, stop+0.5, 1.0)
xy1 = np.array(np.meshgrid(xs1, ys1)).T
xy2 = np.array(np.meshgrid(xs2, ys2)).T
positions = np.concatenate((xy1,xy2)).reshape((-1,2))
nodes = list(range(len(positions)))
edges = delaunay_to_edges(Delaunay(positions))
fw = create_framework(nodes, edges, positions)
draw_framework(fw)

source = (51, 170)
target = (85, 213)
import time
tic = time.perf_counter()
fw = SM_tune_network(fw, source, target, cost_thresh=0.001)
toc = time.perf_counter()
print(f"SM tuning took {toc-tic:0.4f} seconds")

ratios = animate(fw, source, target, "images/anim2/", [1.0])

