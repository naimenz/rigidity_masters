from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

# initialise the seed for reproducibility 
np.random.seed(102)

fw = create_random_fw(50, 1)
# test of creating a dictionary to keep track of edges
edge_legend = {edge: i for i, edge in enumerate(fw.edges)}

# modifying the framework to change two bonds to ghost bonds (target and source)
source = (46,49)
target = (34,44)
fw.edges[source]["lam"] = 0
fw.edges[target]["lam"] = 0


tensions = [0]*len(fw.edges)
tensions[edge_legend[source]] = 10

stresses = exts_to_stresses(fw, extensions(fw, tensions))
draw_stresses(fw, stresses, ghost=True, filename="tuned_stress.png")

# aiming for proportional movement of (16,17) when (9,12) moves
nstars = [1.0]

# calculating ns test
it = 0
min_cost = np.inf
while min_cost > 0.01 and it < 10000:
    costs = []
    exts_list = all_extensions(fw, tensions)
    # for exts in exts_list:
    #     stresses = exts_to_stresses(fw, exts)
    #     ns = [stresses[edge_legend[(16,17)]] / stresses[edge_legend[(9,12)]]]
    #     costs.append(cost_f(ns, nstars))
    for i, exts in enumerate(exts_list):
        stresses = exts_to_stresses(fw, exts)
        ns = [stresses[edge_legend[target]] / stresses[edge_legend[source]]]
        costs.append(cost_f(ns, nstars))
    min_cost = min(costs)
    index_to_remove = costs.index(min_cost)
    edge_to_remove = list(fw.edges)[index_to_remove]
    fw.edges[edge_to_remove]["lam"] = 0
    print("iteration:",it,"cost:",min_cost,"removed:",edge_to_remove)
    it+=1

# draw_framework(fw,ghost=True)

stresses = exts_to_stresses(fw, extensions(fw, tensions))
draw_stresses(fw, stresses, ghost=True, filename="tuned_stress.png")

red_fw = fw.copy()
dead_edges = [edge for edge in fw.edges if fw.edges[edge]["lam"] == 0]
red_fw.remove_edges_from(dead_edges)

res, comps = pebble_game(red_fw)
draw_comps(red_fw, comps)

draw_framework(red_fw)
