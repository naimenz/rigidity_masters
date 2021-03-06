from PIL import Image
from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

im = load_image("bluedottedgraph.png")

blue = np.array([0,0,255])
ys, xs = np.array(np.where(np.all(im == np.array(blue), axis=-1)))
ys = 1000 - ys

positions = np.array([xs/1000, ys/1000]).T
# np.savetxt("nodes.csv",positions,delimiter=",")

nodes = list(range(len(positions)))
edges = np.loadtxt("edges.csv",dtype=np.int, delimiter=',')
# fw.add_edges_from(edges)
fw = create_framework(nodes, edges, positions)
draw_framework(fw, ghost=True)


# returns a tuned network
source = (187,188)
target = (0,1)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)


# import time
# tic = time.perf_counter()
# fwc = SM_tune_network(fw, source, target, tension=1, nstars=nstars)
# toc = time.perf_counter()
# print(f"SM took {toc - tic:0.4f} seconds")

import time
tic = time.perf_counter()
fw = GF_tune_network(fw, source, target, tension=1, nstars=nstars)
toc = time.perf_counter()
print(f"GF took {toc - tic:0.4f} seconds")

red_edges = [(1,5),(9,18),(37,46),(54,69),(77,84),(174,173), (0,1), (187,188)]
# for edge in red_edges:
#     fw.edges[edge]["lam"] = 0 
draw_framework(fw, ghost=True)
tensions = [0]*len(fw.edges)
edge_dict = get_edge_dict(fw)
tensions[edge_dict[source]] = 1
strs = strains(fw, tensions)
print("target, source strains:",strs[edge_dict[target]], strs[edge_dict[source]])

# ratios = animate(fw, source, target, "images/scale_R/", nstars, s_max=1, tensions=1)
