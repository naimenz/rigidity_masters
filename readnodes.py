# ============================================================================== 
# LOAD PAPER GRAPH
# ============================================================================== 
from PIL import Image
from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

# def load_image( infilename ) :
#     img = Image.open( infilename )
#     img.load()
#     data = np.asarray( img, dtype="int32" )
#     return data

# im = load_image("bluedottedgraph.png")

# blue = np.array([0,0,255])
# ys, xs = np.array(np.where(np.all(im == np.array(blue), axis=-1)))
# ys = 1000 - ys

# positions = np.array([xs, ys]).T
# # np.savetxt("nodes.csv",positions,delimiter=",")

# nodes = list(range(len(positions)))
# edges = np.loadtxt("edges.csv",dtype=np.int, delimiter=',')
# # fw.add_edges_from(edges)
# fw = create_framework(nodes, edges, positions)
# source = (187,188)
# target = (0,1)
# nstars = [1.0]
# fw.add_edges_from([source, target])
# fw = add_lengths_and_stiffs(fw)
# fw.edges[source]["lam"] = 0
# fw.edges[target]["lam"] = 0

# ============================================================================== 
# END LOAD PAPER GRAPH
# ============================================================================== 

graphfile = "graph.log"
logfile = "images/2020-04-29/1588158646/log.log"

disps_list = read_disps(logfile)

fw, source, target, nstars = read_graphlog(graphfile)
plot_ratios_list(ratios_list_from_disps(fw, disps_list, source, target), nstars)
