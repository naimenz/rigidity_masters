from PIL import Image
from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import time
import datetime
import os


fileroot = "images/"+str(datetime.date.today()) +"/"+ str(int(time.time())) + "/"

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

im = load_image("bluedottedgraph.png")

blue = np.array([0,0,255])
ys, xs = np.array(np.where(np.all(im == np.array(blue), axis=-1)))
ys = 1000 - ys

positions = np.array([xs, ys]).T
# np.savetxt("nodes.csv",positions,delimiter=",")

nodes = list(range(len(positions)))
edges = np.loadtxt("edges.csv",dtype=np.int, delimiter=',')
# fw.add_edges_from(edges)
fw = create_framework(nodes, edges, positions)
draw_framework(fw, ghost=True)

# returns a tuned network
source = (0,1)
target = (187,188)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)

SSS, SCS = subbases(fw)
# for edge in fw.edges:
#     Si = calc_Si(fw, edge, SSS)
#     if np.isclose(np.inner(Si, Si), 0):
#         print("Edge is 0:", edge, np.inner(Si,Si))
tensions = np.zeros(len(fw.edges))
edge_dict = get_edge_dict(fw)
tensions[edge_dict[source]] = 2
exts = extensions(fw, tensions)

green = G_f(fw, SCS)

# NOTE: test to see if different displacements affect energy
def scaled_displacements(fw, tstar):
    Qbar = Qbar_mat(fw)
    # F = Fbar_mat(fw)
    # Finv = np.linalg.pinv(F, hermitian=True)
    # H = Qbar @ Finv @ Qbar.T
    # Hinv = np.linalg.pinv(H, hermitian=True)
    Hinv = Hbar_inv_mat(fw)

    u = Hinv @ Qbar @ tstar

    return u

# print(rig_mat)
# R = rig_mat(fw)
# print(np.inner(R[1], R[1]))

# CHECKING DISPLACEMENT STUFF
fw = add_lengths_and_stiffs(fw, 1)
print("bond stiffness:", fw.graph["k"])
tensions = np.zeros(len(fw.edges))
edge_dict = get_edge_dict(fw)
tension = 1
tensions[edge_dict[source]] = tension
print("tension:",tension)
disps = displacements(fw, tensions)
scaled_disps = scaled_displacements(fw, tensions)
H = H_mat(fw)
Hbar = Hbar_mat(fw)
print("Energy unscaled H:", 0.5 * disps @ H @ disps.T)
print("Energy scaled H scaled disps:", 0.5 * scaled_disps @ Hbar @ scaled_disps.T)
print("Energy scaled H :", 0.5 * disps @ Hbar @ disps.T)

#######################################################################################
print("#######################################################################################")
#######################################################################################

fw = add_lengths_and_stiffs(fw, 2)
print("bond stiffness:", fw.graph["k"])

tensions = np.zeros(len(fw.edges))
edge_dict = get_edge_dict(fw)
tension = 1
tensions[edge_dict[source]] = tension
print("tension:",tension)
disps = displacements(fw, tensions)
scaled_disps = scaled_displacements(fw, tensions)
H = H_mat(fw)
Hbar = Hbar_mat(fw)
print("Energy unscaled H:", 0.5 * disps @ H @ disps.T)
print("Energy scaled H scaled disps:", 0.5 * scaled_disps @ Hbar @ scaled_disps.T)
print("Energy scaled H :", 0.5 * disps @ Hbar @ disps.T)

# import time
# tic = time.perf_counter()
# fw = tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
# toc = time.perf_counter()
# print(f"BF took {toc - tic:0.4f} seconds")

# import time
# tic = time.perf_counter()
# fw = SM_tune_network(fw, source, target, tension=1, nstars=nstars, draw=True, verbose=True)
# toc = time.perf_counter()
# print(f"SM took {toc - tic:0.4f} seconds")

# tensions = np.zeros(len(fw.edges))
# edge_dict = get_edge_dict(fw)
# tensions[edge_dict[source]] = 1
# strs = strains(fw, tensions)
# print("source strain, target strain", strs[edge_dict[source]], strs[edge_dict[target]])

# # red_edges = [(1,5),(9,18),(37,46),(54,69),(77,84),(174,173), (0,1), (187,188)]
# # for edge in red_edges:
# #     fw.edges[edge]["lam"] = 0 
# draw_framework(fw, ghost=True)
# tensions = [0]*len(fw.edges)
# edge_dict = get_edge_dict(fw)
# tensions[edge_dict[source]] = 1
# strs = strains(fw, tensions)
# print("target, source strains:",strs[edge_dict[target]], strs[edge_dict[source]])

# os.makedirs(fileroot)
# # ratios = animate(fw, source, target,fileroot, nstars, s_max=1, tensions=1)
