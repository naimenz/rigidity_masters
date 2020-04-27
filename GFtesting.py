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
draw_framework(fw)
SSS, SCS = subbases(fw)

Fhalf = Fhalf_mat(fw)
Fminushalf = np.linalg.pinv(Fhalf)
edge = (4,18)
Ci = calc_Ci(fw, edge)
Q = Qbar_mat(fw) 
for s in SSS:
    ok_(np.allclose(Q @ s, 0))
    for c in SCS:
        ok_(np.isclose(s @ c, 0))
        ok_(not np.allclose(Q @ c, 0))

rot = get_rot_mat(SCS[0],Ci)
SCS_rot = (rot @ SCS.T).T

edge_dict = get_edge_dict(fw)
index = edge_dict[edge]
edge_vec = np.zeros(len(fw.edges))
edge_vec[index] = 1

tensions = np.zeros(len(fw.edges))
tensions[index] = 1e25
# working with scaled system
tensions = Fhalf @ tensions

exts = Fminushalf @ extensions(fw, tensions)
Fbar = Fbar_mat(fw)
print(Fbar)
# for s in SSS:
#     print(s.dot(exts))

# for c in SCS:
#     print(c.dot(exts))

starting_exts = np.zeros(len(fw.edges))
k = fw.graph["k"]
for c in SCS:
    for cprime in SCS:
        starting_exts += (1/k) * np.outer(c, cprime) @ tensions

# for s in SSS:
#     print("SSS",s.dot(new_exts))

# for c in SCS:
#     print("SCS",c.dot(new_exts))
source_index = 15
target_index = 5

for i in range(len(fw.edges)):
    edge_vec = np.zeros(len(fw.edges))
    edge_vec[i] = 1
    # print("edges",edge_vec.dot(starting_exts))
    print(list(fw.edges)[i])
    Ci = calc_Ci(fw, list(fw.edges)[i])
    # print(np.inner(Ci, Ci), np.inner(edge_vec, Ci))
    change = np.inner(Ci, tensions) / (k *(1 - np.inner(edge_vec, Ci))) * Ci
    print("scalar",np.inner(Ci, tensions) / (k *(1 - np.inner(edge_vec, Ci))))
    new_exts = starting_exts + change


    k = fw.graph["k"]
    fwc = fw.copy()
    fwc.remove_edges_from([list(fw.edges)[i]])
    SSS, SCS = subbases(fwc)
    recalc_exts = np.zeros(len(fwc.edges))

    edge = (4,18)
    Fhalf = Fhalf_mat(fwc)
    edge_dict = get_edge_dict(fwc)
    index = edge_dict[edge]
    edge_vec = np.zeros(len(fwc.edges))
    edge_vec[index] = 1

    tensions = np.zeros(len(fwc.edges))
    tensions[index] = 1e25
    # working with scaled system
    tensions = Fhalf @ tensions
    for c in SCS:
        for cprime in SCS:
            recalc_exts += (1/k) * np.outer(c, cprime) @ tensions

    for i in range(len(recalc_exts)):
        print(np.isclose(recalc_exts[i], new_exts[i]))




    new_strs = strains(fw, None, Fhalf @ new_exts)
    print(new_strs[target_index], new_strs[source_index])
    ns = [new_strs[target_index] / new_strs[source_index]]
    cost = cost_f(ns, [1.0])
    print(cost)
    change1 = change
    Ci1 = Ci




# WRITING ALGORITHM
source = (39,40)
target = (25, 35)

# GF_tune_network(fw, source, target)







b1 = np.array([1,0,0])
b2 = np.array([0,1,0])
b3 = np.array([0,0,1])
# print(b1.dot(b2))


# A = Hbar_mat(Hfw)
# Ainv = Hbar_inv_mat(Hfw)
# ok_(np.allclose(A, A@Ainv@A))
# edge = (0,2)
# Auinv = update_Hinv(Hfw, edge, Ainv)
# Hfw.edges[edge]["lam"] = 0
# Ar = Hbar_mat(Hfw)
# Arinv = Hbar_inv_mat(Hfw)
# print(np.allclose(Auinv, Arinv))

# c = Ci/np.linalg.norm(Ci)
# c0 = SCS[0]
# A = get_rot_mat(c0, c)
# ok_(np.allclose(c, A.dot(c0)))
# SCS_rot =( A @ SCS.T).T

# ki = 0.5
# Ci2 = Ci.dot(Ci)



