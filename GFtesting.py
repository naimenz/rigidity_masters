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
add_edges = [(17,18), (1,10), (10,21)]
fw.add_edges_from(add_edges)
fw = add_lengths_and_stiffs(fw)
for edge in add_edges:
    fw.edges[edge]["lam"] = 0

# draw_framework(fw,ghost=True)

source = (39,40)
target = (31,42)
def GF_one_step(fw, source, target, tension=1, nstars=[1.0]):
    # establishing source and target
    fw.add_edges_from([source, target])
    fw = add_lengths(fw)
    edge_dict = get_edge_dict(fw)
    fw.edges[source]["lam"] = 0
    fw.edges[target]["lam"] = 0

    source_i, target_i = edge_dict[source], edge_dict[target]

    # tension on source bond
    tensions = np.zeros(len(fw.edges))
    tensions[source_i] = 1

    # greens function for full network and starting extensions
    SSS, SCS = subbases(fw)
    greens = G_f(fw, SCS)
    starting_exts = greens @ tensions

    # for each edge, calculate the change in extensions
    k = fw.graph["k"]
    Fhalf = Fhalf_mat(fw)
    cost_list = []
    nstars = [1.0]
    for i, edge in enumerate(fw.edges):
        # calculate unique SCS bond for Ci
        Ci = calc_Ci(fw, edge, SCS)
        edge_vec = np.zeros(len(fw.edges))
        edge_vec[edge_dict[edge]] = 1
        change = (np.inner(Ci, tensions) / (k *(1 - np.inner(edge_vec, Ci)))) * Ci
        # if fw.edges[edge]["lam"] == 0:
            # print("lam 0 change in exts:", Fhalf @ change)

        # combine change with starting extensions
        new_exts = starting_exts + change

        new_strs = strains(fw, None, Fhalf @ new_exts)
        nohalf_strs = strains(fw, None, new_exts)


        # print(new_strs[target_i], new_strs[source_i])
        ns = [new_strs[target_i] / new_strs[source_i]]
        cost = cost_f(ns, nstars)

        if fw.edges[edge]["lam"] == 0:
            print("LAM 0 NOT ADDING COST:",cost)
            cost_list.append(np.inf)
        else:
            cost_list.append(cost)

    # min cost edge
    min_cost = min(cost_list)
    mindex = cost_list.index(min(cost_list))
    medge = list(fw.edges)[mindex]
    fw.edges[medge]["lam"] = 0

    true_cost = calc_true_cost(fw, source, target, nstars, tension=1)
    print("min cost, true cost:", min_cost, true_cost)
    return medge

print("medge:",GF_one_step(fw, source, target, tension=1, nstars=[1.0]))

# fw = SM_tune_network(fw, source, target, 1, nstars)

































#         # compare to if we had simply not included that edge in the initial graph
#         fwc = fw.copy()
#         fwc.remove_edges_from([edge])
#         c_greens = G_f(fwc)
#         c_tensions = np.zeros(len(fwc.edges))
#         c_tensions[get_edge_dict(fwc)[source]] = 1
#         recalc_exts = c_greens @ c_tensions

#         print(np.allclose(np.concatenate((new_exts[:i], new_exts[i+1:])), recalc_exts))















































# SSS, SCS = subbases(fw)

# Fhalf = Fhalf_mat(fw)
# Fminushalf = np.linalg.pinv(Fhalf)
# edge = (4,18)
# Ci = calc_Ci(fw, edge)
# Q = Qbar_mat(fw) 

# rot = get_rot_mat(SCS[0],Ci)
# SCS_rot = (rot @ SCS.T).T

# edge_dict = get_edge_dict(fw)
# index = edge_dict[edge]
# edge_vec = np.zeros(len(fw.edges))
# edge_vec[index] = 1

# tensions = np.zeros(len(fw.edges))
# tensions[index] = 1 
# # working with scaled system

# exts = extensions(fw, tensions)
# source_index = 15
# target_index = 5

# greens = G_f(fw)
# starting_exts = greens @ tensions
# for i in range(len(fw.edges)):
#     k = fw.graph["k"]
#     edge_vec = np.zeros(len(fw.edges))
#     edge_vec[i] = 1
#     Ci = calc_Ci(fw, list(fw.edges)[i])
#     change = np.inner(Ci, tensions) / (k *(1 - np.inner(edge_vec, Ci))) * Ci
#     new_exts = starting_exts + change

#     fwc = fw.copy()
#     fwc.remove_edges_from([list(fw.edges)[i]])
#     SSS, SCS = subbases(fwc)
#     tensions = np.zeros(len(fwc.edges))
#     tensions[index] = 1
#     recalc_exts = G_f(fwc, SCS) @ tensions

#     # Fhalf = Fhalf_mat(fwc)
#     # edge_dict = get_edge_dict(fwc)
#     # index = edge_dict[edge]
#     # edge_vec = np.zeros(len(fwc.edges))
#     # edge_vec[index] = 1

#     # working with scaled system

#     for i in range(len(recalc_exts)):
#         print(np.isclose(recalc_exts[i], new_exts[i]))

#     new_strs = strains(fw, None, Fhalf @ new_exts)
#     print(new_strs[target_index], new_strs[source_index])
#     ns = [new_strs[target_index] / new_strs[source_index]]
#     cost = cost_f(ns, [1.0])
#     print(cost)
#     change1 = change
#     Ci1 = Ci




# # # WRITING ALGORITHM
# # source = (39,40)
# # target = (25, 35)

# # # GF_tune_network(fw, source, target)







# # b1 = np.array([1,0,0])
# # b2 = np.array([0,1,0])
# # b3 = np.array([0,0,1])
# # # print(b1.dot(b2))


# # # A = Hbar_mat(Hfw)
# # # Ainv = Hbar_inv_mat(Hfw)
# # # ok_(np.allclose(A, A@Ainv@A))
# # # edge = (0,2)
# # # Auinv = update_Hinv(Hfw, edge, Ainv)
# # # Hfw.edges[edge]["lam"] = 0
# # # Ar = Hbar_mat(Hfw)
# # # Arinv = Hbar_inv_mat(Hfw)
# # # print(np.allclose(Auinv, Arinv))

# # # c = Ci/np.linalg.norm(Ci)
# # # c0 = SCS[0]
# # # A = get_rot_mat(c0, c)
# # # ok_(np.allclose(c, A.dot(c0)))
# # # SCS_rot =( A @ SCS.T).T

# # # ki = 0.5
# # # Ci2 = Ci.dot(Ci)



