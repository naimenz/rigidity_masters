from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize

# initialise the seed for reproducibility 
np.random.seed(103)

fw = make_nice_fw(200, 1)
flag, comps = pebble_game(fw)
draw_comps(fw, comps)

# returns a tuned network
source = (189, 199)
target = (179, 183)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)
# test of creating a dictionary to keep track of edges
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

fw = tune_network(fw, source, target, tension=1, nstars=nstars, draw=True)
# fw.edges[source]["lam"] = 0
# fw.edges[target]["lam"] = 0
# what the above does is remove the following two bonds:
# rem_edges = [(132,181), (187,188),(91,191),(130,179),(188,190),(65,66),
#         (143,195),(179,180),(176,177),(49,50),(79,104),(30,32),(71,110),(71,99),(110,159),(97,100),(11,28),(1,8),(160,186)]
# for edge in rem_edges:
#     fw.edges[edge]["lam"] = 0

tensions = [0]*len(fw.edges)
tensions[edge_dict[source]] = 1
# trying tension on neighbouring bonds
# tensions[edge_dict[(195,199)]] = 1
exts = extensions(fw, tensions, True)
print("ext on source, target resp.", exts[edge_dict[source]], exts[edge_dict[target]])
strains = exts_to_strains(fw, exts)
print("strains on source, target resp.",strains[edge_dict[source]], strains[edge_dict[target]])
draw_strains(fw, strains, source, target, ghost=True)

animate(fw, source, target, "images/anim2/",s_max=1, tensions=1)
# # # getting the max strain on the source edge
# s_max = 1# strains[edge_dict[source]]
# # # number of frames in the animation
# n = 30
# real_ratios_list = []
# for i in range(n):
#     strain_val = 0.4 *s_max * (i/(n-1))
#     print("target strain val",strain_val)
#     constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
#     u0 = np.zeros(len(fw.nodes) * 2)
#     mind = minimize(energy, u0, args=(fw), constraints=constraints)
#     print("minimized energy, success, and # iterations",mind.fun, mind.success,mind.nit)
#     real_exts = rig_mat(fw).dot(u0)
#     real_strains = exts_to_strains(fw, real_exts)
#     real_ratio = real_strains[edge_dict[target]]/real_strains[edge_dict[source]]
#     if real_ratio != np.nan:
#         real_ratios_list.append(real_ratio)
#     print("full nonlinear strain ratio:", real_ratio)
#     draw_framework(update_pos(fw, mind.x), filename="images/anim2/anim_"+str(i)+".png",ghost=True, source=source, target=target)
#     plt.close()
#     print("drawn",i+1,"images of",n)

# plt.plot(real_ratios_list[1:])
# plt.show()
