from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize

# initialise the seed for reproducibility 
np.random.seed(102)

fw = create_reduced_fw(20, 1)
flag, comps = pebble_game(fw)
draw_comps(fw, comps)

draw_framework(fw)
# flag, comps = pebble_game(fw)
# draw_comps(fw,comps)
# test of creating a dictionary to keep track of edges
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

# returns a tuned network
source = (9, 11)
target = (15, 17)
nstars = [1.0]

fw = tune_network(fw, source, target, tension=1, nstars=nstars, draw=True)
tensions = [0]*len(fw.edges)
tensions[edge_dict[source]] = 1
strains = exts_to_strains(fw, extensions(fw, tensions))
draw_strains(fw, strains, ghost=True)

# getting the max strain on the source edge
s_max = strains[edge_dict[source]]
# number of frames in the animation
n = 60
# for i in range(n):
#     strain_val = 0.4 *s_max * (i/(n-1))
#     print("target strain val",strain_val)
#     constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
#     u0 = np.zeros(len(fw.nodes) * 2)
#     mind = minimize(energy, u0, args=(fw), constraints=constraints)
#     # print("minimized energy",mind.fun)
#     print("minimiser:",mind)
#     draw_framework(update_pos(fw, mind.x), filename="anim2/anim_"+str(i)+".png",ghost=True)
#     plt.close()
#     print("drawn",i+1,"images of",n)
