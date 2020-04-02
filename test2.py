from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

# initialise the seed for reproducibility 
# np.random.seed(102)

# fw = make_nice_fw(20, 1)
# # fw = create_reduced_fw(20, 1)
# flag, comps = pebble_game(fw)
# draw_comps(fw, comps)

xs1 = np.arange(4, 12, 1.0)
ys1 = np.arange(4, 12, 2.0)
ys2 = np.arange(5, 12, 2.0)
xs2 = np.arange(4.5, 12.5, 1.0)
xy1 = np.array(np.meshgrid(xs1, ys1)).T
xy2 = np.array(np.meshgrid(xs2, ys2)).T
positions = np.concatenate((xy1,xy2)).reshape((-1,2))
nodes = list(range(len(positions)))
edges = delaunay_to_edges(Delaunay(positions))
fw = create_framework(nodes, edges, positions)
source = (9,36)
target = (27,54)
draw_framework(fw, source=source, target=target)
tensions = [0]*len(fw.edges)
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
tensions[edge_dict[source]] = 1
strains = exts_to_strains(fw, extensions(fw, tensions))
draw_strains(fw, strains, source, target, ghost=True)

# xs = np.arange(4,12,1)
# ys = np.arange(4,12,1)
# positions = np.array(np.meshgrid(xs,ys)).T.reshape((-1,2))
# nodes = list(range(len(positions)))
# edges = delaunay_to_edges(Delaunay(positions))
# fw = create_framework(nodes, edges, positions)
# draw_framework(fw)

# source = (24,32)
# target = (31,39)
# fw.add_edges_from([source, target])
# fw = add_lengths_and_stiffs(fw)
fw = tune_network(fw, source, target, nstars=[1.0])
tensions = [0]*len(fw.edges)
edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
tensions[edge_dict[source]] = 1
strains = exts_to_strains(fw, extensions(fw, tensions))
draw_strains(fw, strains, source, target, ghost=True)

s_max = 1# strains[edge_dict[source]]
N = 30

u0 = np.zeros(len(fw.nodes) * 2)
real_ratios_list = [0]
for i in range(N):
    strain_val = 0.4 *s_max * (i/(N-1))
    print("target strain val",strain_val)
    constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
    mind = minimize(energy, u0, args=(fw), constraints=constraints)
    print("minimized energy, success, and # iterations",mind.fun, mind.success,mind.nit)
    # print("minimiser:",mind)
    # use the new positions as the starting guess for the next round of optimisation
    u0 = mind.x
    # USING THE DISPLACEMENTS TO WORK OUT REAL NONLINEAR STRAINS
    real_exts = rig_mat(fw).dot(u0)
    real_strains = exts_to_strains(fw, real_exts)
    real_ratio = real_strains[edge_dict[target]]/real_strains[edge_dict[source]]
    if real_ratio != nan:
        real_ratios_list.append(real_ratio)
    print("full nonlinear strain ratio:", real_ratio)
    draw_framework(update_pos(fw, mind.x), filename="images/anim6/anim_"+str(i)+".png",ghost=True, source=source, target=target)
    plt.close()
    print("drawn",i+1,"images of",N)

# # flag, comps = pebble_game(fw)
# # draw_comps(fw,comps)
# # test of creating a dictionary to keep track of edges
# # edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

# # # modifying the framework to change two bonds to ghost bonds (source and target)
# # source = (9, 12)
# # target = (16, 17)
# # fw.edges[source]["lam"] = 0
# # fw.edges[target]["lam"] = 0

# # tensions = [0]*len(fw.edges)
# # tensions[edge_dict[source]] = 1
# # strains = exts_to_strains(fw, extensions(fw, tensions))
# # print("initial strain ratio:",strains[edge_dict[target]]/strains[edge_dict[source]])
# # draw_strains(fw, strains, ghost=True)

# # # aiming for proportional movement of (16,17) when (9,12) moves
# # nstars = [1.0]

# # # calculating ns test
# # it = 0
# # min_cost = np.inf
# # while min_cost > 0.0001 and it < 10000:
# #     costs = []
# #     exts_list = all_extensions(fw, tensions)
# #     # for exts in exts_list:
# #     #     strains = exts_to_strains(fw, exts)
# #     #     ns = [strains[edge_dict[(16,17)]] / strains[edge_dict[(9,12)]]]
# #     #     costs.append(cost_f(ns, nstars))
# #     for i, exts in enumerate(exts_list):
# #         strains = exts_to_strains(fw, exts)
# #         ns = [strains[edge_dict[target]] / strains[edge_dict[source]]]
# #         costs.append(cost_f(ns, nstars))
# #     min_cost = min(costs)
# #     index_to_remove = costs.index(min_cost)
# #     edge_to_remove = list(fw.edges)[index_to_remove]
# #     fw.edges[edge_to_remove]["lam"] = 0
# #     print("iteration:",it,"cost:",min_cost,"removed:",edge_to_remove)
# #     it+=1

# # strains = exts_to_strains(fw, extensions(fw, tensions))
# # draw_strains(fw, strains, ghost=True)

# # # s_max = 1.2
# # # for i in range(20):
# # #     strain_val = s_max * (i/4)
# # #     constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
# # #     u0 = np.zeros(len(fw.nodes) * 2)
# # #     mind = minimize(energy, u0, args=(fw), constraints=constraints)
# # #     draw_framework(update_pos(fw, mind.x), filename="anim/anim_"+str(i)+".png",ghost=True)
# # #     print("drawn",i+1,"images of",20)
    

# # # triangle = create_framework([0,1,2], [(0,1), (0,2), (1,2)], [(1,0), (3,0), (2,2)])
# # # draw_framework(triangle)
# # # f = np.zeros(2*len(triangle.nodes))
# # # f[0] = 1
# # # f[2] = -1

# # # Rold = rig_mat(triangle,2)
# # # R = create_augmented_rigidity_matrix(triangle,2)
# # # print(R)
# # # print("NEW:",np.array([-1/2, 0, 0,0,0,0]).dot(R))
# # # print("OLD:",np.array([-1/2, 0, 0]).dot(Rold))
# # # print(R.dot(R.T))

# # # draw_strains(triangle, f)

# # # shape = create_framework([0,1,2,3,4], [(0,1), (0,2), (1,2),(3,4), (2,3), (2,4),(1,4)], [(1,0), (3,0), (2,2),(1,3),(3,3)])
# # # draw_framework(shape)
# # # f = np.zeros(2*len(shape.nodes))
# # # f[0] = 0
# # # f[1] = 0
# # # f[2] = 3
# # # f[4] = -1
# # # f[5] = 1.5

# # # draw_strains(shape, f)

# # # square = create_framework([0,1,2,3], [(0,1), (0,2), (3,2),(3,1)], [(1,0), (3,0), (1,2),(3,2)])
# # # draw_framework(square)
# # # f = np.zeros(2*len(square.nodes))
# # # f[0] = 0.1
# # # f[1] = 0.1
# # # f[2] = -0.1
# # # f[3] = -0.1

# # # f[4] = 
# # # f[5] = 1.5

# # # draw_tensions(square, f)
# # # triangle = add_lengths(triangle)
# # # tensions = [0]*len(triangle.edges)
# # # tensions[0] = 1
# # # tensions[1] = -1
# # # print(extensions(triangle, tensions))
    



