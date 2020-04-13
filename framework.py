import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import scipy
import poisson_disc_sample as pd

from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.optimize import linprog

import sys

# simple function wrapping graph creation in nx
# NOTE: sorts the edges before adding them to the graph
def create_structure_graph(nodes, edges):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

# add an embedding to a structure graph, creating a framework
# takes a nx graph with n nodes and a list of n positions and returns a copy
def assign_pos(graph, positions): 
    # create a copy of the structure graph to turn into a framework by adding node positions 
    fw = graph.copy()

    # shorter aliases for graph objects
    nodeview = fw.nodes
    edges = fw.edges

    # assign positions
    for node, position in zip(nodeview, positions):
        nodeview[node]["position"] = position

    return fw

# takes a set of nodes, edges, and positions and create a framework with lengths on edges
def create_framework(nodes, edges, positions):
    g = create_structure_graph(nodes, edges)
    fw = assign_pos(g, positions)
    fw = add_lengths_and_stiffs(fw)
    return fw

# get the lengths of each edge in the framework 
# also add bulk modulus (lambda) of 1
# NOTE: I have no clue which modulus we're talking about,
# but I now assume it's the Young's modulus
def add_lengths_and_stiffs(fw, lam=1):
    # create a copy to add the lengths to and to return
    rv = fw.copy()
    for edge in rv.edges:
        v1 = edge[0]
        v2 = edge[1]
        pos1 = rv.nodes[v1]["position"]
        pos2 = rv.nodes[v2]["position"]
        l = 0
        for i in range(len(pos1)):
            l += (pos1[i] - pos2[i])**2
        l = np.sqrt(l)
        rv.edges[edge]["length"] = l
        rv.edges[edge]["lam"] = lam
        rv.graph["k"] = 1/lam
    return rv

# creates the rigidity matrix for a d-dimensional framework
# takes in a framework (nx graph with positions) and returns a numpy array
def rig_mat(fw, d=2):
    edgeview = fw.edges
    nodeview = fw.nodes
    n = len(list(fw))
    e = len(edgeview)
    M = np.zeros((e, d*n))
    # print("SORTED",edgeview)
        
    for row, edge in enumerate(edgeview):
        i,j = edge
        pos1 = nodeview[i]["position"]
        pos2 = nodeview[j]["position"]

        if d == 1:
            M[row, d*i] = pos1 - pos2
            M[row, d*j] = -pos1 + pos2
        else:
            for k in range(d):
                M[row, d*i+k] = pos1[k] - pos2[k]
                M[row, d*j+k] = -pos1[k] + pos2[k]

    return M

# generate the flexibility matrix (as defined in The Paper)
# NOTE: we assume a constant material modulus (lam) of 1 (as I don't know anything about this)
# because lam=1, and the diag is 1/ki = 1/(lam/length), we just get a diagonal matrix of lengths
# NOTE: I don't know what to do about 0 stiffness, so atm i'll just set it to 0 manually
def flex_mat(fw):
    es = fw.edges
    entries = [es[edge]["length"]/es[edge]["lam"] if es[edge]["lam"]!= 0 else 0 for edge in es]
    # entries = [fw.edges[edge]["length"]/fw.edges[edge]["lam"] for edge in fw.edges]
    return np.diag(entries)

# the square root of the flex matrix
def Fhalf_mat(fw):
    F = flex_mat(fw)
    return np.sqrt(F)

# in the scaled version, all edges have length 1, so it's just 1/lam
def F_bar_mat(fw):
    es = fw.edges
    entries = [1/es[edge]["lam"] if es[edge]["lam"]!= 0 else 0 for edge in es]
    return np.diag(entries)

def Q_bar_mat(fw):
    Fhalf_inv = np.linalg.pinv(Fhalf_mat(fw))
    R = rig_mat(fw)
    Q_bar = R.T.dot(Fhalf_inv)
    return Q_bar

def get_Hinv(fw):
    R = rig_mat(fw,2)
    Rt = R.T
    F = flex_mat(fw)
    print(F)
    Finv = np.linalg.pinv(F)
    H = Rt.dot(Finv).dot(R)
    Hinv = np.linalg.pinv(H)
    
    return Hinv

# function to update the inverse using Sherman-Morrison
def update_Hinv(fw, edge, Hinv=None):
    edge_dict = get_edge_dict(fw)
    i= edge_dict[edge]
    R = rig_mat(fw)

    if Hinv is None:
        Hinv = get_Hinv(fw)

    qi = R[i,:].reshape(-1,1)
    numerator = 0.5 * Hinv @ qi @ qi.T @ Hinv
    denom = 1 + (qi.T @ Hinv @ qi)

    Hinv_bar = Hinv + numerator/denom
    return Hinv_bar
    # print((Hinv @ qi @ qi.T @ Hinv).shape)
    # Hinv_bar = Hinv - (Hinv @
    

    

# convert a delaunay object into a list of edges that can be used to create a framework
def delaunay_to_edges(d):
    rv = []
    for tri in d.simplices:
        rv.append((tri[0], tri[1]))
        rv.append((tri[1], tri[2]))
        rv.append((tri[0], tri[2]))
    return rv

# creates a random framework with n points and minimum distance r, starting at (0,0).
# Edges are from the delaunay triangulation
def create_random_fw(n, r, seed=None):
    positions = pd.poisson_disc_sample(n, r=r, seed=seed)
    nodes = list(range(len(positions)))
    edges = delaunay_to_edges(Delaunay(positions))
    fw = create_framework(nodes, edges, positions)
    return fw

# making nice starting random frameworks using a Delaunay triangulation,
# removing all 'long' edges from the periphery (as long as that doesn't create
# pendant vertices) and then reducing using the two heuristics Louis told 
# me about (from some paper I can't find)
def make_nice_fw(n, r, seed=None):
    fw = create_random_fw(n, r=r, seed=seed)
    N = len(fw.nodes)
    i = 0
    # TEST: first stripping all long edges
    for edge in fw.edges:
        X0 = np.array(fw.nodes[edge[0]]["position"])
        X1 = np.array(fw.nodes[edge[1]]["position"])
        if np.linalg.norm(X0 - X1) > 2 * r and heuristic2(fw, edge, degree=1):
            fw.remove_edge(edge[0], edge[1])

    while len(fw.edges) > 2 * N and i < len(fw.edges):
        i += 1
        index = np.random.choice(len(fw.edges))
        edge = list(fw.edges)[index]
        # TEST: removing all really long edges
        if heuristic2(fw, edge) and heuristic1(fw, edge):
            fw.remove_edge(edge[0], edge[1])
            i = 0
    if i == len(fw.edges):
        print("FAILED TO REMOVE ALL EDGES")
    return fw

# from stackexchange: https://stackoverflow.com/a/43564754
def in_hull(points, x):
    points = np.array(points)
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

# are all edges on the same 'side' of the node, 
# i.e. are they contained in the same half space (defined by an hplane that goes through the node)?
# can be phrased in terms of convex combinations
def heuristic1(fw, edge):
    fwc = fw.copy()
    # checking both nodes
    fwc.remove_edge(edge[0], edge[1])
    for node in edge:
        neighbors = fwc.neighbors(node)
        positions = [fwc.nodes[neighb]["position"] for neighb in neighbors]
        # if the node isn't in the hull of its neighbours, then all the edges are on the same side
        if not in_hull(positions, fwc.nodes[node]["position"]):
            return False
    # if both nodes succeeded, we pass this heuristic
    return True

# check that we aren't creating any degree two nodes
def heuristic2(fw, edge, degree=2):
    return (fw.degree(edge[0]) > degree+1 and fw.degree(edge[1]) > degree+1)


# creates a random framework and then removes edges until
# there are just twice as many edges left 
# NOTE: not guaranteed to be rigid(still in 2D)
def create_reduced_fw(n, r, seed=None):
    fw = create_random_fw(n, r=r, seed=seed)
    while len(fw.edges) > 2*len(fw.nodes):
            index = np.random.choice(len(fw.edges))
            edge = list(fw.edges)[index]
            if fw.degree(edge[0]) > 2 and fw.degree(edge[1]) > 2:
                fw.remove_edge(edge[0], edge[1])
    return fw

# if filename, an image is saved
# if ghost, draw the ghost bonds (lam=0) a different colour
def draw_framework(fw, filename=None, ghost=False, source=None, target=None, equal=True):
    nodeview = fw.nodes
    fig, ax = plt.subplots(figsize=(20,10))
    if equal:
        ax.set_aspect('equal')
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw_networkx_nodes(fw, pos, with_labels=True)
    nx.draw_networkx_labels(fw,pos)
    if source:
        nx.draw_networkx_nodes(fw, pos, nodelist=source, node_color="g")
    if target:
        nx.draw_networkx_nodes(fw, pos, nodelist=target, node_color="r")

    # draw edges, with or without care for ghost edges
    if ghost:
        ghost_es = set([edge for edge in fw.edges if fw.edges[edge]["lam"]==0])
        other_es = set(fw.edges) - ghost_es
        nx.draw_networkx_edges(fw, pos, edgelist=ghost_es, style="dashed", with_labels=True)
        nx.draw_networkx_edges(fw, pos, edgelist=other_es, with_labels=True)
    else:
        nx.draw_networkx_edges(fw, pos, with_labels=True)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

# draws the components 'comps' of the framework 'fw'
# trivial components have their edges drawn in grey
# edges in non-trivial components are solid black
# vertices in one component are red
# vertices in multiple components are green
def draw_comps(fw, comps, filename=None, show=True, recent_edge=None):
    big_comps = [comp for comp in comps if len(comp) > 2]  
    # calculating which vertices are in multiple components
    greens = set()
    for A in big_comps:
        for B in big_comps:
            if A != B:
                greens |= A & B

    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_aspect('equal')
    nodeview = fw.nodes
    reds = set(nodeview) - greens
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw_networkx_nodes(fw, pos, nodelist=list(greens), node_color='g')
    nx.draw_networkx_nodes(fw, pos, nodelist=list(reds), node_color='r')
    if fw.edges:
        nx.draw_networkx_edges(fw, pos, alpha=0.4, width=1)
    nx.draw_networkx_labels(fw, pos)

    if recent_edge:
        nx.draw_networkx_edges(fw, pos,edgelist=[recent_edge], width=6, edge_color='b')
    for comp in big_comps:
        nx.draw_networkx_edges(fw.subgraph(list(comp)), pos, width=3)


    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if show:
        plt.show()


# creates the rigidity matrix with its nullspace appended as rows beneath
def create_augmented_rigidity_matrix(fw, d):
    M = rig_mat(fw, d=2)
    null = scipy.linalg.null_space(M)
    R = np.vstack((M, null.T))
    return R

# function to calculate the nullity (and hence dimension of space of inf. rigid motions) of R
def nullity(R):
    n = R.shape[1]
    return n - np.linalg.matrix_rank(R)

# calculate the dimension of the affine span of vectors
# works by shifting v0 to the origin and calculating the span of that set of vectors
def calc_affine_span_dim(vectors):
    v0 = vectors[0] 
    new_vs = np.array([v - v0 for v in vectors])
    return np.linalg.matrix_rank(new_vs)

# takes a normal (non constricted, i.e. not embedded in lower dimensions) d-space framework
# and returns True if it is inf. rigid, False otherwise
def is_inf_rigid(fw, d):
    size_V = len(fw)
    nodeview = fw.nodes
    vs = np.array([nodeview[node]["position"] for node in nodeview])
    aspan_dim = calc_affine_span_dim(vs) 
    # print(aspan_dim)
    
    # if aspan_dim < min(size_V - 1, d):
    #     return False

    # else:
    R = create_augmented_rigidity_matrix(fw, d)
    # R = rig_mat(fw, d)
    # print("d=",d," - ",np.linalg.matrix_rank(R) ,  d*size_V - (((d+1) * d) / 2))
    return np.linalg.matrix_rank(R) == d*size_V - (((d+1) * d) / 2)


# Code from https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989
# for making the midpoint of the colormap 0
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# draws the framework 'fw' with tensions resulting from force 'f'
def draw_tensions(fw, f):
    R = create_augmented_rigidity_matrix(fw, 2)
    R1 = rig_mat(fw, 2)
    s1 = R.dot(f)
    s = R1.dot(f)
    print("OLD:",s1)
    print("NEW:",s)
    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_aspect('equal')
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}

    e_labels=dict()
    for i, edge in enumerate(fw.edges):
        e_labels[edge] = np.round(s[i], 4)

    # places where the force is being applied will be coloured green
    applied_forces = dict()
    for i in range(0, len(f), 2):
        # if the force is non-zero 
        if (f[i] != 0 or f[i+1] != 0):
            # put the force into a dictionary
            applied_forces[i//2] = (f[i], f[i+1])
       
    cmap = plt.cm.coolwarm
    # norm = MidpointNormalize(midpoint=0)
    nx.draw(fw, pos, edge_color=s,
            width=4, edge_cmap=cmap, edge_vmin=-max(abs(s)), edge_vmax=max(abs(s)), with_labels=True)
    # STRESSED NODES IN GREEN
    nx.draw_networkx_nodes(fw, pos, nodelist=applied_forces.keys(), node_color='green')
    nx.draw_networkx_edge_labels(fw, pos, e_labels)
    # drawing the applied forces as black bars
    for key in applied_forces.keys():
        x, y = fw.nodes[key]["position"]
        plt.plot([x,x+applied_forces[key][0]], [y,y+applied_forces[key][1]], color='k', linestyle='-', linewidth=2)

    # fig.savefig("stress.png")
    plt.show()
    

# get applied forces from applied tension
def forces(fw, tstar):
    R = rig_mat(fw, 2)
    Rt = R.T
    f = Rt.dot(tstar)
    return f

# get displacements from applied tension
def displacements(fw, tstar, fs=None, passR=False, debug=False):
    R = rig_mat(fw,2)
    Rt = R.T
    F = flex_mat(fw)
    Finv = np.linalg.pinv(F)
    H = Rt.dot(Finv).dot(R)
    Hinv = np.linalg.pinv(H)

    if debug:
        print("============ R ============\n",np.around(R,2))
        print("============ F ============\n",np.around(F,2))
        print("============ H ============\n",np.around(H,2))
        print("============ Hinv ============\n",np.around(Hinv,2))

    if not fs is None: 
        u = Hinv.dot(fs)
    else:
        u = Hinv.dot(Rt).dot(tstar)

    if passR:
        return u, R
    else:
        return u

# get displacements from applied tension
def extensions(fw, tstar, disps=None, debug=False): 
    if not (disps is None):
        R = rig_mat(fw, 2)
        exts = R.dot(disps)
    else:
        u, R = displacements(fw, tstar, passR=True, debug=debug)
        exts = R.dot(u)

    return exts

# get strains from applied tension
def strains(fw, tstar, exts=None, debug=False):
    if exts is None:
        exts = extensions(fw, tstar, debug=debug)
    Nb = len(exts)
    strs = np.zeros(Nb)
    edge_list = list(fw.edges)
    for i in range(Nb):
        strs[i] = exts[i] / fw.edges[edge_list[i]]["length"]
    return strs

# gets the SSS and SCS subbases, and returns them in that order
# recall SSS is states of self-stress, i.e. tensions that do not result in net forces on the nodes
# also thought of as incompatible stresses, extensions that don't lead to valid node displacements
# SCS is states of compatible stress, which DO correspond to net forces on the nodes
# NOTE: I'm assuming we calculate this based on the scaled Q matrix
def subbases(fw):
    Q_bar = Q_bar_mat(fw)
    U, sigma, Vt = np.linalg.svd(Q_bar)
    mask = (np.isclose(sigma, 0))
    U_red = U[:,:len(mask)]
    SSS = U_red[:,mask].T
    SCS = U_red[:,~mask].T
    return SSS, SCS

# An attempt at calculating the discrete green's
def G_f(fw, SCS=None):
    if SCS is None:
        _, SCS = subbases(fw)

    k = fw.graph["k"]
    green = k*np.sum(np.array([np.outer(c, c) for c in SCS]),axis=0)
    return green

# function to stop me doing this all the time
def get_edge_dict(fw):
    es = fw.edges
    return {edge: i for i, edge in enumerate(fw.edges)}

def Ci(fw, edge, G=None):
    if G is None:
        G = G_f(fw)
    k = fw.graph["k"]

    edge_dict = get_edge_dict(fw)
    index = edge_dict[edge]
    edge_vec = np.zeros(len(G))
    edge_vec[index] = 1
    C = k * G.dot(edge_vec)
    



# NOTE: OLD VERSION
# # get extensions from applied tension to framework
# def old_extensions(fw, tstar, debug=False):
#     R = rig_mat(fw,2)
#     Rt = R.T
#     F = flex_mat(fw)
#     Finv = np.linalg.pinv(F)
#     # Finv = scipy.linalg.pinv2(F)
    
#     H = Rt.dot(Finv).dot(R)
#     Hinv = np.linalg.pinv(H)
#     # Hinv = scipy.linalg.pinv2(H)#, rcond=1e-2)
#     # TESTING BIGGER RCOND
#     # Hinv = np.linalg.pinv(H, rcond=1e-1)
#     if debug:
#         # trying to figure out why the numbers are so big
#         # print("allclose?",np.allclose(H , np.dot(H, np.dot(Hinv, H))))#, rtol=1e-3))
#         # print("diff?",(H - np.dot(H, np.dot(Hinv, H))))#, rtol=1e-3))
#         # print("allclose?",np.allclose(Hinv, np.dot(Hinv, np.dot(H, Hinv))))#, rtol=1e-3))
#         # print("MEAN:",np.mean(np.abs(Hinv)))
#         # print("LHS:",Rt.dot(tstar))
#         u = Hinv.dot(Rt).dot(tstar)
#         # print("H:",list(H))
#         # plt.imshow(H)
#         # plt.show()
#         # print("RHS:", H.dot(u))
#         # print("disps:",u)
#         # print("Hu == Qt*:", (H.dot(u)- Rt.dot(tstar)))
#     return R.dot(Hinv.dot(Rt).dot(tstar))

# test function to see if I can work out the extensions from removing each edge
def all_extensions(fw, tstar):
    exts = []
    R = rig_mat(fw,2)
    Rt = R.T
    for edge in fw.edges:
        fwc = fw.copy()  
        fwc.edges[edge]["lam"] = 0
        F = flex_mat(fwc)
        Finv = np.linalg.pinv(F)
        # Finv = scipy.linalg.pinv2(F)
        H = Rt.dot(Finv).dot(R)
        
        Hinv = np.linalg.pinv(H)
        # Hinv = scipy.linalg.pinv2(H)#, rcond=1e-2)
        # TESTING BIGGER RCOND
        # Hinv = np.linalg.pinv(H, rcond=1e-1)
        # Hinv = np.linalg.pinv(H, rcond=1e-10)
        # if not (np.allclose(H, np.dot(H, np.dot(Hinv, H))) and np.allclose(Hinv, np.dot(Hinv, np.dot(H, Hinv)))):
        #     print("allclose?",np.allclose(H, np.dot(H, np.dot(Hinv, H))))#, rtol=1e-3))
        #     print("allclose?",np.allclose(Hinv, np.dot(Hinv, np.dot(H, Hinv))))#, rtol=1e-3))
        exts.append(R.dot(Hinv.dot(Rt).dot(tstar)))
    return exts

# converts extensions to strains for a given framework
# NOTE: works on a single set of extensions
def exts_to_strains(fw, exts):
    Nb = len(exts)
    strains = [0]*Nb
    edge_list = list(fw.edges)
    for i in range(Nb):
        strains[i] = exts[i] / fw.edges[edge_list[i]]["length"]
    return strains
        
# implementing the cost function on strains as in the paper
def cost_f(ns, nstars):
    cost = 0
    for nj, njstar in zip(ns, nstars):
        if njstar == 0:
            cost += nj**2
        else:
            cost += (nj/njstar - 1)**2

    return cost

# borrowing heavily from draw_tensions, trying to draw the strains on the bonds
def draw_strains(fw, strains, source=None, target=None, ghost=False, filename=None):
    # works on a numpy array
    strains = np.array(strains)
    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_aspect('equal')
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}

    e_labels=dict()
    for i, edge in enumerate(fw.edges):
        e_labels[edge] = np.round(strains[i], 1)

    cmap = plt.cm.coolwarm
    nx.draw_networkx_edges(fw, pos, edgelist=fw.edges, edge_color=strains,
            width=4, edge_cmap=cmap, edge_vmin=-max(abs(strains)), edge_vmax=max(abs(strains)), with_labels=True)
    if ghost:
        ghost_es = set([edge for edge in fw.edges if fw.edges[edge]["lam"]==0])
        nx.draw_networkx_edges(fw, pos, edgelist=ghost_es, style="dashed")
    if source:
        nx.draw_networkx_nodes(fw, pos, nodelist=source, node_color="g")
    if target:
        nx.draw_networkx_nodes(fw, pos, nodelist=target, node_color="r")
       
    # bbox = dict(boxstyle="round", alpha=0.0)
    nx.draw_networkx_edge_labels(fw, pos, e_labels, font_size=6)#, bbox=bbox)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()

# NOTE: Test to see if there is numerical error in cost
def calc_true_cost(fw_orig, source, target, nstars, tension=1):
    fw = fw_orig.copy()
    rem_edges = [edge for edge in fw.edges if (fw.edges[edge]["lam"] == 0 and (edge != source and edge != target))]
    fw.remove_edges_from(rem_edges)
    # print("true len",len(fw.edges))
    # new edge dict for smaller framework
    edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
    tensions = [0]*len(fw.edges)
    tensions[edge_dict[source]] = tension
    strains = exts_to_strains(fw, extensions(fw, tensions))
    ns = [strains[edge_dict[target]] / strains[edge_dict[source]]]
    true_cost = cost_f(ns, nstars)
    # remove another two edges to check rank
    # fw.remove_edges_from([source,target])
    # R = rig_mat(fw,2)
    # rank = np.linalg.matrix_rank(R)
    # print("rank of R:",rank, "2n - 3:", 2*len(fw.nodes) - 3, "number of edges:",len(fw.edges))
    return true_cost

# Run the tuning algorithm on a network for a given source, target, ratio
def tune_network(fw_orig, source, target, tension=1, nstars=[1.0], cost_thresh=0.001, it_thresh=100, draw=False, verbose=True):
    fw = fw_orig.copy()
    if source not in fw.edges or target not in fw.edges:
        fw.add_edges_from([source, target])
        fw = add_lengths_and_stiffs(fw)

    edge_dict = {edge: i for i, edge in enumerate(fw.edges)}

    # modifying the framework to change two bonds to ghost bonds (source and target)
    fw.edges[source]["lam"] = 0
    fw.edges[target]["lam"] = 0

    tensions = [0]*len(fw.edges)
    tensions[edge_dict[source]] = tension
    strs = strains(fw, tensions)
    # strains = exts_to_strains(fw, extensions(fw, tensions, debug=True))
    if verbose:
        print("initial strain ratio:",strs[edge_dict[target]]/strs[edge_dict[source]])
    if draw:
        draw_strains(fw, strs, source, target, ghost=True)

    # calculating ns test
    it = 0
    min_cost = np.inf
    while min_cost > cost_thresh and it < it_thresh:
        costs = []
        exts_list = all_extensions(fw, tensions)
        for i, exts in enumerate(exts_list):
            strs = exts_to_strains(fw, exts)
            ns = [strs[edge_dict[target]] / strs[edge_dict[source]]]
            costs.append(cost_f(ns, nstars))
        min_cost = min(costs)
        index_to_remove = costs.index(min_cost)
        edge_to_remove = list(fw.edges)[index_to_remove]
        fw.edges[edge_to_remove]["lam"] = 0
        # test to see numerical error
        # if verbose:
            # print("fake len",len(fw.edges))
        true_cost = calc_true_cost(fw, source, target, nstars, tension)

        if verbose:
            print("iteration:",it,"cost:",min_cost,"removed:",edge_to_remove)
            print("true cost:",true_cost)
            print("relative percentage error:",100*abs(true_cost - min_cost) / true_cost)
        it+=1

    return fw


# ============================================================================== 
# ANIMATION FUNCTIONS
# ============================================================================== 
# calculate the energy of the configuration
# accepts a displacement vector u and the framework as *args from minimizer
# displacements are structured as [x0, y0, x1, y1, ..., xn, yn] where there are n nodes
def energy(u, *args):
    fw = args[0]
    energy = 0
    # looping over all bonds in the network
    for edge in fw.edges:
        i = edge[0]
        j = edge[1]
        posi = np.array(fw.nodes[i]["position"])
        posj = np.array(fw.nodes[j]["position"])
        Xi = posi + np.array([u[2*i], u[2*i + 1]])
        Xj = posj + np.array([u[2*j], u[2*j + 1]])
        mag_Xij = np.linalg.norm(Xi - Xj)

        lam = fw.edges[edge]["lam"]
        lij = fw.edges[edge]["length"]
        # stiffness of edge (i,j), kij, is material modulus lambda over equilibrium length
        kij = lam/lij
        energy += 0.5 * kij * (mag_Xij - lij)**2
    return energy

# function to ensure that the strain on the edge "edge" is "val"
def source_strain(u, *args):
    fw = args[0]
    edge = args[1]
    val = args[2]

    # dict to get edge index from name
    edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
    # getting the rigidity matrix (Q^T) to calculate extensions from displacement
    R = rig_mat(fw)
    exts = R.dot(u)
    strs = exts_to_strains(fw, exts)
    index = edge_dict[edge]
    # returns 0 if constraint is met
    return strs[index] - val

# create a copy of the framework with positions changed according to a given displacement
def update_pos(fw, u):
    fwc = fw.copy()
    for i in fwc.nodes:
        disp = np.array([u[2*i], u[2*i + 1]])
        pos = np.array(fwc.nodes[i]["position"])
        new_pos = list(pos + disp)
        fwc.nodes[i]["position"] = new_pos

    return fwc

# # getting the max strain on the source edge
def animate(fw, source, target, fileroot, nstars, s_max=1, tensions=1):
    # # number of frames in the animation
    n = 30
    real_ratios_list = []
    edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
    tensions = [0]*len(fw.edges)
    tensions[edge_dict[source]] = 1
    u0 = np.zeros(len(fw.nodes) * 2)
    for i in range(n):
        strain_val = 0.5 *s_max * (i/(n-1))
        print("=========== ITERATION",str(i),"============")
        print("target strain val:",strain_val)
        constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
        mind = minimize(energy, u0, args=(fw), constraints=constraints)
        print("minimized energy, success, and # iterations:",mind.fun, mind.success,mind.nit)
        # using solution of prev it for start of next
        u0 = mind.x
        real_exts = rig_mat(fw).dot(u0)
        real_strains = exts_to_strains(fw, real_exts)
        real_ratio = real_strains[edge_dict[target]]/real_strains[edge_dict[source]]
        if real_ratio != np.nan:
            real_ratios_list.append(real_ratio)
        print("full nonlinear strain ratio:", real_ratio)
        draw_framework(update_pos(fw, mind.x), filename=fileroot+"anim_"+str(i)+".png",ghost=True, source=source, target=target)
        plt.close()
        print("drawn",i+1,"images of",n)

    fig = plt.figure()
    plt.plot(real_ratios_list)
    plt.axhline(nstars[0],linestyle="--")
    plt.show()
    fig.savefig(fileroot+"ratio.png",bbox_inches="tight")
    return real_ratios_list
