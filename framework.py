import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import scipy
import poisson_disc_sample as pd
from nose.tools import ok_

from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.optimize import linprog

import datetime
import time
import sys
import os

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

# add only the lengths so as not to change already defined stiffnesses
def add_lengths(fw):
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
    return rv

# get the lengths of each edge in the framework 
# also add bulk modulus (lambda) of 1 (NOTE now 2 to avoid numerical issues?)
# I have no clue which modulus we're talking about,
# but I now assume it's the Young's modulus
def add_lengths_and_stiffs(fw, lam=2):
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
    rv.graph["k"] = lam
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

        
    # TEST NOTE: rescaling rigidity matrix throughout
        if d == 1:
            M[row, d*i] = (pos1 - pos2)  / edgeview[edge]["length"]
            M[row, d*j] = (-pos1 + pos2)  / edgeview[edge]["length"]
        else:
            for k in range(d):
                M[row, d*i+k] = (pos1[k] - pos2[k]) / edgeview[edge]["length"]
                M[row, d*j+k] = (-pos1[k] + pos2[k]) / edgeview[edge]["length"]
    return M

# generate the flexibility matrix (as defined in The Paper)
def flex_mat(fw):
    es = fw.edges
    entries = [es[edge]["length"]/es[edge]["lam"] if es[edge]["lam"]!= 0 else 0 for edge in es]
    return np.diag(entries)

# the square root of the flex matrix
def Fhalf_mat(fw):
    F = flex_mat(fw)
    # basically having a 0 in F means a row of Q is killed completely
    # so instead i'm setting them to 1 to avoid any scaling
    for i in range(len(F)):
        if F[i,i] == 0:
            F[i,i] = 1
    return np.sqrt(F)

# in the scaled version, all edges have length 1, so it's just 1/k
def Fbar_mat(fw):
    es = fw.edges
    entries = [1/fw.graph["k"] if es[edge]["lam"]!= 0 else 0 for edge in es]
    return np.diag(entries)

def Qbar_mat(fw):
    Fhalf_inv = np.linalg.pinv(Fhalf_mat(fw), hermitian=True)
    R = rig_mat(fw)
    Q_bar = R.T.dot(Fhalf_inv)
    return Q_bar

def H_mat(fw):
    R = rig_mat(fw,2)
    Rt = R.T
    F = flex_mat(fw)
    Finv = np.linalg.pinv(F, hermitian=True)
    H = Rt.dot(Finv).dot(R)
    return H

# calculate the normal Hinv
def Hinv_mat(fw):
    H = H_mat(fw)
    Hinv = np.linalg.pinv(H)
    return Hinv

# calculate H with identical bond stiffness (rescaled)
def Hbar_mat(fw):
    Qbar = Qbar_mat(fw)
    Fbar = Fbar_mat(fw)
    Fbar_inv = np.linalg.pinv(Fbar, hermitian=True)
    Hbar = Qbar @ Fbar_inv @ Qbar.T
    return Hbar

# calculate Hinv with identical bond stiffness (rescaled)
def Hbar_inv_mat(fw):
    Hbar_inv = np.linalg.pinv(Hbar_mat(fw), hermitian=True)
    return Hbar_inv

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

# returns True if b is in the column space of a
# works by checking if there is a linear comb of the cols of a that gives b
def in_col_space(a,b):
    return np.allclose(a@np.linalg.lstsq(a,b, rcond=None)[0], b)

# perform sm_update (assuming updated matrix is invertible
def sm_update(Ainv, u, v):
    Ainvu = Ainv @ u
    vtAinv = v.T @ Ainv
    Auvtinv = Ainv - (1 / (1 + v.T @ Ainvu)) * Ainvu @ vtAinv
    return Auvtinv

def check_sm_update(A, Ainv, u, v):
    Auvt = A + u @ v.T
    Ainvu = Ainv @ u
    vtAinv = v.T @ Ainv
    Auvtinv = Ainv - (1 / (1 + v.T @ Ainvu)) * Ainvu @ vtAinv
    # print("denom is:",(1 + v.T @ Ainvu))
    return Auvt, Auvtinv

def x_dagger(x):
    return x.T / (np.linalg.norm(x)**2)

# calculate the updated pseudoinverse using meyer's 6 theorems
# NOTE assuming real throughout
def meyer_update(A, Ainv, c, d):
    # pseudoinverse of vectors
    def x_dagger(x):
        return x.T / (np.linalg.norm(x)**2)

    M, N = A.shape
    # d star is just d transpose
    dt = d.T
    k = Ainv @ c
    h = dt @ Ainv
    # daggered k and h as required by the paper
    kd = x_dagger(k)
    hd = x_dagger(h)
    u = c - A @ k
    # ok_(np.allclose(u, (np.eye(M) - A @ Ainv) @ c))
    v = dt - h @ A
    # ok_(np.allclose(v, dt @ (np.eye(N) - Ainv @ A)))
    ud = x_dagger(u)
    vd = x_dagger(v)
    beta = 1 + dt @ Ainv @ c 
    # print("BETA IS:",beta)

    # NOTE: experimenting with saying c, d always in col space
    # because tbh, they might be theoretically but not numerically, i haven't really thought about it
    # c_in_A = np.isclose(np.linalg.norm(u), 0)
    # d_in_At =  np.isclose(np.linalg.norm(v), 0)
    c_in_A = True 
    d_in_At = True 
    # c_in_A = np.isclose(np.linalg.norm(u), 0)
    # d_in_At =  np.isclose(np.linalg.norm(v), 0)
    # print("NORM u:",np.linalg.norm(u))
    # print("NORM v:",np.linalg.norm(v))
    # ok_(c_in_A == in_col_space(A, c), d_in_At == in_col_space(A.T, d))

    def inverse1():
        # print("inv1")
        return Ainv - (k @ ud) - (vd @ h) + (beta * vd @ ud)

    def inverse2():
        # print("inv2")
        return Ainv - (k @ kd @ Ainv) - (vd @ h)

    def inverse3():
        # print("inv3")
        p1 = - (np.linalg.norm(k)**2 / beta)*v.T - k
        q1t = - (np.linalg.norm(v)**2 / beta)*k.T @ Ainv - h
        sigma1 = np.linalg.norm(k)**2 * np.linalg.norm(v)**2 + beta**2
        return Ainv + (1/beta)*v.T @ k.T @ Ainv - (beta/sigma1)* p1 @ q1t

    def inverse4():
        # print("inv4")
        return Ainv - Ainv @ hd @ h - k @ ud

    def inverse5():
        # print("inv5")
        p2 = - (np.linalg.norm(u)**2/beta)*Ainv @ h.T - k
        q2t = - (np.linalg.norm(h)**2/beta)*u.T - h
        sigma2 = np.linalg.norm(h)**2 * np.linalg.norm(u)**2 + beta**2
        return Ainv + (1/beta) * Ainv @ h.T @ u.T - (beta/sigma2)*p2 @ q2t

    def inverse6():
        # print("inv6")
        return Ainv - (k @ kd @ Ainv) - (Ainv @ hd @ h) + ((kd @ Ainv @ hd) * k @ h)

    # c in R(A)
    if c_in_A:
        # d in R(A*)
        if d_in_At:
            # beta == 0
            if np.isclose(beta, 0):
                # c in R(A), d in R(A*), beta = 0
                # print("BETA IS ZERO:", beta)
                return inverse6()

            # beta != 0 
            else: 
                # c in R(A), d in R(A*), beta != 0
                inv3 = inverse3()
                inv5 = inverse5()
                ok_("3 and 5 CLOSE?:",np.allclose(inv3, inv5))
                # print("GREATER THAN ZERO", beta)
                return(inv5)

        # d not in R(A*)
        else: 
            # beta == 0
            if np.isclose(beta, 0):
                # c in R(A), d not in R(A*), beta = 0
                return inverse2()

            # beta != 0 
            else: 
                # c in R(A), d not in R(A*), beta != 0
                return inverse3()


    # c not in R(A)
    else:
        # d in R(A*)
        if d_in_At:
            # beta == 0
            if np.isclose(beta, 0):
                # c not in R(A), d in R(A*), beta = 0
                return inverse4()

            # beta != 0 
            else: 
                # c not in R(A), d in R(A*), beta != 0
                return inverse5()

        # d not in R(A*)
        else:       
            # beta == 0
            if np.isclose(beta, 0):
                # c not in R(A), d not in R(A*), beta = 0
                return inverse1()

            # beta != 0 
            else: 
                # c not in R(A), d not in R(A*), beta != 0
                print("1")
                return inverse1()
    print("UHOH")

# Update the H and Hinv matrices with a rank 1 addition
def update_Hinv(fw, edge, H, Hinv):
    edge_dict = get_edge_dict(fw)
    i= edge_dict[edge]
    Q = rig_mat(fw).T

    k = fw.edges[edge]["lam"] / fw.edges[edge]["length"]
    qi = Q[:,i].reshape(-1,1)
    # NOTE: inner product of these two is -k (I think)
    u = -k*qi
    v = qi
    # print("u in col space, v in row space:", in_col_space(H, u), in_col_space(H.T, v))
    # print("rank H:", np.linalg.matrix_rank(H), H.shape)
    # print("denom is:",(1 + v.T @ Hinv @ u))
    Hu, Hinv_u = H + u @ v.T, meyer_update(H, Hinv, u, v)
    return Hu, Hinv_u

# update the Hbar and Hbarinv matrices
# note it's different to just Hinv
def update_Hbarinv(fw, edge, H, Hinv):
    edge_dict = get_edge_dict(fw)
    i= edge_dict[edge]
    Qbar = Qbar_mat(fw)

    k = fw.graph["k"]
    print("K IN UPDATE:",k)
    qi = Qbar[:,i].reshape(-1,1)
    # NOTE: inner product of these two is -k (I think)
    u = -(k)*qi
    v = qi
    # print("u in col space, v in row space:", in_col_space(H, u), in_col_space(H.T, v))
    # print("rank H:", np.linalg.matrix_rank(H), H.shape)
    # print("denom is:",(1 + v.T @ Hinv @ u))
    Hu, Hinv_u = H + u @ v.T, meyer_update(H, Hinv, u, v)
    return Hu, Hinv_u

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
    # SETTING AXIS LIMITS
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for val in pos.values():
        minx = min(minx, val[0])
        miny = min(miny, val[1])
        maxx = max(maxx, val[0])
        maxy = max(maxy, val[1])
    xbuf = (maxx - minx) / 10
    ybuf = (maxy - miny) / 10
    plt.xlim(minx - xbuf, maxx + xbuf)
    plt.ylim(miny - ybuf, maxy + ybuf)
    # END SETTING AXIS LIMITS
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
def is_inf_rigid(fw, d=2):
    size_V = len(fw)
    # if it has no edges, can't be rigid
    if len(fw.edges) < 1 and size_V > 1:
        return False
    # nodeview = fw.nodes
    # vs = np.array([nodeview[node]["position"] for node in nodeview])
    # aspan_dim = calc_affine_span_dim(vs) 
    # print(aspan_dim)
    
    # if aspan_dim < min(size_V - 1, d):
    #     return False

    # else:
    # R = create_augmented_rigidity_matrix(fw, d)
    R = rig_mat(fw, d)
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
    Finv = np.linalg.pinv(F, hermitian=True)
    H = Rt.dot(Finv).dot(R)
    Hinv = np.linalg.pinv(H, hermitian=True)

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

# get extensions from applied tension
def extensions(fw, tstar, disps=None, debug=False): 
    if not (disps is None):
        R = rig_mat(fw, 2)
        exts = R.dot(disps)
    else:
        u, R = displacements(fw, tstar, passR=True, debug=debug)
        exts = R.dot(u)

    # TEST NOTE: trying to rescale extensions
#     edge_list = list(fw.edges)
#     for i in range(len(edge_list)):
#         exts[i] = exts[i] / fw.edges[edge_list[i]]["length"]
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

# NOTE: each ROW of the SSS and SCS arrays corresponds to a vector
def subbases(fw):
    Q_bar = Qbar_mat(fw)
    U, sigma, Vt = np.linalg.svd(Q_bar)
    mask = (np.isclose(sigma, 0))
    # extra vectors have singular value 0?
    Vt_red = Vt[:len(mask)]
    Vt_0 = Vt[len(mask):]
    SSS = np.concatenate((Vt_red[mask], Vt_0))
    SCS = Vt_red[~mask]
    return SSS, SCS

# An attempt at calculating the discrete green's
def G_f(fw, SCS=None):
    if SCS is None:
        _, SCS = subbases(fw)
    k = fw.graph["k"]
    green = (1/k)*np.sum(np.array([np.outer(c, c) for c in SCS]),axis=0)
    return green

# function to stop me doing this all the time
def get_edge_dict(fw):
    es = fw.edges
    return {edge: i for i, edge in enumerate(fw.edges)}

# unique SCS bond
def calc_Ci(fw, edge, SCS=None, G=None):
    if SCS is None:
        _, SCS = subbases(fw)
    if G is None:
        G = G_f(fw, SCS)
    k = fw.graph["k"]
    edge_dict = get_edge_dict(fw)
    index = edge_dict[edge]
    edge_vec = np.zeros(len(fw.edges))
    edge_vec[index] = 1
    C = k * G.dot(edge_vec)
    return C

# Unique SSS bond
def calc_Si(fw, edge, SSS=None):
    if SSS is None:
        SSS, _ = subbases(fw)
    edge_dict = get_edge_dict(fw)
    index = edge_dict[edge]
    edge_vec = np.zeros(len(fw.edges))
    edge_vec[index] = 1
    Si = np.zeros(len(fw.edges))
    for s in SSS:
        Si += np.outer(s,s) @ edge_vec
    return Si

# unique SSS bond modeled after calc_true_cost, i.e. remove ghost edges
def calc_true_Si(fw_orig, edge):
    fw = fw_orig.copy()
    rem_edges = []
    for edge in fw.edges:
        if fw.edges[edge]["lam"] == 0:
            rem_edges.append(edge)
    fw.remove_edges_from(rem_edges)
    SSS, _ = subbases(fw)
    edge_dict = get_edge_dict(fw)
    index = edge_dict[edge]
    edge_vec = np.zeros(len(fw.edges))
    edge_vec[index] = 1
    Si = np.zeros(len(fw.edges))
    for s in SSS:
        Si += np.outer(s,s) @ edge_vec
    return Si

# Code for gram_schmidt modified to allow a starting basis from:
# https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7#gistcomment-1871542
def gram_schmidt(vectors, start=None):
    if not start is None:
        basis = list(start)
    else:
        basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)



# get a basis for the set V from 'modifying multiply bonds'
def V_basis(fw, rem_edges=None, SCS=None, G=None):
    if SCS is None:
         _, SCS = subbases(fw)
    # if we haven't passed in a green's function
    if G is None:
        G = G_f(fw, SCS)
    # first we find all the ghost bonds, and the one we want to change (if it exists)
    if not rem_edges is None:
        B = rem_edges
    else:
        B = []

    es = fw.edges
    for edge in es:
        if es[edge]["lam"] == 0:
            B.append(edge)
    # now we find V, which is all their unique SCS vectors normalised
    V = []
    for edge in B:
        Ci = calc_Ci(fw, edge, SCS, G)
        V.append(Ci/np.sqrt(np.inner(Ci, Ci)))
    V_orthonorm = gram_schmidt(V, start=None)
    return V_orthonorm

# tilde Kaa', which is matrix elements
def Kaabar_mat(fw, V, Fbar_inv=None):
    if Fbar_inv is None:
        Fbar_inv = np.linalg.pinv(Fbar_mat(fw))
    Kaabar = np.zeros((len(V), len(V)))
    for i, a in enumerate(V):
        ca = a.reshape(-1,1)
        for j, ap in enumerate(V):
            cap = ap.reshape(-1,1)
            Kaabar[i,j] = ap.T @ Fbar_inv @ a
    return Kaabar

# inverse Kaa'bar matrix for S19 and S20
# NOTE: should be invertible so we use inv so that it throws an error if not
def Kaabar_inv_mat(fw, V, Fbar_inv=None):
    return np.linalg.inv(Kaabar_mat(fw, V, Fbar_inv))

# can we simply apply gram-schmidt to get a rotated ortho basis wrt V?
# these should be the tilde ca s from 'modifying multiple bonds'
def calc_rot_basis(fw, V, SCS=None):
    if SCS is None:
        _, SCS = subbases(fw)
    basis = gram_schmidt(SCS, V)
    return basis

# get the rotation matrix required to rotate a to point in the direction of b
# NOTE: assumes a and b are the same length (as in same dimension, not same mag)
def get_rot_mat(a,b):
    if len(a) != len(b):
        print("UHOH vectors aren't same length")
    n = len(a)
    a_hat = a/np.linalg.norm(a)
    b_hat = b/np.linalg.norm(b)
    basis = scipy.linalg.orth(np.stack((a_hat,b_hat),axis=1))
    u = basis[:,0]
    v = basis[:,1]
    cost= a_hat.dot(b_hat)
    sint = np.sin(np.arccos(cost))
    if cost == 1:
        print("already parallel")
    if cost == -1:
        print("anti parallel")
    # sint = np.linalg.norm(np.cross(u,v))
    # from stack overflow: https://math.stackexchange.com/questions/197772/generalized-rotation-matrix-in-n-dimensional-space-around-n-2-unit-vector#comment453048_197778
    A = np.eye(n) + sint*(np.outer(v,u) - np.outer(u,v)) + (cost -1)*(np.outer(u,u) + np.outer(v,v))
    return A

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
def all_extensions(fw, tstar, H=None, Hinv=None):
    exts = []
    # for brute-force
    if Hinv is None:
        R = rig_mat(fw,2)
        Rt = R.T
        for edge in fw.edges:
            if fw.edges[edge]["lam"] != 0:
                fwc = fw.copy()  
                fwc.edges[edge]["lam"] = 0
                F = flex_mat(fwc)
                Finv = np.linalg.pinv(F, hermitian=True)
                H = Rt.dot(Finv).dot(R)
                Hinv = np.linalg.pinv(H, hermitian=True)

                ext = R.dot(Hinv.dot(Rt).dot(tstar))
                # TEST NOTE: trying to rescale extensions
#                 edge_list = list(fw.edges)
#                 for i in range(len(edge_list)):
#                     ext[i] = ext[i] / fw.edges[edge_list[i]]["length"]
                exts.append(ext)
            else:
                exts.append(None)
    # with SM updating
    else:
        Q = rig_mat(fw).T
        # scaling extensions back
        for edge in fw.edges:
            if fw.edges[edge]["lam"] != 0:
                H_new, Hinv_bar = update_Hinv(fw, edge,H, Hinv)
                # if not (np.allclose(H_new, H_new @ Hinv_bar @ H_new)):
                    # print("FAILED IN ALL EXTS")

                # calculating in scaled, then rescaling for calculating cost
                ext = Q.T @ Hinv_bar @ Q@ np.array(tstar)
                # TEST NOTE: trying to rescale extensions
#                 edge_list = list(fw.edges)
#                 for i in range(len(edge_list)):
#                     ext[i] = ext[i] / fw.edges[edge_list[i]]["length"]
                # NOTE: checking if the removed edge introduces zero modes
                exts.append(ext)
            else:
                # if we don't consider this edge, just say None
                exts.append(None)
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
            if np.isnan(cost):
                print("cost nan:",nj, njstar)
    return cost

# drawing the strains resulting from applied forces
def draw_strains_from_forces(fw, f, filename=None):
    disps = displacements(fw, None, f)
    exts = extensions(fw, None, disps)
    strs = strains(fw, None, exts)
    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10), frameon=False)
    ax.set_aspect('equal')
    ax.axis('off')
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}
    # SETTING AXIS LIMITS
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for val in pos.values():
        minx = min(minx, val[0])
        miny = min(miny, val[1])
        maxx = max(maxx, val[0])
        maxy = max(maxy, val[1])
    xbuf = (maxx - minx) / 10
    ybuf = (maxy - miny) / 10
    buf = min(xbuf, ybuf)
    plt.xlim(minx - buf, maxx + buf)
    plt.ylim(miny - buf, maxy + buf)
    # END SETTING AXIS LIMITS

    e_labels=dict()
    for i, edge in enumerate(fw.edges):
        e_labels[edge] = np.round(strs[i], 1)

    cmap = plt.cm.coolwarm
    nx.draw_networkx_edges(fw, pos, edgelist=fw.edges, edge_color=strs,
            width=4, edge_cmap=cmap, edge_vmin=-max(abs(strs)), edge_vmax=max(abs(strs)), with_labels=True)
       
    # bbox = dict(boxstyle="round", alpha=0.0)
    nx.draw_networkx_edge_labels(fw, pos, e_labels, font_size=10)#, bbox=bbox)
    # places where the force is being applied will be coloured green
    applied_forces = dict()
    for i in range(0, len(f), 2):
        # if the force is non-zero 
        if (f[i] != 0 or f[i+1] != 0):
            # put the force into a dictionary
            mag = np.sqrt(f[i]**2 + f[i+1]**2)
            applied_forces[i//2] = (f[i]/mag, f[i+1]/mag)
    for key in applied_forces.keys():
        x, y = fw.nodes[key]["position"]
        dx = applied_forces[key][0]*buf
        dy = applied_forces[key][1]*buf
        # plt.plot([x,x+applied_forces[key][0]*buf], [y,y+applied_forces[key][1]*buf], color='k', linestyle='-', linewidth=2)
        plt.arrow(x, y, dx, dy, width=0.1*buf)


    if not filename is None:
        fig.savefig(filename, bbox_inches='tight')
        
    plt.show()
 

# borrowing heavily from draw_tensions, trying to draw the strains on the bonds
def draw_strains(fw, strs, source=None, target=None, ghost=False, filename=None):
    # works on a numpy array
    strs = np.array(strs)
    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_aspect('equal')
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}
    # SETTING AXIS LIMITS
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for val in pos.values():
        minx = min(minx, val[0])
        miny = min(miny, val[1])
        maxx = max(maxx, val[0])
        maxy = max(maxy, val[1])
    xbuf = (maxx - minx) / 10
    ybuf = (maxy - miny) / 10
    plt.xlim(minx - xbuf, maxx + xbuf)
    plt.ylim(miny - ybuf, maxy + ybuf)
    # END SETTING AXIS LIMITS

    e_labels=dict()
    for i, edge in enumerate(fw.edges):
        e_labels[edge] = np.round(strs[i], 1)

    cmap = plt.cm.coolwarm
    nx.draw_networkx_edges(fw, pos, edgelist=fw.edges, edge_color=strs,
            width=4, edge_cmap=cmap, edge_vmin=-max(abs(strs)), edge_vmax=max(abs(strs)), with_labels=True)
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
    strs = exts_to_strains(fw, extensions(fw, tensions))
    ns = [strs[edge_dict[target]] / strs[edge_dict[source]]]
    true_cost = cost_f(ns, nstars)
    return true_cost

# ============================================================================== 
# TUNING FUNCTIONS
# ============================================================================== 

# Run the brute-force tuning algorithm on a network for a given source, target, ratio
def BF_tune_network(fw_orig, source, target, tension=1, nstars=[1.0], cost_thresh=0.01, it_thresh=100, draw=False, verbose=True):
    print("========================================")
    print("========== STARTING BF TUNING ==========")
    print("========================================")
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
    while np.sqrt(min_cost) > cost_thresh and it < it_thresh:
        if verbose:
            print("\n=========== BF TUNING ITERATION",str(it + 1),"============")
        costs = []
        exts_list = all_extensions(fw, tensions)
        for i, exts in enumerate(exts_list):
            if exts is None:
                costs.append(np.inf)
            else:
                strs = exts_to_strains(fw, exts)
                ns = [strs[edge_dict[target]] / strs[edge_dict[source]]]
                costs.append(cost_f(ns, nstars))
        # making sure the edge doesn't introduce a zero mode
        edge_passed = False
        while not edge_passed:
            smalls = sorted(costs)[:10]
            smalls_edges = [list(fw.edges)[costs.index(x)] for x in smalls]
            print("10 smallest costs:", smalls)
            print("10 smallest cost edges (ordered):", smalls_edges)
            min_cost = min(costs)
            index_to_remove = costs.index(min_cost)
            edge_to_remove = list(fw.edges)[index_to_remove]
            # NOTE: checking if introduces a zero mode
            Si = calc_true_Si(fw, edge_to_remove)
            Si_sq = np.inner(Si, Si)
            if np.isclose(Si_sq, 0):
                costs[index_to_remove] = np.inf
                print("ZERO MODE INTRODUCED:", edge_to_remove)
            else:
                edge_passed = True

        fw.edges[edge_to_remove]["lam"] = 0
        # test to see numerical error
        true_cost = calc_true_cost(fw, source, target, nstars, tension)

        if verbose:
            # print("iteration:",it,"cost:",min_cost,"removed:",edge_to_remove)
            print("cost:",min_cost,"removed:",edge_to_remove)
            print("true cost:",true_cost)
            print("relative percentage error:",100*abs(true_cost - min_cost) / true_cost)
        it+=1

    return fw
# Run the tuning algorithm on a network for a given source, target, ratio USING SHERMAN-MORRISON UPDATING
def SM_tune_network(fw_orig, source, target, tension=1, nstars=[1.0], cost_thresh=0.01, it_thresh=100, draw=False, verbose=True):
    print("========================================")
    print("========== STARTING SM TUNING ==========")
    print("========================================")
    fw = fw_orig.copy()
    if source not in fw.edges or target not in fw.edges:
        fw.add_edges_from([source, target])
        fw = add_lengths_and_stiffs(fw)

    edge_dict = get_edge_dict(fw)

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
    # starting H inverse
    H = H_mat(fw)
    H_inv = Hinv_mat(fw)
    while np.sqrt(min_cost) > cost_thresh and it < it_thresh:
        if verbose:
            print("\n=========== SM TUNING ITERATION",str(it + 1),"============")
        # starting H inverse
        # NOTE: recalculating before each edge removal
        H = H_mat(fw)
        H_inv = Hinv_mat(fw)
        costs = []
        exts_list = all_extensions(fw, tensions, H, H_inv)
        for i, exts in enumerate(exts_list):
            if exts is None:
                costs.append(np.inf)
            else:
                strs = exts_to_strains(fw, exts)
                ns = [strs[edge_dict[target]] / strs[edge_dict[source]]]
                costs.append(cost_f(ns, nstars))
        # making sure the edge doesn't introduce a zero mode
        edge_passed = False
        while not edge_passed:
            smalls = sorted(costs)[:10]
            smalls_edges = [list(fw.edges)[costs.index(x)] for x in smalls]
            print("10 smallest costs:", smalls)
            print("10 smallest cost edges (ordered):", smalls_edges)
            min_cost = min(costs)
            index_to_remove = costs.index(min_cost)
            edge_to_remove = list(fw.edges)[index_to_remove]
            # NOTE: checking if introduces a zero mode
            Si = calc_true_Si(fw, edge_to_remove)
            Si_sq = np.inner(Si, Si)
            if np.isclose(Si_sq, 0):
                costs[index_to_remove] = np.inf
                print("ZERO MODE INTRODUCED:", edge_to_remove)
            else:
                edge_passed = True

        # update Hinv with the edge chosen
        H, H_inv = update_Hinv(fw, edge_to_remove,H, H_inv)
        # ok_(np.allclose(Hbar, Hbar @ Hbar_inv @ Hbar))

        fw.edges[edge_to_remove]["lam"] = 0
        # test to see numerical error
        true_cost = calc_true_cost(fw, source, target, nstars, tension)

        if verbose:
            # print("iteration:",it,"cost:",min_cost,"removed:",edge_to_remove)
            print("cost:",min_cost,"removed:",edge_to_remove)
            print("true cost:",true_cost)
            print("relative percentage error:",100*abs(true_cost - min_cost) / true_cost)
        it+=1


    return fw

# ============================================================================== 
#  GF TUNING FUNCTIONS
# ============================================================================== 

def calc_starting_exts(fw, tstar, V):
    SSS, SCS = subbases(fw)
    rot_basis = calc_rot_basis(fw, V, SCS)

    k = fw.graph["k"]

    starting_exts = np.zeros((len(fw.edges)))
    Kaabar_inv = Kaabar_inv_mat(fw, V)
    for i, a in enumerate(V):
        ca = a.reshape(-1,1)
        for j, ap in enumerate(V):
            cap = ap.reshape(-1,1)
            starting_exts += ca @ (Kaabar_inv[i,j] * cap.T) @ tstar
    for c in rot_basis:
        if not c in V:
            starting_exts += (1/k) * np.outer(c,c) @ tstar

    return starting_exts

# calculate the change when changing one bond to zero stiffness
def change_in_exts(fw, edge, tensions, SCS=None, G=None, new_stiff=0, Fbar_inv=None):
    if SCS is None:
        _, SCS = subbases(fw)
    if G is None:
        G = G_f(fw, SCS)
    V = V_basis(fw, rem_edges=[edge], SCS=SCS, G=G)
    change = np.zeros((len(fw.edges)))
    # calculate change in Kaa' so we can calculate change in exts
    # first we modify Fbar to set edge stiffness to 0
    if Fbar_inv is None:
        Fbar_inv = np.linalg.pinv(Fbar_mat(fw))
    new_Fbar_inv = Fbar_inv.copy()
    edge_dict = get_edge_dict(fw)
    i = edge_dict[edge]
    new_Fbar_inv[i,i] = new_stiff
    old_Kaabar_inv = Kaabar_inv_mat(fw, V, Fbar_inv)
    new_Kaabar_inv = Kaabar_inv_mat(fw, V, new_Fbar_inv)
    del_Kaabar_inv = new_Kaabar_inv - old_Kaabar_inv

    for i, a in enumerate(V):
        ca = a.reshape(-1,1)
        for j, ap in enumerate(V):
            cap = ap.reshape(-1,1)
            change += ca @ (del_Kaabar_inv[i,j] * cap.T) @ tensions
    return change

# perform one step of the algorithm
def GF_one_step(fw, source, target, tension=1, nstars=[1.0]):
    edge_dict = get_edge_dict(fw)
    source_i, target_i = edge_dict[source], edge_dict[target]
    # tension on source bond
    tensions = np.zeros(len(fw.edges))
    tensions[source_i] = 1

    SSS, SCS = subbases(fw)
    G = G_f(fw, SCS)
    Fhalf = Fhalf_mat(fw)
    V = V_basis(fw, rem_edges=None, SCS=SCS, G=G)
    starting_exts = calc_starting_exts(fw, tensions, V)
    starting_strs = strains(fw, None, Fhalf @ starting_exts)

    starting_ns = [starting_strs[target_i] / starting_strs[source_i]]
    cost = cost_f(starting_ns, nstars)

    # for each edge, calculate the change in extensions
    k = fw.graph["k"]
    cost_list = []
    old_Fbar_inv = np.linalg.pinv(Fbar_mat(fw))
    for i, edge in enumerate(fw.edges):
        # we ignore ghost edges as we can't double-remove them
        if fw.edges[edge]["lam"] == 0:
            cost_list.append(np.inf)
        else:
            change = change_in_exts(fw, edge, tensions, SCS, G)
            # combine change with starting extensions and get strains
            new_exts = starting_exts + change
            new_strs = strains(fw, None, Fhalf @ new_exts)

            ns = [new_strs[target_i] / new_strs[source_i]]
            cost = cost_f(ns, nstars)
            cost_list.append(cost)

    # making sure the edge doesn't introduce a zero mode
    edge_passed = False
    while not edge_passed:
        smalls = sorted(cost_list)[:10]
        smalls_edges = [list(fw.edges)[cost_list.index(x)] for x in smalls]
        print("10 smallest costs:", smalls)
        print("10 smallest cost edges (ordered):", smalls_edges)
        min_cost = min(cost_list)
        index_to_remove = cost_list.index(min_cost)
        edge_to_remove = list(fw.edges)[index_to_remove]
        # NOTE: checking if introduces a zero mode
        Si = calc_true_Si(fw, edge_to_remove)
        Si_sq = np.inner(Si, Si)
        if np.isclose(Si_sq, 0):
            cost_list[index_to_remove] = np.inf
            print("ZERO MODE INTRODUCED:", edge_to_remove)
        else:
            edge_passed = True
    fw.remove_edges_from([edge_to_remove])

    true_cost = calc_true_cost(fw, source, target, nstars, tension=1)
    return fw, min_cost, true_cost, edge_to_remove
 
# tune using the green's function based algorithm from the paper
def GF_tune_network(fw_orig, source, target, tension=1, nstars=[1.0], cost_thresh=0.01, it_thresh=100):
    print("========================================")
    print("========== STARTING GF TUNING ==========")
    print("========================================")
    # copying for tuning 
    fw = fw_orig.copy()
    # establishing source and target
    fw.add_edges_from([source, target])
    fw = add_lengths(fw)
    edge_dict = get_edge_dict(fw)
    fw.edges[source]["lam"] = 0
    fw.edges[target]["lam"] = 0
    # copy for returning
    rv = fw.copy()
    min_cost = np.inf
    removed_edges = []
    it = 1
    while np.sqrt(min_cost) > cost_thresh and it < it_thresh:
        print("\n============== GF TUNING ITERATION",it, "===============")
        fw, min_cost, true_cost, edge_to_remove = GF_one_step(fw, source, target, tension=1, nstars=nstars)
        removed_edges.append(edge_to_remove)
        print("cost:",min_cost,"removed:",edge_to_remove)
        print("true cost:",true_cost)
        print("relative percentage error:",100*abs(true_cost - min_cost) / true_cost)
        it += 1

    for edge in removed_edges:
        rv.edges[edge]["lam"] = 0
    return rv



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
    index = edge_dict[edge]
    # TEST NOTE: Rescaled extensions here
#     return exts[index]/(fw.edges[edge]["length"]**2) - val
    strs = exts_to_strains(fw, exts)
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

# animate the strain on the network and save to a folder, plus log the graph and displacements
def animate(fw, source, target, nstars, fileroot=None, s_max=1, tensions=1):
    # NOTE: ALWAYS SAVING IMAGE
    if fileroot is None:
        fileroot = "images/"+str(datetime.date.today()) +"/"+ str(int(time.time())) + "/"
        os.makedirs(fileroot)
    else: 
        from pathlib import Path
        Path(fileroot).mkdir(parents=True, exist_ok=True)

    # # number of frames in the animation
    n = 30
    real_ratios_list = []
    edge_dict = {edge: i for i, edge in enumerate(fw.edges)}
    tensions = [0]*len(fw.edges)
    tensions[edge_dict[source]] = 1
    u0 = np.zeros(len(fw.nodes) * 2)
    for i in range(n):
        strain_val = 0.2 *s_max * (i/(n-1))
        print("=========== ANIMATE ITERATION",str(i + 1),"============")
        print("source strain val constraint:",strain_val)
        constraints = {"type":"eq", "fun":source_strain, "args":(fw, source, strain_val)}
        mind = minimize(energy, u0, args=(fw), constraints=constraints)
        print("minimized energy, success, and # iterations:",mind.fun, mind.success,mind.nit)
        # using solution of prev it for start of next
        u0 = mind.x
        real_exts = rig_mat(fw).dot(u0)
        real_strains = exts_to_strains(fw, real_exts)
    # adjusting for proper strain calculation
        real_target_strain = real_strains[edge_dict[target]] #/ fw.edges[target]["length"]
        real_source_strain = real_strains[edge_dict[source]] #/ fw.edges[source]["length"]
        print("target, source strain:",real_target_strain, real_source_strain)
        real_ratio = real_target_strain/real_source_strain
        if not np.isnan(real_ratio):
            real_ratios_list.append(real_ratio)
        print("full nonlinear strain ratio:", real_ratio)
        draw_framework(update_pos(fw, mind.x), filename=fileroot+"anim_"+str(i)+".png",ghost=True, source=source, target=target)
        plt.close()
        print("drawn",i+1,"images of",n)
    # logging the positions
        with open(fileroot+'log.log', 'a') as f:
              np.savetxt(f,u0)
              f.write(',')

    # save the edges of the graph (positions are written in the log)
    with open(fileroot+'graph.log', 'w') as f:
          f.write(str(source)+"\n")
          f.write(str(target)+"\n")
          f.write(str(nstars)+"\n")
          np.savetxt(f,fw.edges)
          f.write(",\n")
          f.write(str(fw.nodes(data=True)))

    fig = plt.figure()
    plt.plot(real_ratios_list)
    plt.axhline(nstars[0],linestyle="--")
    plt.show()
    fig.savefig(fileroot+"ratio.png",bbox_inches="tight")
    return real_ratios_list

# read the displacements from my poorly formatted log file
def read_disps(filename):
    disps_list = [[]]
    with open(filename) as fp:
        i = 0 
        for line in fp:
            # if this is the start of a new iteration:
            if line[0] == ",":
                # if it's not just a comma
                if len(line) > 1:
                    i += 1
                    # add a new list to hold a vector of positions
                    disps_list.append([])
                    # drop the comma
                    line = line[1:]
                # if it's ajust a comma
                else:
                    continue

            disps_list[i].append(float(line))
    return np.array(disps_list)

# read the poorly formatted graph log file
def read_graphlog(filename):
    from ast import literal_eval
    with open(filename) as fp:
        edge_list = []
        comma = False
        for i, line in enumerate(fp):
            if i == 0 :
                source = literal_eval(line) 
            elif i == 1:
                target = literal_eval(line) 
            elif i ==2:
                nstars = literal_eval(line)
            else:
                if line == ',\n':
                    comma = True
                else:
                    if not comma:
                        edge_list.append(tuple(int(float(x)) for x in line.split(" ")))
                    else:
                        pos = literal_eval(line)
        fw = nx.Graph()
        fw.add_nodes_from(pos)
        fw.add_edges_from(edge_list)
        fw = add_lengths_and_stiffs(fw)
        return fw, source, target, nstars

# reconstruct the optimisation process using just the disps
def ratios_list_from_disps(fw, disps_list, source, target):
    R = rig_mat(fw)
    edge_dict = get_edge_dict(fw)
    ratios_list=[]
    for disps in disps_list:
        exts = R @ disps
        strs = exts_to_strains(fw, exts)
        # adjusting for proper strain calculation
        target_strain = strs[edge_dict[target]] 
        source_strain = strs[edge_dict[source]]
        ratio = target_strain/source_strain
        if not np.isnan(ratio):
            ratios_list.append(ratio)

    return ratios_list

def plot_ratios_list(ratios_list, nstars):
    nstar = nstars[0]
    ymin = min(nstar, min(ratios_list), 0)
    ymax = max(nstar, max(ratios_list), 0)
    ybuf = (ymax - ymin) / 10
    fig = plt.figure()
    plt.ylim(ymin - ybuf, ymax + ybuf)
    plt.plot(ratios_list)
    plt.axhline(nstar,linestyle="--")
    plt.show()
