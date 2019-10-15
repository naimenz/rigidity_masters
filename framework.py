import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as ln
import poisson as pd
from scipy.spatial import Delaunay

# simple function wrapping graph creation in nx
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

# takes a set of nodes, edges, and positions and create a framework
def create_framework(nodes, edges, positions):
    g = create_structure_graph(nodes, edges)
    fw = assign_pos(g, positions)

    return fw

# convert a delaunay object into a list of edges that can be used to create a framework
def delaunay_to_edges(d):
    rv = []
    for tri in d.simplices:
        rv.append((tri[0], tri[1]))
        rv.append((tri[1], tri[2]))
        rv.append((tri[0], tri[2]))
    return rv

# creates a random framework in a grid of (width x height) with min dist r
# bottom left corner on (0,0). Edges are from the delaunay triangulation
def create_random_fw(w, h, r):
    positions = pd.poisson_disc_sample(w, h, r)
    nodes = list(range(len(positions)))
    edges = delaunay_to_edges(Delaunay(positions))
    fw = create_framework(nodes, edges, positions)
    return fw

# creates a random framework and then removes edges until 
# (still in 2D)
def create_reduced_fw(w,h,r):
    fw = create_random_fw(w,h,r)
    while len(fw.edges) > 2*len(fw.nodes):
            index = np.random.choice(len(fw.edges))
            edge = list(fw.edges)[index]
            if fw.degree(edge[0]) > 2 and fw.degree(edge[1]) > 2:
                fw.remove_edge(edge[0], edge[1])
    return fw

def draw_framework(fw):
    nodeview = fw.nodes
    plt.figure(figsize=(20,10))
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw_networkx_nodes(fw, pos, with_labels=True)
    nx.draw_networkx_edges(fw, pos, with_labels=True)
    nx.draw_networkx_labels(fw,pos)
    plt.show()

# draws the components 'comps' of the framework 'fw'
# trivial components have their edges drawn in grey
# edges in non-trivial components are solid black
# vertices in one component are red
# vertices in multiple components are green
def draw_comps(fw, comps):
    big_comps = [comp for comp in comps if len(comp) > 2]  
    # calculating which vertices are in multiple components
    greens = set()
    for A in big_comps:
        for B in big_comps:
            if A != B:
                greens |= A & B

    # drawing the nodes of the graph
    plt.figure(figsize=(20,10))
    nodeview = fw.nodes
    reds = set(nodeview) - greens
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw_networkx_nodes(fw, pos, nodelist = list(greens), node_color='g')
    nx.draw_networkx_nodes(fw, pos, nodelist = list(reds), node_color='r')
    nx.draw_networkx_edges(fw, pos, alpha=0.2, width=1)
    nx.draw_networkx_labels(fw,pos)

    for comp in big_comps:
        nx.draw_networkx_edges(fw.subgraph(list(comp)), pos, width=3)
    plt.show()

# creates the rigidity matrix for a d-dimensional framework
# takes in a framework (nx graph with positions) and returns a numpy array
def create_rigidity_matrix(fw, d):
    edgeview = fw.edges
    nodeview = fw.nodes
    n = len(list(fw))
    e = len(edgeview)
    M = np.zeros((e, d*n))
        
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

# function to calculate the nullity (and hence dimension of space of inf. rigid motions) of R
def nullity(R):
    n = R.shape[1]
    return n - ln.matrix_rank(R)

# calculate the dimension of the affine span of vectors
# works by shifting v0 to the origin and calculating the span of that set of vectors
def calc_affine_span_dim(vectors):
    v0 = vectors[0] 
    new_vs = np.array([v - v0 for v in vectors])
    return ln.matrix_rank(new_vs)

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
    R = create_rigidity_matrix(fw, d)
    # print("d=",d," - ",ln.matrix_rank(R) ,  d*size_V - (((d+1) * d) / 2))
    return ln.matrix_rank(R) == d*size_V - (((d+1) * d) / 2)

