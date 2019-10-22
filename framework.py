import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as ln
import poisson_disc_sample as pd
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

# creates a random framework with n points and minimum distance r, starting at (0,0).
# Edges are from the delaunay triangulation
def create_random_fw(n, r, seed=None):
    positions = pd.poisson_disc_sample(n, r, seed)
    nodes = list(range(len(positions)))
    edges = delaunay_to_edges(Delaunay(positions))
    fw = create_framework(nodes, edges, positions)
    return fw

# creates a random framework and then removes edges until 
# (still in 2D)
def create_reduced_fw(n, r, seed=None):
    fw = create_random_fw(n, r, seed)
    while len(fw.edges) > 2*len(fw.nodes):
            index = np.random.choice(len(fw.edges))
            edge = list(fw.edges)[index]
            if fw.degree(edge[0]) > 2 and fw.degree(edge[1]) > 2:
                fw.remove_edge(edge[0], edge[1])
    return fw

def draw_framework(fw, filename=None):
    nodeview = fw.nodes
    fig = plt.figure(figsize=(20,10))
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw_networkx_nodes(fw, pos, with_labels=True)
    nx.draw_networkx_edges(fw, pos, with_labels=True)
    nx.draw_networkx_labels(fw,pos)
    if filename:
        fig.savefig(filename, bbox_inches='tight')
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
    fig = plt.figure(figsize=(20,10))
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


# creates the rigidity matrix for a d-dimensional framework
# takes in a framework (nx graph with positions) and returns a numpy array
def create_rigidity_matrix(fw, d):
    edgeview = fw.edges
    nodeview = fw.nodes
    n = len(list(fw))
    e = len(edgeview)
    M = np.zeros((e, d*n))
        
    for row, edge in enumerate(sorted(edgeview)):
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

# draws the framework 'fw' with stresses resulting from force 'f'
def draw_stresses(fw, f):
    R = create_rigidity_matrix(fw, 2)
    s = R.dot(f)
    # drawing the nodes of the graph
    fig = plt.figure(figsize=(8,8))
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}

    e_labels=dict()
    for i, edge in enumerate(sorted(fw.edges)):
        e_labels[edge] = np.round(s[i], 4)

    # places where the force is being applied will be coloured green
    applied_forces = dict()
    for i in range(len(f)):
        # if the force is non-zero and this is the x coord of it
        if f[i] != 0 and i%2 == 0:
            # put the force into a dictionary
            applied_forces[i//2] = (f[i], f[i+1])
       
    # nx.draw_networkx_labels(fw, pos)
    # nx.draw_networkx_edges(fw, pos, color=s, cmap=plt.cm.plasma, width=5)
    nx.draw(fw, pos, edge_color=s,
            width=4, edge_cmap=plt.cm.coolwarm, with_labels=True)
    nx.draw_networkx_nodes(fw, pos, nodelist=applied_forces.keys(), node_color='green')
    nx.draw_networkx_edge_labels(fw, pos, e_labels)
    # nx.draw_networkx_nodes(fw, pos, node_color='r')
    for key in applied_forces.keys():
        x, y = fw.nodes[key]["position"]
        plt.plot([x,x+applied_forces[key][0]], [y,y+applied_forces[key][1]], color='k', linestyle='-', linewidth=4)

    plt.show()
