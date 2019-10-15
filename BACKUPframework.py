import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as ln
import math

# simple function wrapping graph creation in nx
def create_structure_graph(nodes, edges):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

# add an embedding to a structure graph, creating a framework
# takes a nx graph with n nodes and a list of n positions and modifies the
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

def draw_framework(fw):
    nodeview = fw.nodes
    pos = {node: nodeview[node]["position"] for node in nodeview}
    nx.draw(fw, pos, with_labels=True)
    plt.show()

# calculate the row based on n, i, j defined below
def calc_row(n, i, j):
    rv = 0
    # adding up all rows from previous i values
    for x in range(i):
        rv += n-(x+1)

    # adding offset for current j being considered
    rv += (j-1)-i

    return rv

# creates the rigidity matrix for a d-dimensional framework
# takes in a framework (nx graph with positions) and returns a numpy array

# NOTE just realised could make fw into a complete graph and then calculate every edge like below
# Less elegant but simpler potentially
def create_rigidity_matrix(fw, d):
    # this is a node view (whatever that is)
    nodeview = fw.nodes
    # this is a standard python list
    node_list = list(fw)
    n = len(node_list)
    R = np.zeros((n*(n-1) // 2, d*n))
    
    for i in range(n-1):
        node1 = node_list[i]
        for j in range(i+1, n):
            node2 = node_list[j]
            # e.g. we want to put p1 - p2 in the (1 + 2 - 1) = 1st column
            pos1 = nodeview[node1]["position"]
            pos2 = nodeview[node2]["position"]
            # using above function to work out row number
            row = calc_row(n, i, j)

            if d == 1:
                R[row, d*i] = pos1 - pos2
                R[row, d*j] = -pos1 + pos2
            else:
                for k in range(d):
                    R[row, d*i+k] = pos1[k] - pos2[k]
                    R[row, d*j+k] = -pos1[k] + pos2[k]
    
    return R

# not sure what to call this one, but it's a matrix to represent the linear system whose null space is the
# allowed infinitesimal motions in d-space, one equation for each edge
def create_motions_matrix(fw, d):
    edgeview = fw.edges
    nodeview = fw.nodes
    n = len(list(fw))
    e = len(edgeview)
    M = np.zeros((e, d*n))
        
    row = 0
    for edge in edgeview:
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

        row += 1

    return M


# function to calculate the nullity (and hence dimension of space of inf. rigid motions) of R
def nullity(R):
    n = R.shape[1]
    return n - ln.matrix_rank(R)

# takes a normal (non constricted, i.e. not embedded in lower dimensions) d-space framework
# and returns True if it is inf. rigid, False otherwise
def is_inf_rigid(fw, d):
    R = create_rigidity_matrix(fw, d)
    M = create_motions_matrix(fw, d)
    return nullity(R) == nullity(M)

