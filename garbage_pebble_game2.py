import networkx as nx
from framework import *
import matplotlib.pyplot as plt
import numpy as np
import heapq as hq

# reverses an edge in an nx digraph
# NOTE: acts in-place
def reverse(G, e):
    e0 = e[0]
    e1 = e[1]
    G.remove_edge(e0, e1)
    G.add_edge(e1, e0)

# detect the components using a directed graph D into which edge e has just been inserted
# returns [] if no component, and otherwise returns the component as a list
# l is a parameter from the pebble game
def detect_comps(D, e, l):
    v1 = e[0]
    v2 = e[1]

    # first we check if v1 and v2 have more than l pebbles
    dnodes = D.nodes(data=True)
    node_list = list(D)
    if dnodes[v1]["pebbles"] + dnodes[v2]["pebbles"] > l:
        return []

    # next we do a depth-first search to find a pebble in the reach of v1 and v2
    # and keep track of all elements in the reach
    heap = []
    reach = set((v1, v2))

    for s in D.successors(v1):
        if s not in reach:
            reach.add(s)
            hq.heappush(heap, (0, s))

    for s in D.successors(v2):
        if s not in reach:
            reach.add(s)
            hq.heappush(heap, (0, s))

    while len(heap) > 0:
        count, w = hq.heappop(heap)
        # if at least one free pebble on w, no componenet
        if dnodes[w]["pebbles"] > 0:
            return []

        reach.add(w)
        # add all new points to the heap
        for s in D.successors(w):
            if s not in reach:
                reach.add(s)
                hq.heappush(heap, (count - 1, s))
    
    # if this part completes, we now construct Dprime (reverse all edges) and search from the vertices
    # with at least one pebble that AREN'T in reach
    # yet another depth-first search to find all vertices visited from the starters
    starters = [w for w in node_list if (w not in reach and dnodes[w]["pebbles"] > 0)]
    visited = set(starters)
    # a copy of D with edges reversed
    Dprime = D.reverse()

    for w in starters:
        heap = []
        hq.heappush(heap, (0, w))

        for s in Dprime.successors(w):
            if s not in visited:
                visited.add(s)
                hq.heappush(heap, (0, s))

        while len(heap) > 0:
            count, vertex = hq.heappop(heap)

            for s in Dprime.successors(vertex):
                if s not in visited:
                    visited.add(s)
                    hq.heappush(heap, (count-1, s))

    return frozenset(set(node_list) - visited)

# takes a pebble game directed graph D, an edge e, the parameter l from the game, a list of rigid components 
# so far, and a matrix of elements in the same rigid component.

# NOTE returns the new comp_list, but modifies comp_matrix in place
global_counter = 0

def update_components(D, e, l, comp_list, comp_matrix):
    global global_counter
    new_comp = detect_comps(D, e, l)
    # keep track of which components are subsumed by the new component
    # if they intersect with more than one element then the old one must be contained in the new
    subsumed_comps = [comp for comp in comp_list if len(comp.intersection(new_comp)) > 1]
    
    # vbar keeps track of union of all subsumed comps
    Vbar = set()

    # we now loop over all pairs of subsumed components, adding them to the matrix
    for i in range(len(subsumed_comps)):
        Vbar = Vbar.union(subsumed_comps[i])
        for j in range(len(subsumed_comps)):
            comp1 = subsumed_comps[i] - subsumed_comps[j]
            comp2 = subsumed_comps[j] - subsumed_comps[i]
            for u in tuple(comp1):
                for v in tuple(comp2):
                    global_counter = global_counter + 1
                    comp_matrix[u][v] = 1

    # finally handle all elements not in Vbar
    for u in tuple(new_comp - Vbar):
        for v in tuple(new_comp):
            global_counter = global_counter + 1
            comp_matrix[u][v] = 1

    subsumed_set = set(subsumed_comps)
    rv = [comp for comp in comp_list if comp not in subsumed_set]
    rv.append(new_comp)

    return rv


# Function to run the (k,l) pebble game on a graph (ultimately for rigidity)
# input : nx graph with vertices and edges and k, l parameters (kn - l)
# G: nx graph
# k: number multiplying number of vertices
# l: constant subtracted

# output : a number corresponding to to the result of the algorithm
# 0: well constrained (tight)
# 1: under-constrained (sparse)
# 2: over-constrained (spanning)
# 3: other (neither sparse nor spanning)

# NOTE specifically designed for simple graphs with (k,l)=(2,3), so might not work for all ks and ls
# After each edge is added we track the number of components and plot them using graph H
def constructive_pebble_game(G, k, l):
    E = G.edges

    pos = {node: G.nodes(data=True)[node]["position"] for node in G.nodes}
    H = create_framework(G.nodes, [], pos) 
    # constructing the directed graph D to keep track of progress
    D = nx.DiGraph()
    D.add_nodes_from(G)
    dnodes = D.nodes
    n = len(dnodes)
    
    # collect rejected edges
    rejected_edges = []
    
    # matrix and list to keep track of components
    comp_matrix = np.zeros((n, n))
    comp_list = []
    # adding initial pebbles to each vertex
    for v in dnodes:
        D.nodes[v]["pebbles"] = k

    for e in E:
        H.add_edges_from([e])

        print(H.nodes(data=True))
        print(H.edges)
        draw_comps(H, comp_list, filename="component_images/comps", show=False)
        u = e[0]
        v = e[1]
        # first check if they are in the same component, and if so, reject
        if comp_matrix[u][v] == 1:
            rejected_edges.append(e)

        elif dnodes[u]["pebbles"] + dnodes[v]["pebbles"] >= l+1:
            if dnodes[u]["pebbles"] > 0:
                dnodes[u]["pebbles"] -= 1
                D.add_edge(u, v)
                # detecting components and updating the matrix
                comp_list = update_components(D, e, l, comp_list, comp_matrix)

            else:
                dnodes[v]["pebbles"] -= 1
                D.add_edge(v, u)
                comp_list = update_components(D, e, l, comp_list, comp_matrix)

        else:
            seen = set([u, v])
        
            # initialise the heap with successors of u and v
            heap = []

            for s in D.successors(u):
                if s not in seen:
                    seen.add(s)
                    hq.heappush(heap, (0, s, [u]))

            for s in D.successors(v):
                if s not in seen:
                    seen.add(s)
                    hq.heappush(heap, (0, s, [v]))

            while len(heap)>0 and dnodes[u]["pebbles"] + dnodes[v]["pebbles"] < l + 1:
                # print("heap:",heap)
                count, vertex, path = hq.heappop(heap)

                # extend the path with the current vertex
                path.append(vertex)
                
                # print("current vertex",vertex,D.nodes(data=True)[vertex])

                # if the current vertex has pebbles and u/v has fewer than k, move a pebble from verteex to u/v 
                # and reverse edges, then reset the search
                if dnodes[vertex]["pebbles"] > 0 and dnodes[path[0]]["pebbles"] < k:
                    dnodes[vertex]["pebbles"] -= 1
                    dnodes[path[0]]["pebbles"] += 1

                    for i in range(len(path)-1):
                        reverse(D, (path[i], path[i+1]))                    

                    # NOTE After changing direction of edges, need to reset the search
                    seen = set([u, v])
                
                    # initialise the heap with successors of u and v
                    heap = []

                    for s in D.successors(u):
                        if s not in seen:
                            seen.add(s)
                            hq.heappush(heap, (0, s, [u]))

                    for s in D.successors(v):
                        if s not in seen:
                            seen.add(s)
                            hq.heappush(heap, (0, s, [v]))
                                
                else:
                    successors = D.successors(vertex)
                    for s in successors:
                        if s not in seen:
                            seen.add(s)
                            hq.heappush(heap, (count-1, s, path[:]))


            # if after all the searching u and v now combined have enough pebbles, add the edge
            if dnodes[u]["pebbles"] + dnodes[v]["pebbles"] >= l+1:
                if dnodes[u]["pebbles"] > 0:
                    dnodes[u]["pebbles"] -= 1
                    D.add_edge(u, v)
                    comp_list = update_components(D, e, l, comp_list, comp_matrix)

                else:
                    dnodes[v]["pebbles"] -= 1
                    D.add_edge(v, u)
                    comp_list = update_components(D, e, l, comp_list, comp_matrix)

            else:
                rejected_edges.append(e)


    total_pebs = 0
    for v in dnodes:
        total_pebs += dnodes[v]["pebbles"]

    # determine which case we are in based on the remaining pebbles and the number of rejected edges
    if len(rejected_edges) == 0:
        if total_pebs == l:
            print("tight")
            return 0, comp_list
        else:
            print("sparse")
            return 1, comp_list

    else:
        if total_pebs == l:
            print("spanning")
            return 2, comp_list
        else:
            print("neither")
            return 3, comp_list

# g = nx.Graph()
# # g.add_nodes_from([0,1,2,3,4])
# # g.add_edges_from([(0,1), (1,2), (0,2), (0,3),(0,4), (3,4)])
# # g.add_nodes_from([0,1,2])
# # g.add_edges_from([(0,1), (1,2), (0,2)])

# fig2b = nx.Graph()
# fig2b.add_nodes_from([0,1,2,3,4,5,6,7,8])
# fig2b.add_edges_from([(0,1), (0,7), (1,2), (1,7), (2,3), (2,4), (3,4),
#                         (3,8), (4,5), (5,6), (5,8), (6,7), (6,8), (7,8)])

# bar = nx.Graph()
# bar.add_nodes_from([0,1,2,3,4,5])
# bar.add_edges_from([(0,1), (0,3), (0,4), (1,2), (1,5), (2,3), (2,5), (3,4)])
# # p = pebble_game(bar, 2, 3)

# # nx.draw(fig2b)
# # plt.show()
# # p = pebble_game(fig2b, 2, 3)
# # print(p)
# # print(global_counter)
