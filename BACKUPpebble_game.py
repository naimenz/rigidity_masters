import networkx as nx
import heapq as hq

# reverses an edge in an nx digraph
# NOTE: acts in-place
def reverse(G, e):
    e0 = e[0]
    e1 = e[1]
    G.remove_edge(e0, e1)
    G.add_edge(e1, e0)

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
def pebble_game(G, k, l):

    E = G.edges

    # constructing the directed graph D to keep track of progress
    D = nx.DiGraph()
    D.add_nodes_from(G)
    dnodes = D.nodes
    
    # collect rejected edges
    rejected_edges = []
    # adding initial pebbles to each vertex
    for v in dnodes:
        D.nodes[v]["pebbles"] = k

    for e in E:
        u = e[0]
        v = e[1]

        if dnodes[u]["pebbles"] + dnodes[v]["pebbles"]  >= l+1:
            dnodes[u]["pebbles"] -= 1
            D.add_edge(u, v)

        else:
            seen = set([u, v])
            # initialise the heap with neighbours of u and v
            heap = []
            
            # if u doesn't have enough pebbles:
            if dnodes[u]["pebbles"] < l + 1:
                seen = set([u, v])
                # initialise the heap with neighbours of u and v
                heap = []

                successors = D.successors(u)
                for s in successors:
                    hq.heappush(heap, (0, s, [u])) 

                while len(heap)>0 and dnodes[u]["pebbles"] < l + 1:
                    count, vertex, path = hq.heappop(heap)
                    path.append(vertex)
                    seen.add(vertex)
                    if dnodes[vertex]["pebbles"] > 0 and dnodes[u]["pebbles"] < k:
                        dnodes[vertex]["pebbles"] -= 1
                        dnodes[u]["pebbles"] += 1

                        for i in range(len(path)-1):
                            reverse(D, (path[i], path[i+1]))                    

                    else:
                        successors = D.successors(vertex)
                        for s in successors:
                            hq.heappush(heap, (count-1, s, path))

            # if u still doesn't have enough pebbles, reject the edge
            if dnodes[u]["pebbles"] < l+1:
                rejected_edges.append(e)
                continue

            # if v doesn't have enough pebbles:
            if dnodes[v]["pebbles"] < l + 1:
                seen = set([u, v])
                # initialise the heap with neighbours of u and v
                heap = []

                successors = D.successors(v)
                for s in successors:
                    hq.heappush(heap, (0, s, [v])) 

                while len(heap)>0 and dnodes[v]["pebbles"] < l + 1:
                    count, vertex, path = hq.heappop(heap)
                    path.append(vertex)
                    seen.add(vertex)
                    if dnodes[vertex]["pebbles"] > 0 and dnodes[v]["pebbles"] < k:
                        dnodes[vertex]["pebbles"] -= 1
                        dnodes[v]["pebbles"] += 1

                        for i in range(len(path)-1):
                            reverse(D, (path[i], path[i+1]))                    

                    else:
                        successors = D.successors(vertex)
                        for s in successors:
                            hq.heappush(heap, (count-1, s, path))

            # if v still doesn't have enough pebbles, reject the edge
            if dnodes[v]["pebbles"]  < l+1:
                rejected_edges.append(e)
                continue

            else:
                print(D.nodes(data=True))
                dnodes[u]["pebbles"] -= 1
                print(D.nodes(data=True))
                D.add_edge(u, v)

    total_pebs = 0
    for v in dnodes:
        total_pebs += dnodes[v]["pebbles"]

    print(total_pebs, rejected_edges)
    if total_pebs == l:
        if len(rejected_edges) == 0:
            return 0

        else:
            return 1

    else:
        if len(rejected_edges) == 0:
            return 2

        else:
            return 3

g = nx.Graph()
g.add_nodes_from([0,1,2,3])
g.add_edges_from([(0,1), (1,2), (0,2), (0,3), (1,3)])

p = pebble_game(g, 2, 3)
print(p)
