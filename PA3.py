import heapq
import itertools
import time


class Steiner: # In order to deal with the tree more conveniently, we declare a class called Steiner.
    def __init__(self, V, E, w): # Record the information of graph.
        self.V = set(V) # Vertices
        self.E = E # Edges

        self.w = w # Weights
        self.w_hash = self.function_w(self.E, self.w) # Make the hash table of edges and weights

        self.E = list(self.w_hash.keys())
        self.w = list(self.w_hash.values())
        self.G = (self.V, self.E) # G = (V, E)

        self.dist, self.next = self.FW_stt_helper()

    # In order to make the traversal more convenient, we treat the undirected graph as "directed graph with the same weight in both directions".
    def function_w(self, E, w):
        w_hash = dict(zip(E, w))
        for u, v in E:
            dummy = w_hash[(u, v)]
            w_hash[(v, u)] = dummy

        for u, v in E:
            assert w_hash[(u, v)] == w_hash[(v, u)]

        return w_hash

    # Calculation of the shortest distance (weight) between two vertices
    def shortest(self, s, t):
        # Initialization
        distances = {u: {v: float('inf') for v in self.V} for u in self.V}

        for v in self.V:
            distances[v][v] = 0 # Record all self-to-self edges as zero-weighted

        for (u, v), w in self.w_hash.items(): # Firstly record directly connected edges
            distances[u][v] = w

        # Replace the weights into smaller indirect weight sums if necessary
        for k in self.V:
            for i in self.V:
                for j in self.V:
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

        return distances[s][t]

    # Floyd Warshall algorithm for assistance of st_t function
    def FW_stt_helper(self):
        n = len(self.V)
        dist = [[float('inf')] * n for _ in range(n)] # Initialization of distances
        next = [[None] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0

        for (i, j) in self.E:
            dist[i][j] = self.w_hash[(i, j)]
            next[i][j] = j

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next[i][j] = next[i][k]

        return dist, next


    # st_2 function, most similar to Dijkstra
    def st_2(self, G):
        start = 1 # One of the vertices is 1
        vertices, edges = G[0], G[1] # G = (V, E)

        # Priority queue (Hash)
        pq = [(0, start, [start])]

        # Record all distances.
        distances = {node: float('inf') for node in vertices}
        distances[start] = 0

        paths = {node: [] for node in vertices} # Record all distances from vertex 1 to others
        paths[start] = [start]

        # Explore all shortest paths
        while pq:
            dist, node, path = heapq.heappop(pq)
            if dist > distances[node]:
                continue

            for edge in edges:
                if edge[0] == node:
                    nb = edge[1]
                    w = self.w_hash[edge]
                    new_dist = dist + w

                    if new_dist < distances[nb]:
                        distances[nb] = new_dist
                        new_path = path + [nb]
                        paths[nb] = new_path
                        heapq.heappush(pq, (new_dist, nb, new_path))

        # Dictionary for the final result of the smallest weight from vertex 1 to each other vertex
        dijkstra_hash = {}
        for node in paths:
            p = paths[node]
            path_list = [] # Record the path
            for i in range(1, len(p)):
                path_list.append((p[i-1], p[i]))

            dijkstra_hash[node] = (distances[node], paths[node], path_list)

        new_V = {1, 999} # n = 999
        new_E = [] # Record all nodes connecting node 1 and node 999
        new_w = 0 # Total weight of the Steiner Tree

        # Edges, vertices, weight
        for path in dijkstra_hash[999][2]:
            v0, v1 = path
            if (v0, v1) not in new_E and (v1, v0) not in new_E:
                new_E.append((v0, v1))
                new_V.add(v0)
                new_V.add(v1)
                new_w += self.w_hash[(v0, v1)]

        # Graph of st_2
        G_prime = (new_V, new_E)
        return G_prime, new_w

    # Similar to MST
    def st_v(self, G):
        V, E = G[0], G[1]
        start = next(iter(V)) # Use the first vertex as the starting point of MST
        MST = []
        visited = {start} # Record the visited vertex/vertices
        pq = [] # Heap / priority queue

        # Push all edges into the heap
        for v in V:
            if (start, v) in self.w_hash:
                heapq.heappush(pq, (self.w_hash[(start, v)], start, v))
            elif (v, start) in self.w_hash:
                heapq.heappush(pq, (self.w_hash[(v, start)], v, start))

        # Traversal until all vertices are visited
        while len(visited) < len(V) and pq:
            w, u, v = heapq.heappop(pq)

            if v in visited:
                continue

            MST.append((u, v, w))
            visited.add(v)
            for nb in V - visited:
                if (v, nb) in self.w_hash:
                    heapq.heappush(pq, (self.w_hash[v, nb], v, nb))
                elif (nb, v) in self.w_hash:
                    heapq.heappush(pq, (self.w_hash[nb, v], nb, v))

        # Get the total weight and make the new graph (tree)
        V_prime, E_prime = self.V, []
        w_prime = 0
        for path in MST:
            E_prime.append((path[0], path[1]))
            w_prime += path[2]

        G_prime = (V_prime, E_prime)
        return G_prime, w_prime

    # Helper function of shortest paths from one vertex to another for st_t
    def shortest_paths(self, G):
        V, E = G[0].copy(), G[1].copy()
        n = len(V)
        d = [[float('inf')] * n for _ in range(n)]

        for i in range(n):
            d[i][i] = 0

        for i, j in E:
            d[i][j] = self.w_hash[(i, j)]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])

        return d

    # Helper function of making MST based on terminal vertices
    def limited_MST(self, T: set):
        if not T:
            return []

        pq = []
        visited = set()
        mst = set()
        parent = {}

        start = list(T)[0]
        visited.add(start)
        dist = self.shortest_paths(self.G)

        for v in list(T)[1:]:
            heapq.heappush(pq, (dist[start][v], v, start))

        while pq:
            d, u, p = heapq.heappop(pq)
            if u not in visited:
                visited.add(u)
                parent[u] = p
                mst.add((p, u))
                for v in T:
                    if v not in visited:
                        heapq.heappush(pq, (dist[u][v], v, u))

        return mst

    '''
    
    The concept in the file is too complex, and we can assume that when the length of T is between 2 and the length of vertex list,
    it can be treated as optimization of an MST based on terminal nodes.
    
    '''
    def st_t(self, G, T: set):
        mst_edges = self.limited_MST(T) # Initialization of the MST based on T
        dist = self.shortest_paths(G) # Record all smallest paths in the original graph

        dummy = True # Flag to control start and stop of the algorithm
        while dummy:
            dummy = False
            paths = list(mst_edges)

            # Optimization of paths
            for p in paths:
                for v in G[0] - T:
                    u1, u2 = p[0], p[1]
                    temp = mst_edges - {(u1, u2)}
                    temp.update({(min(u1, v), max(u1, v)), (min(v, u2), max(v, u2))})

                    nodes_count = {u for edge in temp for u in edge}
                    if len(temp) >= len(nodes_count):
                        continue

                    current = dist[p[0]][p[1]]
                    new = dist[v][u1] + dist[v][u2]

                    if new < current:
                        mst_edges.remove((u1, u2))
                        mst_edges.append((u1, v))
                        mst_edges.append((u2, v))
                        dummy = True
                        break

                if dummy:
                    break

        # Construct the tree
        final = set()
        for (u, v) in mst_edges:
            path = self._path(u, v)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                final.add((a, b)) # Edges

        total = sum(self.w_hash[e] for e in set(final))

        w_prime = total # Weight

        E_prime = list(final)

        V_prime = [] # Vertices
        for pair in final:
            if pair[0] not in V_prime:
                V_prime.append(pair[0])
            if pair[1] not in V_prime:
                V_prime.append(pair[1])

        for edge in E_prime:
            if (edge[1], edge[0]) in E_prime:
                E_prime.remove(edge)
                w_prime -= dist[edge[0]][edge[1]] # Remove (v, u), which is a duplication of (u, v)

        V_prime = set(V_prime)

        G_prime = (V_prime, E_prime)
        return G_prime, w_prime

    # Helper function for st_t
    def _path(self, u, v):
        if not self.next:
            return []
        path = [u]
        while u != v:
            u = self.next[u][v]
            path.append(u)
        return path

    # st_h function, removing all leaf nodes that do not belong to terminal vertices from the MST.
    def st_h(self, G, T: set):
        G_prime = self.st_v(G)[0] # st_v is similar to building an MST
        G_prime_weight = self.st_v(G)[1]

        V, E = G_prime[0].copy(), G_prime[1].copy() # Hard copy from the original graph
        E = set(frozenset(e) for e in E if e[0] in V and e[1] in V) # Edges
        V_sub = V - T # All nodes that may require removing

        removed = True # Flag for whether to remove or not
        while removed:
            removed = False
            deg = {v: 0 for v in V} # Record the degrees
            for e in E:
                u, v = e
                if u in deg and v in deg:
                    deg[u] += 1
                    deg[v] += 1

            to_be_removed = {v for v in V_sub if deg.get(v, 0) == 1}

            # remove all leaf nodes
            for leaf in to_be_removed:
                if leaf in V:
                    to_remove = None
                    for e in E:
                        if leaf in e:
                            to_remove = e
                            break

                    if to_remove:
                        E.remove(to_remove)
                        V.remove(leaf)
                        G_prime_weight -= self.dist[tuple(to_remove)[0]][tuple(to_remove)[1]] # Update the weight
                        removed = True

        # Update edges
        E = [tuple(e) for e in E]

        return (V, E), G_prime_weight



# Get the proper subsets of a particular set, if size is given, it will output the given-sized length of sets, otherwise it will output all subsets.
def proper_subsets(Set, size=None):
    return [set(s) for i in range(len(Set)+1) for s in itertools.combinations(Set, i)] if not size else [set(s) for s in itertools.combinations(Set, size)]



# connected_components requires depth first search (dfs).
def connected_components(G_prime):
    vertices, edges = G_prime[0], G_prime[1]
    adj = {vertex: set() for vertex in vertices}

    # Record adjacent vertices
    for u, v in edges:
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    total_compo = 0

    # Use DFS, or called backtrack to find whether the graph consists of only one connected component.
    def DFS(vertex):
        stack = [vertex]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(adj[curr])

    # Count the total components
    for vertex in vertices:
        if vertex not in visited:
            DFS(vertex)
            total_compo += 1

    return total_compo


# Cycle test
def cycle_test(G_prime):
    V, E = G_prime[0], G_prime[1]
    parent = {v: v for v in V} # Parent nodes

    # Find the parent of a particular node
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # Using union-find algorithm to explore whether there are loops.
    # If loops exist, there will be nodes whose parents are visited (connected) nodes.
    for u, v in E: # Make sure that both u and v are in parent
        if u not in parent:
            parent[u] = u
        if v not in parent:
            parent[v] = v
        root_u, root_v = find(u), find(v) # Root nodes of u and v
        if root_u == root_v: # if same it means that there is a loop
            return 1
        parent[root_v] = root_u # Otherwise, they will be combined together

    return 0

if __name__ == '__main__':
    path_hash_table = {}
    nodes = []

    with open('graph_1000_100000.edgelist', 'r') as file:
        for line in file:
            u, v, w = line.split()
            u = int(u)
            v = int(v)
            path_hash_table[(u, v)] = int(w)
            if u not in nodes:
                nodes.append(u)
            if v not in nodes:
                nodes.append(v)


    steiner = Steiner(V=nodes, E=list(path_hash_table.keys()), w=list(path_hash_table.values()))

    G_prime, G_prime_weight = steiner.st_2(steiner.G)
    assert connected_components(G_prime) == 1
    assert cycle_test(G_prime) == 0

    G_prime_prime, G_prime_prime_weight = steiner.st_v(steiner.G)
    assert connected_components(G_prime_prime) == 1
    assert cycle_test(G_prime_prime) == 0

    steiner = Steiner(V=nodes, E=list(path_hash_table.keys()), w=list(path_hash_table.values()))

    T = {1, 999//2, 999}
    start_time = time.time()
    G_triple_prime, G_triple_prime_weight = steiner.st_t(steiner.G, T)
    end_time = time.time()
    delta = end_time - start_time
    print(f'St_t on T takes {delta:.4f} seconds.')
    assert connected_components(G_triple_prime) == 1
    assert cycle_test(G_triple_prime) == 0
    print(f'weight by st_t on T is {G_triple_prime_weight}.')

    T2 = {1, 333, 666, 999}
    start_time = time.time()
    G_triple_prime, G_triple_prime_weight = steiner.st_t(steiner.G, T2)
    end_time = time.time()
    delta = end_time - start_time
    print(f'St_t on T2 takes {delta:.4f} seconds.')
    assert connected_components(G_triple_prime) == 1
    assert cycle_test(G_triple_prime) == 0
    print(f'weight by st_t on T2 is {G_triple_prime_weight}.')

    steiner = Steiner(V=nodes, E=list(path_hash_table.keys()), w=list(path_hash_table.values()))

    start_time = time.time()
    G_fourth, G_fourth_weight = steiner.st_h(steiner.G, T)
    end_time = time.time()
    delta = end_time - start_time
    print(f'St_h on T takes {delta:.4f} seconds.')
    assert connected_components(G_fourth) == 1
    assert cycle_test(G_fourth) == 0
    print(f'weight by st_h on T is {G_fourth_weight}.')

    start_time = time.time()
    G_fourth, G_fourth_weight = steiner.st_h(steiner.G, T2)
    end_time = time.time()
    delta = end_time - start_time
    print(f'St_h on T2 takes {delta:.4f} seconds.')
    assert connected_components(G_fourth) == 1
    assert cycle_test(G_fourth) == 0
    print(f'weight by st_h on T2 is {G_fourth_weight}.')
