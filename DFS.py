'''

"graph" is a dictionary, in which the key-value pairs are: u: (v1, v2, v3, ...),
where u is the starting vertex and v*s are ending vertices.

'''

def DFS(graph, start):
    if start not in graph:
        return []

    stack = [start]
    DFS_tree = []
    visited = set()

    while stack:
        u = stack.pop()
        DFS_tree.append(u)
        for v in graph.get(u, ()):
            if v not in visited:
                visited.add(v)
                stack.append(v)

    return DFS_tree

if __name__ == '__main__':
    graph = {1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9), 5: (10, 11), 6: (12, 13), 7: (14, 15)}

    print(DFS(graph, 1))

    graph2 = {1: (2, 3), 2: (4, 5), 3: (4, 5), 4: (6,), 5: (6,)}
    print(DFS(graph2, 1))