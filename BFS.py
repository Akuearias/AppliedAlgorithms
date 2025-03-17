'''

"graph" is a dictionary, in which the key-value pairs are: u: (v1, v2, v3, ...),
where u is the starting vertex and v*s are ending vertices.

'''
from collections import defaultdict


def BFS(graph, start):
    if not start in graph:
        return []

    visited = set()
    BFS_tree = [start]
    queue = [start]
    visited.add(start)

    while queue:
        u = queue.pop(0)
        for v in graph.get(u, ()):
            if v not in visited:
                visited.add(v)
                queue.append(v)
                BFS_tree.append(v)

    return BFS_tree


if __name__ == '__main__':
    graph = {1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9), 5: (10, 11), 6: (12, 13), 7: (14, 15)}

    print(BFS(graph, 1))

    graph2 = {1: (2, 3), 2: (4, 5), 3: (4, 5), 4: (6,), 5: (6,)}
    print(BFS(graph2, 1))