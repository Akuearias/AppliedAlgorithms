'''



'''
import heapq
from collections import defaultdict


def Dijkstra(graph, source):
    if source not in graph:
        return

    path_dict = defaultdict()
    for vertex in graph:
        path_dict[vertex] = float('inf')

    path_dict[source] = 0
    visited = set()
    pq = [(source, 0)]

    while pq:
        u, current_dist = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        for v, w in graph[u]:
            if v not in visited and current_dist + w < path_dict[v]:
                path_dict[v] = current_dist + w
                heapq.heappush(pq, (v, path_dict[v]))

    return path_dict


if __name__ == '__main__':
    graph = {
        's': [('t', 3), ('u', 6), ('v', 4), ('w', 8), ('x', 5)],
        't': [('u', 1), ('v', 1)],
        'u': [('x', 1)],
        'v': [('w', 1)],
        'w': [('x', 7)],
        'x': []
    }
    print(Dijkstra(graph, 's'))