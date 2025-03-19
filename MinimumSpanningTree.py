'''



'''
import heapq


def Prim(graph, start):
    if start not in graph:
        return

    MST = []
    visited = set()
    heap = []

    visited.add(start)

    for neighbor, weight in graph[start]:
        heapq.heappush(heap, (weight, start, neighbor))

    while heap:
        weight, u, v = heapq.heappop(heap)
        if v not in visited:
            visited.add(v)
            MST.append((weight, u, v))

            for neighbor, weight in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(heap, (weight, v, neighbor))

    return MST

if __name__ == '__main__':
    graph = {
        's': [('t', 3), ('u', 6), ('v', 4), ('w', 8), ('x', 5)],
        't': [('u', 1), ('v', 1)],
        'u': [('x', 1)],
        'v': [('w', 1)],
        'w': [('x', 7)],
        'x': []
    }
    print(Prim(graph, 'A'))

    graph2 = {
        's': [('t', 13), ('v', 11), ('x', 6), ('u', 9)],
        't': [('v', 2)],
        'u': [('v', 7), ('x', 4), ('w', 2)],
        'v': [('x', 5)],
        'w': [('x', 3)],
        'x': []
    }
    print(Prim(graph2, 's'))