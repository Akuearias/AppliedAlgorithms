'''

Naive pattern matching algorithm. Time complexity: O(m*n).

'''

def NaiveSM(T, P):
    n = len(T)
    m = len(P)
    for i in range(n - m):
        if P[0:m] == T[i:i + m]:
            print(f'Same string detected at position {i}.')


def RabinKarp(T, P, d, q):
    n = len(T)
    m = len(P)
    h = (d ** m - 1) % q
    p = 0

    

if __name__ == '__main__':
    T = 'she sells seashells by the seashore'
    P = 'ell'
    NaiveSM(T, P)

