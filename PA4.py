import math
from typing import List

# Exercise 1: rk_k
def rk_k(S: str, T: List[str]):
    starts = [] # List recording all start points
    d = 256 # ASCII
    prime = 10 ** 9 + 7 # A huge prime number for Rabin Karp
    for P in T:
        n = len(S) # Original String
        m = len(P) # Pattern
        h = math.pow(d, m - 1) % prime
        p = 0 # Hash value of the pattern
        t0 = 0 # Hash value of current substring

        # Initialization of current hash values
        for i in range(m):
            p = (d * p + ord(P[i])) % prime
            t0 = (d * t0 + ord(S[i])) % prime

        # Use sliding window to match
        for i in range(n - m + 1):
            if p == t0:
                # Then check whether they really match if the hash values are the same.
                if S[i:i + m] == P:
                    starts.append(i) # Append if really match each other

            # Calculate the hash value of the next window.
            if i < n - m:
                t0 = (d * (t0 - ord(S[i]) * h) + ord(S[i+m])) % prime
                if t0 < 0:
                    t0 += prime

    return starts

# Exercise 2: rk_2d
def rk_2d(S: List[str], T: List[str]):
    M, N = len(S), len(S[0]) # Numbers of rows and columns of the text
    m, n = len(T), len(T[0]) # Numbers of rows and columns of the pattern

    b = 256
    prime = 10 ** 9 + 7

    pow_n = math.pow(b, n - 1) % prime

    # Internal function for row hashing of the whole pattern
    def row_hash(row):
        H = 0
        for column in row:
            H = (H * b + ord(column)) % prime
        return H

    # First we compute rolling hashes for each row.
    ptn_row_hashes = [row_hash(T[i]) for i in range(m)]
    ptn_hash = 0
    for H in ptn_row_hashes:
        ptn_hash = (ptn_hash * b + H) % prime

    # Use sliding windows to calculate hash of each sub-chunk in the text matrix.
    all_matches = []
    # Pre-computation of hash of substrings with length n in each row
    row_hashes = [[0] * (N - n + 1) for _ in range(M)]
    for i in range(M):
        h = 0
        for j in range(n):
            h = (h * b + ord(S[i][j])) % prime
        row_hashes[i][0] = h
        for j in range(1, N - n + 1):
            h = (h - ord(S[i][j - 1]) * pow_n) % prime
            h = (h * b + ord(S[i][j + n - 1])) % prime
            row_hashes[i][j] = h

    # Vertical traversal
    for i in range(M - m + 1):
        for j in range(N - n + 1):
            chunk_hash = 0
            for k in range(m):
                chunk_hash = (chunk_hash * b + row_hashes[i+k][j]) % prime
            if chunk_hash == ptn_hash:
                dummy = True # Flag for controlling whether the characters really match
                for x in range(m):
                    for y in range(n):
                        if S[i+x][j+y] != T[x][y]:
                            dummy = False
                            break
                if dummy:
                    all_matches.append((i, j))

    return all_matches


if __name__ == '__main__':
    word_k = open('test_k.txt', 'r').readlines() # text_k file
    word_2d = open('test_2d.txt', 'r').readlines() # text_2d file
    word_k_string = word_k[0].strip() # Remove the line break of the text

    word_k_patterns = [line.strip() for line in word_k[1:]] # Remove line breaks in all patterns

    for i in range(len(word_2d)):
        word_2d[i] = word_2d[i].strip() # Remove line breaks in word_2d

    # Text and pattern for Exercise 2
    word_2d_list = word_2d[0:len(word_2d) - 2]
    word_2d_pattern = word_2d[len(word_2d) - 2:]

    # Exercise 1
    RK_1 = rk_k(word_k_string, word_k_patterns)
    print(RK_1)

    # Exercise 2
    RK_2 = rk_2d(word_2d_list, word_2d_pattern)
    print(RK_2)
