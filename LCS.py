# DP case: longest common subsequence
from typing import List

def LCS(s1, s2) -> List[List[int]]:
    m, n = len(s1), len(s2)
    DP_table = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for j in range(n + 1):
        for i in range(m + 1):
            if i == 0 or j == 0:
                DP_table[j][i] = 0
            elif s1[i-1] == s2[j-1]:
                DP_table[j][i] = DP_table[j-1][i-1] + 1
            else:
                DP_table[j][i] = max(DP_table[j - 1][i], DP_table[j][i - 1])

    return DP_table

# Complexity: O(m*n)

if __name__ == '__main__':
    s1, s2 = "bacdabacdaa", "adada"
    lcs = LCS(s1, s2)
    for i in lcs:
        print(i)