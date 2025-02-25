from typing import List


class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        f = [0] + [-1] * k
        for pile in piles:
            for i in range(k, 0, -1):
                value = 0
                for t in range(1, len(pile) + 1):
                    value += pile[t - 1]
                    if i >= t and f[i - t] != -1:
                        f[i] = max(f[i], f[i - t] + value)
        return f[k]


solution = Solution()
print(solution.maxValueOfCoins([[100],[100],[100],[100],[100],[100],[1,1,1,1,1,1,700]], 7))
