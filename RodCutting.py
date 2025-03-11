'''

Rod cutting is a typical dynamic programming algorithm.
The goal is to find the maximum profits from cutting a particular length of rod,
with profits of each length of rods cut.

'''

def rodCutting(prices):
    DP = [0] * (len(prices) + 1)
    for i in range(1, len(prices)+1):
        for j in range(1, i+1):
            DP[i] = max(DP[i], prices[j - 1] + DP[i - j])

    return DP[-1]

if __name__ == '__main__':
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    assert rodCutting(prices) == 22
