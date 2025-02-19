# DP case: rod cutting
def rod_cutting(prices, n):
    # DP array initialization
    dp = [0] * (n + 1)

    # Fill the DP array
    for j in range(1, n + 1):
        max_value = 0
        for i in range(1, j + 1):
            max_value = max(max_value, prices[i - 1] + dp[j - i])
        dp[j] = max_value

    return dp[n]


if __name__ == '__main__':
    prices = [1, 5, 8, 9, 10, 17, 17, 20]  # Prices according to each length
    n = 8  # total length
    print("Max benefit:", rod_cutting(prices, n))
