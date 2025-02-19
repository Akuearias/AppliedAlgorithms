from typing import List


def LIS(nums: List[int]) -> int:
    if not nums:
        return 0

    l = len(nums)
    dp = [1] * l
    max_len = 1

    for i in range(1, l):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
        max_len = max(max_len, dp[i])

    return dp[-1]

def IDS(nums: List[int]) -> int:
    if not nums:
        return 0

    dummy = nums.index(max(nums))
    nums1 = nums[:dummy+1]
    nums2 = nums[(len(nums)-1):(dummy-1):-1]

    return LIS(nums1) + LIS(nums2) - 1

print(IDS([10, 9, 2, 5, 3, 7, 101, 18]))
