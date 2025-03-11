'''

Merge sort is a sorting algorithm using divide-and-conquer approach,
which recursively divides the array into smaller subarrays and sorts the arrays,
then merges the arrays back together to get the sorted array.

Complexity: O(n * log n)

'''

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2

    arrB = merge_sort(arr[:mid])
    arrC = merge_sort(arr[mid:])

    return merge(arrB, arrC)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


arr = [1,4,7,5,3,6,9,8,2,0]
print(merge_sort(arr))

