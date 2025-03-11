'''

Quick Sort also uses divide-and-conquer, but it picks a pivot
and partitions the array around the pivot by placing the pivot
in the correct position in the new sorted array.

Time complexity: Average O(n log n), worst case O(n^2)

'''

def quickSort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[0]

    arrB = [x for x in arr[1:] if x <= pivot]
    arrC = [x for x in arr[1:] if x > pivot]

    return quickSort(arrB) + [pivot] + quickSort(arrC)


print(quickSort([1,4,7,5,3,6,9]))