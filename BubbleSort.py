'''

Bubble sort algorithm is an algorithm that repeatedly goes through all elements in an unsorted array,
comparing the two adjacent elements to each other and swapping their values when necessary.
Time complexity: O(n^2)

'''

def bubbleSort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                dummy = arr[j+1]
                arr[j+1] = arr[j]
                arr[j] = dummy
    return arr

array = [1, 3, 7, 4, 2]
print(bubbleSort(array))
