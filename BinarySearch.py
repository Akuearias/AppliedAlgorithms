'''

Binary search is an algorithm which can search the target value
in a sorted array by repeatedly dividing the search interval in half.
Time complexity: O(log n)

'''

def binarySearch(key, l, r, A):
    if l >= r:
        return -1

    mid = (l + r) // 2
    if A[mid] == key:
        return mid
    elif A[mid] < key:
        return binarySearch(key, mid + 1, r, A)
    return binarySearch(key, l, mid - 1, A)

if __name__ == '__main__':
    A = [18, 25, 37, 46, 59, 72, 79, 84, 99, 101]
    print(binarySearch(99, 0, len(A)-1, A))