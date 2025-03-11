'''

Insertion sort is a sorting algorithm which iteratively inserting each element of an unsorted list into the
correct position in a sorted portion of the list.

'''

def insertionSort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1

        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key

array = [7,6,5,4,3,2,1]
insertionSort(array)
assert array == [1,2,3,4,5,6,7]
