import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np

# For convenience in calculating heap size, we build a heap class here.
class Heap:
    # Constructor
    def __init__(self, A):
        self.A = A
        self.length = len(A) - 1
        self.heapsize = len(A) - 1

    # Return left child index
    def left(self, i):
        return 2*i+1

    # Return right child index
    def right(self, i):
        return 2*i+2

    # Return parent root index
    def parent(self, i):
        return (i - 1)//2

    # Max heapify function
    def max_heapify(self, A, i):
        l = self.left(i)
        r = self.right(i)
        if l < self.heapsize + 1 and A[l] > A[i]:
            largest = l
        else:
            largest = i
        if r <= self.heapsize and A[r] > A[largest]:
            largest = r
        if largest != i:
            dummy = A[largest]
            A[largest] = A[i]
            A[i] = dummy
            self.max_heapify(A, largest)


    # Exercise 1: Build max heap
    def build_max_heap(self, A):
        for i in range(self.heapsize//2, -1, -1):
            self.max_heapify(A, i)


    # Exercise 2: heap_sort
    def heap_sort(self, A):
        self.build_max_heap(A)
        for i in range(len(A) - 1, 0, -1):
            dummy = A[i]
            A[i] = A[0]
            A[0] = dummy
            self.heapsize -= 1
            self.max_heapify(A, 0)
        self.heapsize = len(A) - 1


# Priority queue class, inherited from heap
class priority_queue(Heap):
    def __init__(self, S):
        super().__init__(S)

    # Get maximum value from the heap
    def heap_max(self, S):
        return S[0]

    # Increase_key function from the textbook
    def increase_key(self, S, i, key):
        if key < S[i]:
            return
        S[i] = key
        while i > 0 and S[self.parent(i)] < S[i]:
            dummy = S[self.parent(i)]
            S[self.parent(i)] = S[i]
            S[i] = dummy
            i = self.parent(i)

    # Insert function fromm the textbook
    def insert(self, S, key):
        self.heapsize += 1
        S.append(-math.inf)
        self.increase_key(S, self.heapsize, key)

    # Extract max function from the textbook
    def extract_max(self, S):
        if self.heapsize < 0:
            return

        max_val = S[0]
        S[0] = S[self.heapsize]
        self.heapsize -= 1
        S.pop()
        self.max_heapify(S, 0)
        return max_val

    # Exercise 3: Dynamic sort
    def dynamic_sort(self, S):
        new_pq = priority_queue([])
        len_sort = self.heapsize
        for i in range(len_sort+1):
            max_val = self.extract_max(S)
            new_pq.insert(new_pq.A, max_val)
            new_pq.length += 1
        for i in range(len(new_pq.A)//2 + 1):
            dummy = new_pq.A[i]
            new_pq.A[i] = new_pq.A[len(new_pq.A) - 1 - i]
            new_pq.A[len(new_pq.A) - 1 - i] = dummy
        self.heapsize = self.length
        self.A = new_pq.A


if __name__ == "__main__":
    # Set sizes of test cases
    size_random = [500+500*x for x in range(50)]
    size_reverse = [500+500*x for x in range(50)]

    # Runtimes for different lengths of test cases on different functions
    runtime_random_max_heap = []
    runtime_random_heap_sort = []
    runtime_random_dynamic_sort = []
    runtime_reverse_max_heap = []
    runtime_reverse_heap_sort = []
    runtime_reverse_dynamic_sort = []

    for i in range(500, 25500, 25000//50):
        # Test cases in different lengths
        A_random = [random.randint(1, 1000) for _ in range(i)]
        A_random_2 = [random.randint(1, 1000) for _ in range(i)]
        A_random_3 = [random.randint(1, 1000) for _ in range(i)]
        pq = priority_queue(A_random)
        # Calculate the runtime
        start_1 = time.time()
        pq.build_max_heap(A_random)
        end_1 = time.time()
        runtime_random_max_heap.append(end_1 - start_1)
        start_2 = time.time()
        pq.heap_sort(A_random_2)
        end_2 = time.time()
        runtime_random_heap_sort.append(end_2 - start_2)
        start_3 = time.time()
        pq.dynamic_sort(A_random_3)
        end_3 = time.time()
        runtime_random_dynamic_sort.append(end_3 - start_3)
    # These are the same as the last for loop, but test cases are reverse arrays
    # instead of random ones
    for i in range(500, 25500, 25000//50):
        A_reverse = [i - x for x in range(i)]
        A_reverse_2 = [i - x for x in range(i)]
        A_reverse_3 = [i - x for x in range(i)]
        pq = priority_queue(A_reverse)
        start_1 = time.time()
        pq.build_max_heap(A_reverse)
        end_1 = time.time()
        runtime_reverse_max_heap.append(end_1 - start_1)
        start_2 = time.time()
        pq.heap_sort(A_reverse_2)
        end_2 = time.time()
        runtime_reverse_heap_sort.append(end_2 - start_2)
        start_3 = time.time()
        pq.dynamic_sort(A_reverse_3)
        end_3 = time.time()
        runtime_reverse_dynamic_sort.append(end_3 - start_3)
    # Turn the arrays into numpy arrays in order to plot line graphs
    size_random = np.array(size_random)
    size_reverse = np.array(size_reverse)
    runtime_random_max_heap = np.array(runtime_random_max_heap)
    runtime_random_heap_sort = np.array(runtime_random_heap_sort)
    runtime_random_dynamic_sort = np.array(runtime_random_dynamic_sort)
    runtime_reverse_max_heap = np.array(runtime_reverse_max_heap)
    runtime_reverse_heap_sort = np.array(runtime_reverse_heap_sort)
    runtime_reverse_dynamic_sort = np.array(runtime_reverse_dynamic_sort)
    # Build linear regression functions for each function on different arrays
    coef_1 = np.polyfit(size_random, runtime_random_max_heap, deg=1)
    poly_1 = np.poly1d(coef_1)
    coef_2 = np.polyfit(size_random, runtime_random_heap_sort, deg=1)
    poly_2 = np.poly1d(coef_2)
    coef_3 = np.polyfit(size_random, runtime_random_dynamic_sort, deg=1)
    poly_3 = np.poly1d(coef_3)
    coef_4 = np.polyfit(size_reverse, runtime_reverse_max_heap, deg=1)
    poly_4 = np.poly1d(coef_4)
    coef_5 = np.polyfit(size_reverse, runtime_reverse_heap_sort, deg=1)
    poly_5 = np.poly1d(coef_5)
    coef_6 = np.polyfit(size_reverse, runtime_reverse_dynamic_sort, deg=1)
    poly_6 = np.poly1d(coef_6)
    # Make the line smooth
    x_smooth = np.linspace(min(size_reverse), max(size_reverse), 1000)
    y_smooth_1 = poly_1(x_smooth)
    y_smooth_2 = poly_2(x_smooth)
    y_smooth_3 = poly_3(x_smooth)
    y_smooth_4 = poly_4(x_smooth)
    y_smooth_5 = poly_5(x_smooth)
    y_smooth_6 = poly_6(x_smooth)
    # Plot the line graphs
    plt.title("Max Heap - Random Array")
    plt.plot(x_smooth, y_smooth_1, label="Max Heap on Random", color="red")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()
    plt.title("Heap Sort - Random Array")
    plt.plot(x_smooth, y_smooth_2, label="Heap Sort on Random", color="orange")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()
    plt.title("Dynamic Sort - Random Array")
    plt.plot(x_smooth, y_smooth_3, label="Dynamic Sort on Random", color="yellow")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()
    plt.title("Max Heap - Reverse Array")
    plt.plot(x_smooth, y_smooth_4, label="Max Heap on Reverse", color="green")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()
    plt.title("Heap Sort - Reverse Array")
    plt.plot(x_smooth, y_smooth_5, label="Heap Sort on Reverse", color="blue")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()
    plt.title("Dynamic Sort - Reverse Array")
    plt.plot(x_smooth, y_smooth_6, label="Dynamic Sort on Reverse", color="purple")
    plt.xlabel("Size")
    plt.ylabel("Runtime")
    plt.show()