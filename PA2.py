'''
As mentioned in problem description,
there is relationship between
longest common sequence (LCS) and edit distance (ED).
In order to track the pairing result of LCS, we improved the
conventional LCS algorithm, making it able to track which characters
are paired and which are not.
'''
import traceback
# For convenience in operating convert function, we use linked list data structure.
# Linked lists can be added, deleted, and modified more flexibly than traditional
# arrays
# because each node in a linked list has its own memory, unlike arrays,
# which have fixed memory and increase memory complexity when adding, deleting, and
# modifying.
class ListNode:
    def __init__(self, x, next=None):
        self.val = x
        self.next = next

    def print_nodes(self):
        current = self
        nodes = []
        while current:
            if current.val:
                nodes.append(current.val)
            current = current.next
        return ''.join(nodes)

def build(nodes):
    if not nodes:
        return ListNode(None)
    else:
        listNode = ListNode(None, next=ListNode(nodes[0]))
        head = listNode
        current = head.next
        for i in range(1, len(nodes)):
            current.next = ListNode(nodes[i])
            current = current.next
        return head

def LCS(S, T):
    # Original LCS algorithm
    m, n = len(S), len(T)
    DP_table = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if i == 0 or j == 0:
                DP_table[i][j] = 0
            elif S[i-1] == T[j-1]:
                DP_table[i][j] = DP_table[i-1][j-1] + 1
            else:
                DP_table[i][j] = max(DP_table[i - 1][j], DP_table[i][j - 1])
    return DP_table

def ed_ins_del(S, T):
    DP_table = LCS(S, T)
    m, n = len(S), len(T)
    # Then we try tracking the concrete pairing status
    i, j = m, n
    paired = [] # Record the indices of two strings where the characters are the same
    while i > 0 and j > 0:
        if S[i-1] == T[j-1]:
            paired.append((i-1, j-1)) # If same, append it to the array
            i -= 1
            j -= 1
        elif DP_table[i-1][j] >= DP_table[i][j-1]:
            i -= 1
        else:
            j -= 1
    paired.reverse() # Reverse the array

    # We output the status of each pair of characters.
    # As we all know, in LCS, different characters will be handled by moving the pointer of the one having greater
    # matching value right by 1 (in order to keep the maximum value from previous circulations).
    # Here we will simulate the LCS but output whether paired or not. If paired we will use "+", otherwise "-".

    pair1, pair2, pair = [], [], []
    k, l = 0, 0
    for i1, i2 in paired:
        while k < i1: # Dealing with the 1st string where the characters do not match in the same index
            pair1.append(S[k])
            pair2.append("-")
            pair.append("-")
            k += 1
        while l < i2:
            pair1.append("-")
            pair2.append(T[l])
            pair.append("-")
            l += 1
# Dealing with matched characters
        pair1.append(S[i1])
        pair2.append(T[i2])
        pair.append("+")
        k += 1
        l += 1
# Dealing with the rest after finding all matched characters
    while k < m:
        pair1.append(S[k])
        pair2.append("-")
        pair.append("-")
        k += 1

    while l < n:
        pair1.append("-")
        pair2.append(T[l])
        pair.append("-")
        l += 1

    oper_seq = [pair1, pair2]
    # DP table for ED
    cost_list = [0] * len(pair)
    cost_list[0] = 1 if pair1[0] != pair2[0] else 0
    for i in range(1, len(pair)):
        if pair1[i] != pair2[i]:
            cost_list[i] = cost_list[i - 1] + 1
        else:
            cost_list[i] = cost_list[i - 1]
    # As substitution is not allowed, we can say that each "-" means one ED.
    # We can just return the number of "-"s for ED.
    # The operation sequence consists of a matrix with two rows.
    return cost_list[-1], oper_seq

def convert(S, oper_seq):
    series = oper_seq[1]
    s, t = series[0], series[1]
    S_listnode = build(S)
    head = S_listnode
    prev = head # previous node
    current = head.next # current node
    i = 0
    while i < len(s):
        if t[i] != s[i]:
            if t[i] == '-':
                if i+1 < len(s) and s[i+1] == '-': # substitution, with one deletion and one insertion in Exercise 1
                    current.val = t[i+1]
                    prev = prev.next
                    current = current.next
                    i += 2
                else: # deletion
                    prev.next = current.next
                    current = current.next
                    i += 1

            elif s[i] == '-': # insertion
                new_node = ListNode(t[i], next=current)
                prev.next = new_node
                prev = prev.next
                i += 1
            else: # substitution for Exercises 2 and 3
                current.val = t[i]
                prev = prev.next
                current = current.next
                i += 1

        else: # When the characters on the same position are the same
            prev = prev.next
            current = current.next
            i += 1
    return head.print_nodes()


'''

For ed_std in Exercise 2, most of the code is the same as the one in
ed_ins_del.
However, we added a mechanism which can pair different characters instead of
using two adjacent "-"s in different
sequence lists.

'''


def ed_std(S, T):
    DP_table = LCS(S, T)
    m, n = len(S), len(T)
    # Track the concrete pairing status
    i, j = m, n
    paired = [] # Record the indices of two strings where the characters are the same
    while i > 0 and j > 0:
        if S[i - 1] == T[j - 1]:
            paired.append((i - 1, j - 1)) # If same, append it to the array
            i -= 1
            j -= 1
        elif DP_table[i - 1][j] >= DP_table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    paired.reverse() # Reverse the array

    pair1, pair2, pair = [], [], []
    k, l = 0, 0

    for i1, i2 in paired:
        while k < i1: # Dealing with the 1st string where the characters do not match in the same index
            pair1.append(S[k])
            pair2.append("-")
            pair.append("-")
            k += 1
        while l < i2:
            pair1.append("-")
            pair2.append(T[l])
            pair.append("-")
            l += 1
        # Dealing with matched characters
        pair1.append(S[i1])
        pair2.append(T[i2])
        pair.append("+")
        k += 1
        l += 1

    # Dealing with the rest after finding all matched characters
    while k < m:
        pair1.append(S[k])
        pair2.append("-")
        pair.append("-")
        k += 1
    while l < n:
        pair1.append("-")
        pair2.append(T[l])
        pair.append("-")
        l += 1
    oper_seq = [pair1, pair2]
    match = []
    for p in range(len(pair)):
        if pair[p] == '+':
            match.append(p)

    # Optimization of the sequence lists and calculation of the cost.
    prev = 0 # Use prev variables to limit the searching range.
    new_oper_seq1, new_oper_seq2 = [], []
    for i in match: # Transfer "insertion + deletion"s into substitutions.
        slice1, slice2 = oper_seq[0][prev:i], oper_seq[1][prev:i]
        j = 0
        while j < len(slice1) - 1:
            if slice1[j+1] == '-' and slice2[j] == '-':
                slice1.remove(slice1[j+1])
                slice2.remove(slice2[j])
                j = 0
            j += 1

        new_oper_seq1 += slice1
        new_oper_seq2 += slice2
        prev = i

    i = len(oper_seq[0])
    slice1, slice2 = oper_seq[0][prev:i], oper_seq[1][prev:i]
    j = 0

    while j < len(slice1) - 1:
        if slice1[j + 1] == '-' and slice2[j] == '-':
            slice1.remove(slice1[j + 1])
            slice2.remove(slice2[j])
            j = 0
        j += 1

    new_oper_seq1 += slice1
    new_oper_seq2 += slice2
    new_oper_seq = [new_oper_seq1, new_oper_seq2]
    cost = [0] * len(new_oper_seq1)
    cost[0] = 1 if new_oper_seq1[0] != new_oper_seq2[0] else 0
    for k in range(1, len(new_oper_seq1)): # Calculate the total cost
        if new_oper_seq1[k] != new_oper_seq2[k]:
            cost[k] = cost[k - 1] + 1
        else:
            cost[k] = cost[k - 1]
    return cost[-1], new_oper_seq

# ed_weighted is also mostly similar to ed_ins_del, and mostly the same as ed_std,
# but the key difference is calculation of the cost.
def ed_weighted(S, T, wi, wd, ws):
    DP_table = LCS(S, T)
    m, n = len(S), len(T)
    # Track the concrete pairing status
    i, j = m, n
    paired = [] # Record the indices of two strings where the characters are the same

    while i > 0 and j > 0:
        if S[i - 1] == T[j - 1]:
            paired.append((i - 1, j - 1)) # If same, append it to the array
            i -= 1
            j -= 1
        elif DP_table[i - 1][j] >= DP_table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    paired.reverse() # Reverse the array
    pair1, pair2, pair = [], [], []
    k, l = 0, 0
    for i1, i2 in paired:
        while k < i1: # Dealing with the 1st string where the characters do not match in the same index
            pair1.append(S[k])
            pair2.append("-")
            pair.append("-")
            k += 1

        while l < i2:
            pair1.append("-")
            pair2.append(T[l])
            pair.append("-")
            l += 1

        # Dealing with matched characters
        pair1.append(S[i1])
        pair2.append(T[i2])
        pair.append("+")
        k += 1
        l += 1

    # Dealing with the rest after finding all matched characters
    while k < m:
        pair1.append(S[k])
        pair2.append("-")
        pair.append("-")
        k += 1

    while l < n:
        pair1.append("-")
        pair2.append(T[l])
        pair.append("-")
        l += 1

    oper_seq = [pair1, pair2]
    match = []
    for p in range(len(pair)):
        if pair[p] == '+':
            match.append(p)

    # Optimization of the sequence lists and calculation of the cost.
    prev = 0 # Use prev variables to limit the searching range.
    new_oper_seq1, new_oper_seq2 = [], []
    for i in match: # Transfer "insertion + deletion"s into substitutions.
        slice1, slice2 = oper_seq[0][prev:i], oper_seq[1][prev:i]
        j = 0
        while j < len(slice1) - 1:
            if slice1[j + 1] == '-' and slice2[j] == '-':
                slice1.remove(slice1[j + 1])
                slice2.remove(slice2[j])
                j = 0
            j += 1
        new_oper_seq1 += slice1
        new_oper_seq2 += slice2
        prev = i

    i = len(oper_seq[0])
    slice1, slice2 = oper_seq[0][prev:i], oper_seq[1][prev:i]
    j = 0

    while j < len(slice1) - 1:
        if slice1[j + 1] == '-' and slice2[j] == '-':
            slice1.remove(slice1[j + 1])
            slice2.remove(slice2[j])
            j = 0
        j += 1

    new_oper_seq1 += slice1
    new_oper_seq2 += slice2
    new_oper_seq = [new_oper_seq1, new_oper_seq2]

    cost = [0] * len(new_oper_seq1)
    if new_oper_seq1[0] != new_oper_seq2[0]:
        if new_oper_seq1[0] == '-':
            cost[0] = wi
        elif new_oper_seq2[0] == '-':
            cost[0] = wd
        else:
            cost[0] = ws

    for k in range(len(new_oper_seq1)): # Calculate the total cost
        if new_oper_seq1[k] != new_oper_seq2[k]:
            if new_oper_seq1[k] == '-':
                cost[k] = cost[k-1] + wi
            elif new_oper_seq2[k] == '-':
                cost[k] = cost[k-1] + wd
            else:
                cost[k] = cost[k-1] + ws
        else:
            cost[k] = cost[k-1]

    return cost[-1], new_oper_seq

if __name__ == '__main__':
    wi, wd, ws = 100, 20, 40
    S, T = 'ab' * 500, 'ba' * 500

    try:
        assert convert(S, ed_ins_del(S, T)) == T
        assert convert(S, ed_std(S, T)) == T
        assert convert(S, ed_weighted(S, T, wi, wd, ws)) == T
    except AssertionError:
        tb = traceback.extract_tb(traceback.format_exc().__traceback__)
        for frame in tb:
            print(f"Test failed at line {frame.lineno}: {frame.line}.")
    print(ed_ins_del(S, T))
    print(convert(S, ed_ins_del(S, T)))
    print(ed_std(S, T))
    print(convert(S, ed_std(S, T)))
    print(ed_weighted(S, T, wi, wd, ws))
    print(convert(S, ed_weighted(S, T, wi, wd, ws)))

    S, T = 'abcd' * 250, 'abcde' * 200
    try:
        assert convert(S, ed_ins_del(S, T)) == T
        assert convert(S, ed_std(S, T)) == T
        assert convert(S, ed_weighted(S, T, wi, wd, ws)) == T
    except AssertionError:
        tb = traceback.extract_tb(traceback.format_exc().__traceback__)
        for frame in tb:
            print(f"Test failed at line {frame.lineno}: {frame.line}.")
    print(ed_ins_del(S, T))
    print(convert(S, ed_ins_del(S, T)))
    print(ed_std(S, T))
    print(convert(S, ed_std(S, T)))
    print(ed_weighted(S, T, wi, wd, ws))
    print(convert(S, ed_weighted(S, T, wi, wd, ws)))

    S = "Howmanycookiescouldagoodcookcookifagoodcookcouldcookcookies"
    T = "Agoodcookcouldcookasmuchcookiesasagoodcookwhocouldcookcookies"

    try:
        assert convert(S, ed_ins_del(S, T)) == T
        assert convert(S, ed_std(S, T)) == T
        assert convert(S, ed_weighted(S, T, wi, wd, ws)) == T
    except AssertionError:
        tb = traceback.extract_tb(traceback.format_exc().__traceback__)
        for frame in tb:
            print(f"Test failed at line {frame.lineno}: {frame.line}.")

    print(ed_ins_del(S, T))
    print(convert(S, ed_ins_del(S, T)))
    print(ed_std(S, T))
    print(convert(S, ed_std(S, T)))
    print(ed_weighted(S, T, wi, wd, ws))
    print(convert(S, ed_weighted(S, T, wi, wd, ws)))