# Priority Queue Implementation With Min-Heap
class PriorityQueueNode:
    def __init__(self, vertex, weight, parent=None):
        self.vertex = vertex
        self.weight = weight
        self.parent = parent

    def __str__(self):
        return f"{self.vertex}, {self.weight}, {self.parent}"

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def is_empty(self):
        return len(self.heap) == 0

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def enqueue(self, vertex, weight, parent=None):
        node = PriorityQueueNode(vertex, weight, parent)
        self.heap.append(node)
        self.heapify_up(len(self.heap) - 1)

    def dequeue(self):
        if self.is_empty():
            return None

        root = self.heap[0]
        last_node = self.heap.pop()

        if not self.is_empty():
            self.heap[0] = last_node
            self.heapify_down(0)

        return root

    def heapify_up(self, index):
        while index > 0:
            parent_idx = self.parent(index)
            if self.heap[index].weight < self.heap[parent_idx].weight:
                self.heap[index], self.heap[parent_idx] = self.heap[parent_idx], self.heap[index]
                index = parent_idx
            else:
                break

    def heapify_down(self, index):
        size = len(self.heap)
        while True:
            left = self.left_child(index)
            right = self.right_child(index)
            smallest = index

            if left < size and self.heap[left].weight < self.heap[smallest].weight:
                smallest = left
            if right < size and self.heap[right].weight < self.heap[smallest].weight:
                smallest = right

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest

    def print_queue(self):
        if self.is_empty():
            print("empty")
            return
        for node in self.heap:
            print(node, end=" ")
        print()




# class PriorityQueueNode:
#     def __init__(self, vertex, weight, parent = None):
#         self.vertex = vertex
#         self.weight = weight
#         self.parent = parent
#         self.next = None
#
#     def __str__(self):
#         return f"{self.vertex}, {self.weight}, {self.parent}"
#
# class PriorityQueue:
#
#     def __init__(self):
#         self.front = None
#
#     def is_empty(self):
#         return self.front is None
#
#     def enqueue(self, vertex, weight, parent = None):
#         new_node = PriorityQueueNode(vertex, weight, parent)
#
#         if self.is_empty() or self.front.weight > weight:
#             new_node.next = self.front
#             self.front = new_node
#             return
#
#         curr = self.front
#
#         while curr.next is not None and new_node.weight >= curr.next.weight:
#             curr = curr.next
#
#         new_node.next = curr.next
#         curr.next = new_node
#
#     def dequeue(self):
#         if self.is_empty():
#             return None
#         temp = self.front
#         self.front = self.front.next
#         return temp
#
#     def print_queue(self):
#         if self.is_empty():
#             print("empty")
#             return
#
#         curr = self.front
#         print("\n")
#         while curr:
#             print(curr, end =" ")
#             curr = curr.next
#
#
#
