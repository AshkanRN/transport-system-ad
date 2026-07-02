class QueueNode:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __str__(self):
        return f"{self.value}"


class Queue:

    def __init__(self):
        self.front = None
        self.rear = None

    def is_empty(self):
        return self.front is None

    def enqueue(self, value):
        new_node = QueueNode(value)

        if self.is_empty():
            self.rear = self.front = new_node

        else:
            self.rear.next = new_node
            self.rear = new_node

    def dequeue(self):
        if self.is_empty():
            print("\nQueue is Empty")
            return None

        temp = self.front
        self.front = self.front.next

        if self.is_empty():
            self.rear = None

        return temp

    def print_queue(self):
        if self.is_empty():
            print("empty")
            return

        curr = self.front
        print("\n")
        while curr:
            print(curr, end=" ")
            curr = curr.next