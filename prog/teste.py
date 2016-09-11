import itertools, heapq, bisect

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.nodes = {}

    def put(self, priority, item):
        if item in self.nodes:
            pos = bisect.bisect_left(self.queue, [self.nodes[item], item])
            del self.queue[pos]
        bisect.insort_right(self.queue, [priority, item])
        self.nodes[item] = priority

    def pop(self):
        if self.queue:
            item = self.queue.pop(0)[1]
            del self.nodes[item]
            return item
        raise KeyError('pop from an empty priority queue')




class Coisa:
    def __init__(self, priority, coisa):
        self.priority = priority
        self.coisa = coisa

    def __lt__(self, other):
        return self.priority < other.priority


q = PriorityQueue()
q.put(1, "haha")
q.put(20, "hoho")
q.put(22, "huhu")
print(q.queue, q.nodes)

q.put(23, "hoho")
print(q.queue, q.nodes)

print(q.pop())
print(q.queue, q.nodes)
print(q.pop())
print(q.queue, q.nodes)
print(q.pop())
print(q.pop())
