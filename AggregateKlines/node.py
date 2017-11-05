class Node(object):

    def __init__(self, index, beta = None):
        self.index = index
        self.beta = beta

    def update_index(self, x):
        self.index = x


if __name__ == "__main__":
    v_map = {}
    nd1 = Node(1)
    nd2 = Node(2)
    nd3 = Node(3)
    v_map = {}
    v_map[nd1] = 1
    print(v_map[nd1])
    nd1.update_index(10)
    print(v_map[nd1])
