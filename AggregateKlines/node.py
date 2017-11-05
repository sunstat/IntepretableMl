class Node(object):

    def __init__(self, index, centroid = None, beta = None):
        self.index = index
        self.beta = beta
        self.centroid = centroid

    def update_index(self, x):
        self.index = x

    def set_centroid(self, centroid):
        self.centroid = centroid

    def update_beta(self, beta):
        self.beta = beta

if __name__ == "__main__":

    '''
    v_map = {}
    nd1 = Node(1)
    nd2 = Node(2)
    nd3 = Node(3)
    v_map = {}
    v_map[nd1] = 1
    print(v_map[nd1])
    nd1.update_index(10)
    print(v_map[nd1])
    '''

    nd1 = Node(1)
    nd2 = Node(2)
    tuple = (nd1, nd2)
    for item in tuple:
        print(nd1==item)



