class Node(object):

    def __init__(self, index, centroid = None, coefs = None):
        self.index = index
        self.coefs = coefs
        self.centroid = centroid

    def update_index(self, x):
        self.index = x

    def update_centroid(self, centroid):
        self.centroid = centroid

    def update_coefs(self, coefs):
        self.coefs = coefs

    def get_index(self):
        return self.index

    def get_centroid(self):
        return self.centroid

    def get_coefs(self):
        return self.coefs

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
