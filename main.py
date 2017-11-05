def main():
	# init
	# get all data points
	#data_points = load...
	cluster_dic = {}
	cluster_num = len(data_points)

	for i in range(cluster_num):
		cur_node = Node(data_points[i])
		cur_cluster = Cluster()
		cur_cluster.add(cur_node)
		cluster_dic[i] = cur_cluster

	# compute initial distance matrix
	distance_matrix = computeDistanceMatrix(cluster_dic)

	# start merge clusters
	while cluster_num > expected_cluster_num:
		# find two cluster with the cloest distance		
		cluster1, cluster2 = findCloestCluster(cluster_dic, distance_matrix)
		distance_matrix = mergeCluster(cluster1,cluster2,cluster_dic, distance_matrix)




def findCloestCluster(cluster_dic, distance_matrix):
	pass
	# return the index of cluster1, cluster2	


def mergeCluster(cluster1, cluster2, cluster_dic, distance_matrix):
	pass
	# update the cluster_dic

    # merge cluster2 into cluster1
    cluster_dic[cluster1].merge(cluster_dic[cluster2])
    # remove cluster2 from cluster_dic
    del cluster_dic[cluster2]
	# update the distance_matrix
	
	return distance_matrix
