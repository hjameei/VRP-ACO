import networkx as nx
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import sys

def read_file():
	input_file = open('data.txt', 'r')
	nodes_position = []
	nodes_load = []
	for i, line in enumerate(input_file):
		values = line.split(' ')
		if i == 0:
			n_tours = int(values[0])
			init_capacity = int(values[1])
		else:
			if int(values[3])==0:
				depot = int(values[0])-1
			nodes_position.append([int(values[1]), int(values[2])])
			nodes_load.append(int(values[3]))
	input_file.close()
	n_nodes = len(nodes_position)
	nodes_position = np.array(nodes_position)
	nodes_load = np.array(nodes_load)
	return n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load

def get_distances(nodes_position):
	distances = pdist(nodes_position)
	distances = squareform(distances)
	return distances

def calculate_visibility(n_nodes, distances):
	visbility = np.zeros((n_nodes,n_nodes))
	for i in range(n_nodes):
		for j in range(n_nodes):
			if i != j:
				visbility[i][j] = 1.0/distances[i][j]
	return visbility

def initialize_pheromones(n_nodes):
	pheromone = np.zeros((n_nodes, n_nodes))
	for i in range(n_nodes):
		for j in range(n_nodes):
			pheromone[i][j] = 0.1
	return pheromone

def initialize_location(n_ants, depot, init_capacity):
	ants_location = {}
	ants_capacity = np.zeros(n_ants)
	for i in range(n_ants):
		ants_location[i] = [depot]
		ants_capacity[i] = init_capacity
	return ants_location, ants_capacity

def get_ant_distance(ants_location, ant, distances):
	dist = 0.0
	ant_location = ants_location[ant]
	for k in range(1, len(ant_location)):
		i = ant_location[k-1]
		j = ant_location[k]
		dist += distances[i][j]
	return dist

def get_best_ant_distance(best_ants_location, distances):
	dist = 0.0
	for k in range(1, len(best_ants_location)):
		i = best_ants_location[k-1]
		j = best_ants_location[k]
		dist += distances[i][j]
	return dist

def update_pheromones(pheromone, ants_location, n_nodes, distances, rho, ant):
	for i in range(n_nodes):
		for j in range(n_nodes):
				pheromone[i][j] *= rho
	ant_location = ants_location[ant]
	Lk = get_ant_distance(ants_location, ant, distances)
	for i in range(n_nodes):
		for j in range(n_nodes):			
			for k in range(1,len(ant_location)):
				if ant_location[k-1] == i and ant_location[k] == j:
					pheromone[i][j] += 0.00001
	return pheromone

def update_pheromones_global(pheromone, best_ants_location, n_nodes, distances, rho):
	for i in range(n_nodes):
		for j in range(n_nodes):
				pheromone[i][j] *= rho
	Lk = get_best_ant_distance(best_ants_location, distances)
	for i in range(n_nodes):
		for j in range(n_nodes):			
			for k in range(1,len(best_ants_location)):
				if best_ants_location[k-1] == i and best_ants_location[k] == j:
					pheromone[i][j] += 1.0/Lk
	return pheromone

def VRP(n_iteration=1, alpha=2, beta=2, rho=0.7):
	n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load = read_file()
	distances = get_distances(nodes_position)
	visbility = calculate_visibility(n_nodes, distances)
	pheromone = initialize_pheromones(n_nodes)
	n_ants = n_nodes - 1

	ant_distances = np.zeros((n_iteration, n_ants))
	best_sol_dist = np.zeros(n_iteration)

	min_dist = sys.float_info.max
	best_locs = None
	for iteration in range(n_iteration):
		ants_location, ants_capacity = initialize_location(n_ants, depot, init_capacity)
		for ant in range(n_ants):
			visited = np.zeros(n_nodes)
			visited[depot] = 1
			m = 0
			while m < n_tours:
				i = ants_location[ant][-1]
				feasible_nodes = np.where(visited == 0)[0]
				if feasible_nodes.shape[0] > 0:
					sum_val = 0
					for node in feasible_nodes:
						sum_val += ((pheromone[i][node])**alpha) * (visbility[i][node]**beta) 						
					p_transition = np.zeros(n_nodes)
					for j in range(n_nodes):
						if j in feasible_nodes:
							p_transition[j] = ((pheromone[i][j])**alpha) *(visbility[i][j]**beta)/sum_val
					
					random_num = np.random.uniform(0,1)
					for k, prob in enumerate(p_transition):
						random_num -= prob
						if random_num <= 0:
							new_city = k
							break

					if ants_capacity[ant] >= nodes_load[new_city] and new_city != depot:
						ants_capacity[ant] -= nodes_load[new_city]
						ants_location[ant].append(new_city)
						visited[new_city] = 1
					else:
						ants_location[ant].append(depot)
						ants_capacity[ant] = init_capacity
						m += 1
				else:
					break
			ant_distances[iteration][ant] = get_ant_distance(ants_location, ant, distances)
			# pheromone = update_pheromones(pheromone, ants_location, n_nodes, distances, rho, ant)
			# pheromone = update_pheromones(pheromone, ants_location, n_nodes, distances, rho, ant)
		
		best_ant = np.argmin(ant_distances[iteration])
		best_sol_dist[iteration] = ant_distances[iteration][best_ant]
		if best_sol_dist[iteration] < min_dist:
			min_dist = best_sol_dist[iteration]
			best_locs = ants_location[best_ant]
		pheromone = update_pheromones_global(pheromone, best_locs, n_nodes, distances, rho)
		
	return best_sol_dist, best_locs

def sliding_window(n_iteration, ys, k=20):
	ys_window = np.zeros(n_iteration)
	for i in range(n_iteration):
		k_prime = k
		if i<k:
			k_prim = i
		elif i >= n_iteration-k:
			k_prim = n_iteration-i-1
		s = 0
		for j in range (i-k_prim, i+k_prim+1):
			s += (k_prim+1-abs(i-j))
			ys_window[i]+=float((k_prim+1-abs(i-j))*ys[j])
		ys_window[i] /= s
	return ys_window

def plot_dist1(n_iteration=100):
	n_test = 6
	ys = np.zeros((n_test, n_iteration))
	a = [1, 1, 2, 2, 3, 3]
	b = [1, 2, 3, 4, 4, 5]
	r = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
	for i in range(n_test):
		ys[i], _ = VRP(n_iteration, a[i], b[i], r[5])

	fig = plt.figure()
	ax = fig.add_subplot(111)	
	for i in range(n_test):
		ax.plot([j+1 for j in range(n_iteration)], sliding_window(n_iteration, ys[i], k=20), 
			label="a="+str(a[i])+",b="+str(b[i])+",ro="+str(r[5]))

	ax.legend()
	ax.set_xlabel("iteration")
	ax.set_ylabel("distances")
	ax.set_title("Convergence plot")
	ax.figure.savefig("dist1.png")
	plt.close()

def plot_dist2(n_iteration=100):
	n_test = 6
	ys = np.zeros((n_test, n_iteration))
	a = [1, 1, 2, 2, 3, 3]
	b = [1, 2, 3, 4, 4, 5]
	r = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
	for i in range(n_test):
		ys[i], _ = VRP(n_iteration, a[5], b[5], r[i])

	fig = plt.figure()
	ax = fig.add_subplot(111)	
	for i in range(n_test):
		ax.plot([j+1 for j in range(n_iteration)], sliding_window(n_iteration, ys[i], k=20), 
			label="a="+str(a[5])+",b="+str(b[5])+",ro="+str(r[i]))
	ax.legend()
	ax.set_xlabel("iteration")
	ax.set_ylabel("distances")
	ax.set_title("Convergence plot")
	ax.figure.savefig("dist2.png")
	plt.close()

def draw_best(best_locs):
	_, _, _, _, nodes_position, _ = read_file()
	pos = {}
	nodes_label = []
	for i, node in enumerate(nodes_position):
		pos[i] = node
		nodes_label.append(str(i))
	graph = nx.DiGraph()
	graph.add_nodes_from(pos.keys())
	for n, p in pos.iteritems():
		graph.node[n]['pos'] = p
	for v in graph.nodes():
		graph.node[v]['label']= str(v)
	for i in range(1,len(best_locs)):
		graph.add_edge(best_locs[i-1], best_locs[i])
	
	nx.draw(graph, pos)
	node_labels = nx.get_node_attributes(graph,'label')
	nx.draw_networkx_labels(graph, pos, labels = node_labels)
	plt.savefig('best_locs.png')



def test(n_iteration=100):
	y, best_locs = VRP(100, 3, 5, 0.6)
	print best_locs
	n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load = read_file()
	distances = get_distances(nodes_position)
	print get_best_ant_distance(best_locs, distances)
	draw_best(best_locs)
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ys = sliding_window(n_iteration, y, k=20)
	ax.plot(range(n_iteration), ys)
	ax.set_xlabel("iteration")
	ax.set_ylabel("distances")
	ax.set_title("Convergence plot")
	plt.savefig('best.png')

# test()
# plot_dist1()
# plot_dist2()