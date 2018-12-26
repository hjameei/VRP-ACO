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

def initialize_nodes():
	graph = nx.DiGraph()
	data_file = open('data.txt', 'r')
	for i, line in enumerate(data_file):
		values = line.split(' ')
		if i == 0:
			m = int(values[0])
			q = int(values[1])
		else:
			if int(values[3])==0:
				depot=True
			else:
				depot=False
			graph.add_node(int(values[0])-1, pos=[int(values[1]), int(values[2])], q = int(values[3]), depot=depot)
	data_file.close()
	return m, q, graph

def initialize_ants(depot_node, n_ants, init_capacity):
	ants_location = {}
	ants_capacity = np.zeros(n_ants)
	for i in range(n_ants):
		ants_location[i] = [depot_node]
		ants_capacity[i] = init_capacity
	return ants_location, ants_capacity

def graph_draw(graph):
	pos = nx.get_node_attributes(graph,'pos')
	q = nx.get_node_attributes(graph,'q')
	nx.draw(graph, pos)
	plt.show()

def get_distances(pos):
	positions = np.array(pos.values())
	distances = pdist(positions)
	distances = squareform(distances)
	return distances

def get_distances2(nodes_position):
	distances = pdist(nodes_position)
	distances = squareform(distances)
	return distances

def calculate_visibility(pos, n):
	distances = get_distances(pos)
	visbility = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i != j:
				visbility[i][j] = 1.0/distances[i][j]
	return visbility

def get_Lk(ant_location, distances):
	dist = 0.0
	for m in range(1,len(ant_location)):
		i = ant_location[m-1]
		j = ant_location[m]
		dist += distances[i][j]
	return dist

def get_Lks(ants_location, n, m, distances):
	Lk = np.zeros(m)
	for ant in range(m):
		ant_location = ants_location[ant]
		Lk[ant] = get_Lk(ant_location, distances)
	return Lk

def get_total_distance(ants_location, n, m, distances):
	Lk = get_Lks(ants_location, n, m, distances)
	return np.sum(Lk)

def get_best_ant_distance(best_ants_location, distances):
	dist = 0.0
	for k in range(1, len(best_ants_location)):
		i = best_ants_location[k-1]
		j = best_ants_location[k]
		dist += distances[i][j]
	return dist

def get_tour_qs(ant_location, qs):
	tour_q = 0.0
	for location in ant_location:
		tour_q += qs[location]
	return tour_q

def get_qs(ants_location, m, qs):
	tour_qs = np.zeros(m)
	for ant in range(m):
		ant_location = ants_location[ant]
		tour_qs[ant] = get_tour_qs(ant_location, qs)
	return tour_qs

def get_total_qs(ants_location, m, qs):
	tour_qs = get_qs(ants_location, m, qs)
	return np.sum(tour_qs)

def update_pheromones(pheromones, ants_location, n, m, distances, qs, rho):
	Lk = get_Lks(ants_location, n, m, distances)
	tour_qs = get_qs(ants_location, m, qs)
	total_dist = (ants_location, n, m, distances)
	for i in range(n):
		for j in range(n):
			pheromones[i][j] *= rho
			for ant in range(m):
				ant_location = ants_location[ant]
				for k in range(1,len(ant_location)):
					if ant_location[k-1] == i and ant_location[k] == j:
						pheromones[i][j] += 1.0/Lk[ant]
	return pheromones

def update_pheromones_rho(pheromones, ants_location, n, m, distances, qs, rho):
	for i in range(n):
		for j in range(n):
				pheromones[i][j] = rho*pheromones[i][j] 
	return pheromones

def update_ant_route_pheromone(pheromones, ants_location, n, m, distances, rho, ant):
	ant_location = ants_location[ant]
	Lk = get_Lk(ant_location, distances)
	for i in range(n):
		for j in range(n):			
			for k in range(1,len(ant_location)):
				if ant_location[k-1] == i and ant_location[k] == j:
					pheromones[i][j] += 1.0/Lk
	return pheromones

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


def VRP(n_iteration, alpha, beta, rho):
	m, q, graph = initialize_nodes()
	n = len(graph.nodes())
	pos = nx.get_node_attributes(graph, 'pos')
	depots = nx.get_node_attributes(graph,'depot')	
	qs = nx.get_node_attributes(graph,'q')
	for key, val in depots.iteritems():
		if val == True:
			depot = key
			break

	solution_distances = np.zeros(n_iteration)
	distances = get_distances(pos)
	visbility = calculate_visibility(pos, n)
	pheromones = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			pheromones[i][j] = 1.0/(n*n)

	ant_distances = np.zeros((n_iteration, m))

	min_dist = sys.float_info.max
	best_locs = None
	for iteration in range(n_iteration):
		# init ant locations
		ants_location, ants_capacity = initialize_ants(depot, m, q)
		visited = np.zeros(n)
		# visited[depot] = 0

		# found_sol = False
		ant_completed = np.zeros(m)
		if iteration > 0:
			pheromones = update_pheromones_rho(pheromones, ants_location, n, m, distances, qs, rho)
		
		for ant in range(m):
			while ant_completed[ant]==0:
				# find current path and current node and feasible nodes
				current_path = ants_location[ant]
				i = current_path[-1]

				feasible_nodes =[]
				tmp_nodes = np.where(visited == 0)[0]
				for node in tmp_nodes:
					if ants_capacity[ant] > qs[node]:
						feasible_nodes.append(node)
				
				# print visbility[i][node]**beta
				if len (feasible_nodes) > 1:
					# calculate transitions
					sum_val = 0
					for node in feasible_nodes:
						sum_val += ((pheromones[i][node])**alpha) * (visbility[i][node]**beta) 						
					p_transition = np.zeros(n)
					for j in range(n):
						if j in feasible_nodes:
							p_transition[j] = ((pheromones[i][j])**alpha) *(visbility[i][j]**beta)/sum_val

					random_num = np.random.uniform(0,1)
					for k, prob in enumerate(p_transition):
						random_num -= prob
						if random_num <= 0:
							new_city = k
							break

					visited[new_city] = 1
					ants_capacity[ant] -= qs[new_city]
					ants_location[ant].append(new_city)
				else:
					ant_completed[ant] = 1
					ants_location[ant].append(depot)
					ant_distances[iteration][ant] = get_Lk(ants_location[ant], distances)
			pheromones = update_ant_route_pheromone(pheromones, ants_location, n, m, distances, rho, ant)

		solution_distances[iteration] = get_total_distance(ants_location, n, m, distances)
		if solution_distances[iteration] < min_dist:
			min_dist = solution_distances[iteration]
			best_locs = ants_location

	locs = []
	for ant in range(m):
		for node in best_locs[ant]:
			locs.append(node)
	return solution_distances, locs
	
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
	ax.figure.savefig("old-dist1.png")
	plt.close()
		
def plot_dist2(n_iteration=100):
	n_test = 6
	ys = np.zeros((n_test, n_iteration))
	a = [1, 1, 2, 2, 3, 3]
	b = [1, 2, 3, 4, 4, 5]
	r = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
	for i in range(n_test):
		ys[i], _ = VRP(n_iteration, a[3], b[3], r[i])

	fig = plt.figure()
	ax = fig.add_subplot(111)	
	for i in range(n_test):
		ax.plot([j+1 for j in range(n_iteration)], sliding_window(n_iteration, ys[i], k=20), 
			label="a="+str(a[3])+",b="+str(b[3])+",ro="+str(r[i]))
	ax.legend()
	ax.set_xlabel("iteration")
	ax.set_ylabel("distances")
	ax.set_title("Convergence plot")
	ax.figure.savefig("old-dist2.png")
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
	plt.savefig('old-best_locs.png')

def test(n_iteration=120):
	y, best_locs = VRP(n_iteration, 2, 4, 0.8)
	print best_locs
	n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load = read_file()
	distances = get_distances2(nodes_position)
	print get_best_ant_distance(best_locs, distances)
	draw_best(best_locs)
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ys = sliding_window(n_iteration, y, k=20)
	ax.plot(range(n_iteration), ys)
	ax.set_xlabel("iteration")
	ax.set_ylabel("distances")
	ax.set_title("Convergence plot")
	plt.savefig('old-best.png')

# test()
# plot_dist1()
# plot_dist2()
