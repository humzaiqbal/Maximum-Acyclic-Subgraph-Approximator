import Queue
import random



class Graph():
	"""
	Graph class capable of doing all graph operations in the same run time as that of the 170 model of a graph 
	as specified by https://piazza.com/class/idjqdfip9c574h?cid=237
	"""
	def __init__(self):
		self.__outgoing_edges = {}
		self.__incoming_edges = {}
		self.__outgoing_edges = {}
		self.__vertices = set()
		self.__edges = set()
		self.__source_vertices = set()

    # Handles the case where we have isolated vertices with no incoming or outgoing edges
	def add_vertex(src):
		if src not in self.vertices:
			self.__vertices.add(src)
			self.__source_vertices.add(src)

	def add_edge(self, src, dst):
		if src not in self.vertices:
			self.__vertices.add(src)
			self.__outgoing_edges[src] = set()
			self.__outgoing_edges[src].add(dst)
		# Checking to see if there is a potential source vertex the first time
		if src not in self.source_vertices:
			if src not in self.__incoming_edges.keys():
				self.__source_vertices.add(src)
		else:
			self.__outgoing_edges[src].add(dst)
		if dst not in self.vertices:
			self.__vertices.add(dst)
			self.__incoming_edges[dst] = set()
			self.__incoming_edges[dst].add(src)
		else:
			self._incoming_edges[dst].add(src)
		# Check to see if a vertex that we thought was a source isn't one 
		if dst in __self.source_vertices:
			self.__source_vertices.remove(dst)
		self.__edges.add((src, dst))

	def check_edge(self, src, dst):
		return (src, dst) in self.__edges


	def get_vertices(self):
		return self.__vertices

	def get_children(self, src):
		if src in self.__outgoing_edges.keys():
			return self.__outgoing_edges(src)
		else:
			return set()

	def get_parents(self, dst):
		if dst in self.__incoming_edges.keys():
			return self.__incoming_edges(dst)
		else:
			return set()
	def get_source_vertices(self):
		return self.__source_vertices

	def get_edges(self):
		return self.__edges()

	def remove_edge(self, src, dst):
		if self.check_edge(src, dst):
			self.__edges.remove((src, dst))
			self.__incoming_edges[dst].remove(src)
			self.__outgoing_edges[src].remove(dst)

	def generate_subgraph(self, vertices):
		c = Graph()
		for vertex in vertices:
			c.add_vertex(vertex)
		for edge in self.__edges:
			if edge[0] in vertices and edge[1] in vertices:
				c.add_edge(edge[0], edge[1])
		return c 

	def get_outgoing_edges(self, src):
		return self.__outgoing_edges[src]

	def get_incoming_edges(self, dst):
		return self.__incoming_edges[dst]



def topological_sort(Graph):
	"""
	Code to topologically sort a graph taken from Wikipedia
	"""
	L = []
	S = Graph.get_source_vertices()
	while len(S) > 0:
		new_node = S.pop()
		L.append(new_node)
		for node in Graph.get_outgoing_edges():
			Graph.remove_edge(new_node, node)
			if len(Graph.get_incoming_edges(node)) == 0:
				S.add(node)
	return L





def has_cycle(Graph):
	"""
	Uses DFS to determine whether or not we have a cycle 
	"""
	visited = set()
	fringe = Queue.LifoQueue()
	for vertex in Graph.get_source_vertices():
		visited.add(vertex)
		fringe.put(vertex)
	while not fringe.empty():
		new_vertex = fringe.get()
		visited.add(new_vertex)
		for neighbor in Graph.get_children(new_vertex):
			if neighbor in visited:
				return True
			fringe.put(neighbor)
	return False


def num_edges(Graph, subgraph_1, subgraph_2):
	"""
	Helper function which determines the number of edges 
	from sugraph_1 to subgraph_2. Will be used by other 
	functions to determine the ordering among subgraphs
	"""
	num_edges = 0
	for elem in subgraph_1:
		for elem2 in subgraph_2:
			if Graph.check_edge(elem, elem2):
				num_edges += 1



def naive_algorithm(Graph):
	"""
	Naive algorithm which partitions our graph into two sets based on the order 
	of the source vertex and destination vertex of the edge. 
	This algorithm is guaranteed to produce something acyclic 
	"""
	num_vertices = len(Graph.get_vertices())
	if not has_cycle(Graph):
		return Graph 
	else:
		set_1 = []
		set_2 = []
		for edge in Graph.get_edges():
			if edge[0] < edge[1]:
				set_1.add(edge)
			else:
				set_2.add(edge)
		if num_edges(Graph, set_1, set_2) > num_edges(Graph, set_2, set_1):
			return set_1 
		else:
			return set_2

"""
The four functions below are helper functions which give me a way of 
picking my vertices when going through my permutation algorithm 
generate_permutation rather than just random vertex each time 
"""

def W_i(Graph, vertex):
	return max(len(Graph.get_children(vertex)) , len(Graph.get_parents(vertex))) 

def w_i(Graph, vertex):
	return min(len(Graph.get_children(vertex)) , len(Graph.get_parents(vertex))) 

def w_hat_i(Graph, vertex):
	return abs(len(Graph.get_children(vertex)) - len(Graph.get_parents(vertex))) 

def r_i(Graph, vertex):
	# Handles case where the vertex is either a source or sink
	# In which case one of the ratios is undefined and the other will be 0
	if not (Graph.get_children(vertex) and Graph.get_parents(vertex)):
		return 0
	else:
		out_to_in = float(Graph.get_children(vertex)) / Graph.get_parents(vertex)
		in_to_out = float(Graph.get_parents(vertex)) / Graph.get_children(vertex)
		return max(out_to_in, in_to_out)


"""
Helper function for determing the order to sort my vertices by 
when picking them for my permutation algorithm. 
See section 5 of http://www.shlomir.com/papers/acyclic.pdf
for more details
"""
def sort_vertices_by(Graph, vertices, func):
	output_list = []
	if func == "None":
		return list(vertices)
	if func == "W_i":
		output_list = [W_i(Graph, vertex) for vertex in vertices] 


	elif func == "w_i":
		output_list = [w_i(Graph, vertex) for vertex in vertices] 

	elif func == "w_hat_i":
		output_list = [w_hat_i(Graph, vertex) for vertex in vertices] 

	elif func == "r_i":
		output_list = [w_hat_i(Graph, vertex) for vertex in vertices]
	lst = list(vertices)
	vert_to_func = {}
	new_vertices = []
	for index in range(len(output_list)):
		vert_to_func[output_list[index]] = lst[index]
	output_list.sort()
	for index in range(len(output_list)):
		new_vertices.append(vert_to_func[output_list[index]])
	return new_vertices









def generate_permutation(Graph, func="None"):
	"""
	Subroutine which picks the permutation of the vertices which would be best. 
	Intutively this works because of topological sorting 
	"""
	S = Graph 
	l = 1
	u = len(G.get_vertices)
	lst = sort_vertices_by(Graph, G.get_vertices(), func)
	permutation = [0] * u 
	while l > u:
		i = random.choice(lst)
		if len(G.get_children(i)) > len(G.get_parents(i)):
			permutation[i] = l 
			l += 1 
		else:
			permutation[i] = u 
			u -= 1 
		lst.remove(i)


def make_graph(input_Graph, permutation):
	new_graph = Graph()
	possible_edges = [(src, dest) for src in permutation for dest in permutation if src < dest]
	for edge in possible_edges:
		if input_Graph.check_edge(edge):
			new_graph.add_edge(edge)
	return new_graph




def random_algorithm(Graph):
	"""
	Second algorithm in which we employ randomization 
	(not guaranteed to return an acyclic subgraph on any particular iteration 
	so we run it until it gives us one) 
	I also return the permutation so I may use it later for my genetic algorithm
	"""
	is_cycle = True 
	potential_graph = None
	permutation = None
	while is_cycle:
		set_1 = set()
		set_2 = set()
		for vertex in Graph.get_vertices():
			choice = random.choice([0, 1])
			if choice:
				set_1.add(vertex)
			else:
				set_2.add(vertex)
		subgraph_1 = Graph.generate_subgraph(set_1)
		subgraph_2 = Graph.generate_subgraph(set_2)
		permutation_1 = generate_permutation(subgraph_1)
		permutation_2 = generate_permutation(subgraph_2) 
		potential_graph_1 = make_graph(Graph, permutation_1)
		potential_graph_2 = gmake_graph(Graph, permutation_2)

		if len(potential_graph_1.get_edges()) > len(potential_graph_2.get_edges()):
			potential_graph = potential_graph_1
		else:
			potential_graph = potential_graph_2 
			 
		if not has_cycle(potential_graph):
			is_cycle = False 
	return potential_graph, permutation 


"""
Genetic algorithm which takes a previous permutation and tweaks it a little to see if 
I get something better. (Performs tweaking a number of times as determined by num_times)
Guaranteed to be acyclic 
"""
def local_search(Graph, permutation, num_times=100):
	local_optimum = Graph 
	num_elements = len(permutation)
	# Making a copy so I still have my original permutation in case the mutation performs worse
	perm_to_change = permutation[:]
	while num_times:
		indices = random.sample(range(num_elements), 2) 
		perm_to_change[indices[0]], permutation[indices[1]] = perm_to_change[indices[1]], perm_to_change[indices[0]]
		mutated_graph = get_edges(Graph, perm_to_change)
		if len(mutated_graph.get_edges()) > len(Graph.get_edges()) and not has_cycle(mutated_graph):
			local_optimum = mutated_graph 
		num_times -= 1 
	return local_optimum


