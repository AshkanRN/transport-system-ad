from Queue import Queue
from priority_queue import PriorityQueue
from passenger import *



class Node:
    def __init__(self, vertex, cost, capacity, start_time, end_time):
        self.vertex = vertex
        self.cost = cost
        self.capacity = capacity
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"(to: {self.vertex}, cost: {self.cost}, cap: {self.capacity}, time: {self.start_time}-{self.end_time})"


class Graph:
    def __init__(self):
        self.adj_list = {}
        self.passenger_info = {}
        self.current_size = 0
        self.G = nx.Graph()
        self.edges = set()
        # self.passenger_queue = PassengerQueue()

    def add_vertex(self):
        self.adj_list[self.current_size] = []
        print(f"\nVertex {self.current_size} Created.")
        self.G.add_node(self.current_size)
        self.current_size += 1

    def add_edge(self, u, v, cost, capacity, start_time, end_time):

        if any(edge.vertex == v for edge in self.adj_list[u]):
            self.adj_list[u] = [edge for edge in self.adj_list[u] if edge.vertex != v]
            self.adj_list[v] = [edge for edge in self.adj_list[v] if edge.vertex != u]

        self.adj_list[u].append(Node(v, cost, capacity, start_time, end_time))
        self.adj_list[v].append(Node(u, cost, capacity, start_time, end_time))

        # for avoid to create duplicate edge: min(u, v), max(u, v)
        self.G.add_edge(min(u, v), max(u, v), weight = cost, capacity = capacity,
                        start_time = start_time, end_time = end_time)

        if 'usage' not in self.G[u][v]:
            self.G[u][v]['usage'] = 0

        self.edges.add((min(u, v), max(u, v), cost, capacity, start_time, end_time))


    def display_vertices(self):
        if self.current_size == 0:
            print("\nEMPTY\n")
            return
        print()
        for i in range(self.current_size):
            print(f"Vertex: {i}")


    def print_graph(self):
        if self.current_size == 0:
            print("\nEMPTY\n")
            return

        for u in range(self.current_size):
            i = 0
            print(f"V{u} :", end=" ")

            for edge in self.adj_list[u]:
                if i == len(self.adj_list[u]) - 1 :
                    print(edge, end=" ")
                else:
                    print(edge, end=" -> ")
                i += 1

            print("")


    def display_passengers(self):
        if not self.passenger_info:
            print("\nNO Passenger Yet.")
            return

        # print(self.passenger_info)
        for name,(edges,vertices) in self.passenger_info.items():
            print(f"{name}: {edges}   ,   {vertices} ")


    def highlight_edges(self, edges):

        pos = graphviz_layout(self.G, prog='sfdp')

        plt.figure(figsize=(12, 10))

        manager = plt.get_current_fig_manager()
        try:
            manager.window.wm_geometry("+600+0")
        except AttributeError:
            pass

        edge_colors = []
        for u, v in self.G.edges:
            if (u, v) in edges or (v, u) in edges:
                edge_colors.append('red')
            else:
                edge_colors.append('gray')

        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='skyblue',
            node_size=900,
            font_size=14,
            edge_color=edge_colors,
            width=2,
        )

        # edge_labels = nx.get_edge_attributes(self.G, 'weight')
        edge_labels = {
            (u, v): f"{d['weight']}, {d['capacity']}" for u, v, d in self.G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            self.G, pos,
            edge_labels=edge_labels,
            font_size=12,
            label_pos=0.5,
            rotate=False
        )

        plt.title("Graph with Highlighted Edges")
        plt.show()


    def mst_prim(self):
        if self.current_size == 0:
            print("Graph is Empty")
            return

        if not self.edges:
            print("There is no Edge Yet")
            return

        visited = [False] * self.current_size
        all_components = []
        total_cost = 0

        for start_vertex in range(self.current_size):
            if visited[start_vertex]:
                continue

            pq = PriorityQueue()
            component_edge = []
            component_cost = 0

            visited[start_vertex] = True

            for edge in self.adj_list[start_vertex]:
                pq.enqueue(edge.vertex, edge.cost, start_vertex)

            while not pq.is_empty():
                node = pq.dequeue()
                u = node.parent
                v = node.vertex
                cost = node.weight

                if visited[v]:
                    continue

                visited[v] = True
                component_edge.append((u, v, cost))
                component_cost += cost

                for edge in self.adj_list[v]:
                    if not visited[edge.vertex]:
                      pq.enqueue(edge.vertex, edge.cost, v)

            all_components.append((component_edge, component_cost))
            total_cost += component_cost

        # print(all_components)

        mst_edges = set()
        component_num = 1
        # all_component: [ ([(0, 3, 4), (3, 4, 6)], 10), ([(1, 5, 2), (5, 2, 5)], 7) ]
        for component in all_components:
            # component: ( [(0, 3, 4), (3, 4, 6)], 10)
            edges, cost = component
            # edges: [ (0, 3, 4), (3, 4, 6) ]  , cost: 10

            print(f"\nComponent {component_num} MST:")
            for u, v, c in edges:
                print(f"{u} -- {v} (cost: {c})")
                mst_edges.add((min(u, v), max(u, v)))

            print(f"Total cost of component {component_num}: {cost}")
            component_num += 1

        print(f"\n{mst_edges}")

        self.highlight_edges(mst_edges)


    def shortest_path(self, src, dest, passenger_s_time, passenger_e_time, consider_capacity=False,
                      consider_times=False, tsp_mode=False):

        if src not in self.adj_list:
            print("\nsrc vertex does not exist")
            return None

        if dest not in self.adj_list:
            print("\ndest vertex does not exist")
            return None

        visited = [False] * self.current_size
        distance = [float('inf')] * self.current_size
        parent = [-1] * self.current_size

        distance[src] = 0
        pq = PriorityQueue()
        pq.enqueue(src, 0, None)

        while not pq.is_empty():
            node = pq.dequeue()
            u = node.vertex

            if visited[u]:
                continue

            visited[u] = True

            if dest is not None and u == dest:
                break

            for edge in self.adj_list[u]:
                v = edge.vertex
                cost = edge.cost
                capacity = edge.capacity
                start_time = edge.start_time
                end_time = edge.end_time

                if consider_times:
                    if passenger_s_time < start_time or passenger_e_time > end_time:
                        continue

                if consider_capacity:
                    if capacity is None or capacity <= 0:
                        continue

                if not visited[v] and distance[u] + cost < distance[v]:
                    distance[v] = distance[u] + cost
                    parent[v] = u
                    pq.enqueue(v, distance[v], u)

        if dest is not None:
            if distance[dest] == float('inf'):
                if not tsp_mode:
                    string = ("No Alternative Path"
                              if consider_capacity
                              else "No Path")
                    print(f"\n{string} from {src} to {dest} at {passenger_s_time},{passenger_e_time} time")
                return float('inf') if tsp_mode else None

            if tsp_mode:
                return distance[dest], parent

            path = []
            curr = dest
            shortest_path_edges = []

            while curr != -1:
                path.append(curr)
                curr = parent[curr]

            path.reverse()

            for i in range(len(path)-1):
                shortest_path_edges.append((min(path[i], path[i+1]), max(path[i], path[i+1])))
            if consider_capacity:
                print("\nAlternative:")
            # print("\npath: ",path)
            print("\nedges: ",shortest_path_edges)
            print(f"Shortest path from {src} to {dest}: {' -> '.join(map(str, path))}")
            print(f"Total cost: {distance[dest]}")
            return shortest_path_edges, path
        else:
            if tsp_mode:
                return distance
            print(f"Shortest distances from node {src}:")
            for i in range(self.current_size):
                print(f"to {i}: {distance[i]}")
            return None


    def display_graph(self):
        if self.current_size == 0:
            print("\nEMPTY\n")
            return

        # pos: a dictionary that contains coordinate (x,y) of each vertex
        pos = graphviz_layout(self.G, prog='sfdp')


        if not self.G.edges:
            edge_colors = '#cccccc'
        else:
            usage_values = [self.G[u][v].get('usage', 0) for u, v in self.G.edges]
            max_usage = max(usage_values)

            if max_usage == 0:
                edge_colors = ['#cccccc' for _ in self.G.edges]
            else:
                edge_colors = [plt.cm.Reds(usage / max_usage) for usage in usage_values]

        plt.figure(figsize=(14, 12))

        manager = plt.get_current_fig_manager()
        try:
            manager.window.wm_geometry("+550+0")
        except AttributeError:
            pass

        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='skyblue',
            node_size=900,
            font_size=14,
            edge_color=edge_colors,
            width=2
        )

        # edge_labels = nx.get_edge_attributes(self.G, 'weight')

        # edge_labels = {
        #     (u, v): f"{d['weight']}, {d['capacity']}"
        #     for u, v, d in self.G.edges(data=True)
        # }

        edge_labels = {
            (u, v): f"{d['weight']}, {d['capacity']}, ({d['start_time']},{d['end_time']}) , {d['usage']}"
            for u, v, d in self.G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_size=12,
            label_pos=0.5,
            rotate=False
        )

        plt.title("Graph heatmap", fontsize=16)
        plt.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()




def get_neighbours(graph, vertex):
    neighbours = [edge.vertex for edge in graph.adj_list[vertex]]
    return neighbours


def check_radius_bfs(graph, radius):
    for start in range(graph.current_size):
        visited = [False] * graph.current_size
        distance = [-1] * graph.current_size
        queue = Queue()

        visited[start] = True
        distance[start] = 0
        queue.enqueue(start)

        while not queue.is_empty():
            front = queue.dequeue()
            vertex = front.value
            for neighbour in get_neighbours(graph, vertex):
                if not visited[neighbour]:
                    visited[neighbour] = True
                    distance[neighbour] = distance[vertex] + 1

                    if distance[neighbour] > radius:
                        return False

                    queue.enqueue(neighbour)

    return True


def dijkstra(graph, src, dest = None):
    if src not in graph.adj_list:
        print("\nsrc vertex does not exist")
        return None

    if dest is not None and dest not in graph.adj_list:
        print("\ndest vertex does not exist")
        return None

    visited = [False] * graph.current_size
    distance = [float('inf')] * graph.current_size
    parent = [-1] * graph.current_size

    distance[src] = 0
    pq = PriorityQueue()
    pq.enqueue(src, 0, None)

    while not pq.is_empty():
        node = pq.dequeue()
        u = node.vertex

        if visited[u]:
            continue

        visited[u] = True

        if dest is not None and u == dest:
            break

        for edge in graph.adj_list[u]:
            v = edge.vertex
            cost = edge.cost

            if not visited[v] and distance[u] + cost < distance[v]:
                distance[v] = distance[u] + cost
                parent[v] = u
                pq.enqueue(v, distance[v], u)

    if dest is not None:
        return distance[dest]

    return distance, parent



def build_cost_matrix(graph, start, destinations):

    nodes = [start] + destinations
    n = len(nodes)

    cost_matrix = [[float('inf')] * n for _ in range(n)]
    shortest_paths = dict()

    for i in range(n):
        dist, parent = dijkstra(graph, nodes[i])

        for j in range(n):
            dest = nodes[j]
            cost_matrix[i][j] = dist[nodes[j]]


            path = []
            current = dest
            if dist[dest] != float('inf'):
                while current != -1:
                    path.append(current)
                    current = parent[current]
                path.reverse()
            shortest_paths[(i, j)] = path


    return cost_matrix, nodes, shortest_paths



def tsp_dp(pos, visited, cost_matrix, n, memo, parent):
    # if all nodes visited
    if visited == (1 << n) - 1:
        return 0

    # memo is a dictionary that store (pos, visited)
    if (pos, visited) in memo:
        return memo[(pos, visited)]

    min_cost = float('inf')
    next_node = -1

    for nxt in range(n):
        # if nxt is not visited:
        if (visited & (1 << nxt)) == 0:
            cost = cost_matrix[pos][nxt] + tsp_dp(nxt, visited | (1 << nxt), cost_matrix, n, memo, parent)
            if cost < min_cost:
                min_cost = cost
                next_node = nxt

    memo[(pos, visited)] = min_cost
    parent[(pos, visited)] = next_node
    return min_cost


def tsp_with_path(cost_matrix, shortest_paths, nodes, start_index=0):
    n = len(cost_matrix)
    memo = {}
    parent = {}

    # 1 << start_index : start_index visited
    min_cost = tsp_dp(start_index, 1 << start_index, cost_matrix, n, memo, parent)

    path_indices = [start_index]
    visited = 1 << start_index
    pos = start_index

    while True:
        nxt = parent.get((pos, visited), -1)
        if nxt == -1:
            break
        path_indices.append(nxt)
        # nxt visited
        visited |= (1 << nxt)
        pos = nxt

    real_path = []
    for i in range( len(path_indices) - 1):
        u, v = path_indices[i], path_indices[i + 1]
        segment = shortest_paths[(u, v)]
        if i == 0:
            real_path += segment
        else:
            real_path += segment[1:]

    visiting_order = [ nodes[i] for i in path_indices ]

    return min_cost, visiting_order, real_path