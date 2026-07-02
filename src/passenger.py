import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from networkx.drawing.nx_agraph import graphviz_layout


class PassengerQueueNode:
    def __init__(self, name, edges,vertices):
        self.name = name
        self.edges = edges
        self.vertices = vertices
        self.next = None

    def __str__(self):
        return f"{self.name}: {self.edges}   ,   {self.vertices}"

class PassengerQueue:
    def __init__(self):
        self.front = None
        self.rear = None

    def is_empty(self):
        return self.front is None

    def enqueue(self, name, edges, vertices):
        new_node = PassengerQueueNode(name, edges, vertices)

        if self.is_empty():
            self.rear = self.front = new_node

        else:
            self.rear.next = new_node
            self.rear = new_node

    def dequeue(self):
        if self.is_empty():
            print("Passengers Queue is Empty")
            return None

        temp = self.front
        self.front = self.front.next

        if self.is_empty():
            self.rear = None

        return temp

    def get_front(self):
        return self.front

    def print_queue(self):
        if self.is_empty():
            print("empty")
            return

        curr = self.front
        print("\n")
        while curr:
            print(curr)
            curr = curr.next



def reserve_route(graph, name, edges_vertices, passenger_queue):
    # edges_vertices is a tuple with 2 element, the first element is edges in SP and the second is Vertices
    norm_edges = [normalize_edge(u, v) for u, v in edges_vertices[0]]

    # reserve_route() return Values:
    # 0: Error Or "Return to Main Menu" is Selected, 1: Shortest Path Reserved,
    # 2: Enqueued, 3: Alternative Shortest Path

    if name.lower() in graph.passenger_info:
        print("\nThis passenger is Already on a Route")
        return 0

    if not check_capacity(graph, norm_edges):
        print("\nThe route does not have enough capacity.")

        while True:
            print("\n[1]: Enter To the Queue"
                  "\n[2]: Alternative Route"
                  "\n[3]: Return To Main Menu")
            cmd = input("--> ")
            if cmd == '1':
                passenger_queue.enqueue(name , norm_edges, edges_vertices[1])
                print("\nEnqueued")
                return 2

            elif cmd == '2':
                return 3

            elif cmd == '3':
                return 0

            else:
                print("\nInvalid !")


    for u, v in norm_edges:
        decrease_capacity(graph, u, v,graph.G)

    graph.passenger_info[name.lower()] = (norm_edges,edges_vertices[1])
    # Passenger_info is a dictionary, key is name of the passenger
    # The value is a tuple like edge_vertices

    print("\nReserved successfully.")
    return 1





def animate_edge_traversal(g, vertex_list, steps_per_edge=25, interval=10):

    pos = graphviz_layout(g, prog='sfdp')
    # pos: a dictionary that contains coordinate (x,y) of each vertex

    fig, ax = plt.subplots(figsize=(12, 10))

    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("+600+0")  # X = 600px, Y = 0px
    except AttributeError:
        pass

    edge_list = [ (vertex_list[i], vertex_list[i+1]) for i in range(len(vertex_list) - 1) ]

    node_colors = []
    for node in g.nodes():
        if node == vertex_list[0]:
            node_colors.append('lightskyblue')  # Light sky blue for START vertex in the path
        elif node == vertex_list[-1]:
            node_colors.append('lightgreen')  # Light green for END vertex in the path
        elif node in vertex_list:
            node_colors.append('red')  # red fot OTHER vertices that ARE IN the path
        else:
            node_colors.append('#adacac')  # Light gray for Vertices that is not in the path


    path_points = []
    for u, v in edge_list:
        x_vals = np.linspace(pos[u][0], pos[v][0], steps_per_edge)
        y_vals = np.linspace(pos[u][1], pos[v][1], steps_per_edge)
        points = list(zip(x_vals, y_vals))
        path_points.extend(points)


    def update(i):
        ax.clear()

        nx.draw(
            g,
            pos,
            node_size=900,
            font_size=14,
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            ax=ax
        )


        if i < len(path_points):
            x, y = path_points[i]
            ax.plot(x, y, 'ro', markersize=15)
            # ro : red circle (o shape)

        if i > 0:
            traversed_points = path_points[0 : i + 1]
            x_coords = [ p[0] for p in traversed_points ]
            y_coords = [ p[1] for p in traversed_points ]
            ax.plot(x_coords, y_coords, color='red', linewidth=2)

    ani = animation.FuncAnimation(fig, update, frames=len(path_points), interval=interval, repeat=False)

    plt.show()


def release_route_capacity(graph, name):
    if not graph.passenger_info:
        print("NO Passenger Yet.")
        return

    name = name.lower()
    if name not in graph.passenger_info:
        print("\nPassenger not found.")
        return

    vertices_path = graph.passenger_info[name][1]
    animate_edge_traversal(graph.G, vertices_path)

    for u, v in graph.passenger_info[name][0]:
        increase_capacity(graph, u, v, graph.G)

    del graph.passenger_info[name]
    print("\nCapacity released.")



def passenger_queue_process(graph, passenger_queue):

    first_passenger = passenger_queue.get_front()
    if not first_passenger:
        print("passengers Queue is Empty")
        return False

    norm_edges = [normalize_edge(u, v) for u, v in first_passenger.edges]

    if not check_capacity(graph, norm_edges):
        print("The Route of first passenger has not enough Capacity Yet")
        return False

    else:
        for u, v in norm_edges:
            decrease_capacity(graph, u, v,graph.G)

        graph.passenger_info[first_passenger.name.lower()] = (norm_edges, first_passenger.vertices)
        print("\nReserved successfully.")
        passenger_queue.dequeue()
        return first_passenger



def decrease_capacity(graph, u, v, networkx_g):

    if networkx_g[u][v]['capacity'] > 0:
        networkx_g[u][v]['capacity'] -= 1

    networkx_g[u][v]['usage'] += 1

    for edge in graph.adj_list[v]:
        if edge.vertex == u:
            if edge.capacity > 0:
                edge.capacity -= 1
            break
    for edge in graph.adj_list[u]:
        if edge.vertex == v:
            if edge.capacity > 0:
                edge.capacity -= 1
            break

def increase_capacity(graph, u, v, networkx_g):

    networkx_g[u][v]['capacity'] += 1

    if networkx_g[u][v]['usage'] > 0:
        networkx_g[u][v]['usage'] -= 1

    for edge in graph.adj_list[v]:
        if edge.vertex == u:
            if edge.capacity >= 0:
                edge.capacity += 1
            break
    for edge in graph.adj_list[u]:
        if edge.vertex == v:
            if edge.capacity >= 0:
                edge.capacity += 1
            break


def check_capacity(graph, passenger_edge):
    for u, v in passenger_edge:
        for edge in graph.adj_list[u]:
            if edge.vertex == v:
                if edge.capacity <= 0:
                    return False
    return True

def normalize_edge(u, v):
    return min(u, v), max(u, v)