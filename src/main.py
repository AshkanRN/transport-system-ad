from graph import *
from passenger import *



def main():

    graph = Graph()
    passenger_queue = PassengerQueue()
    radius_limit = 3

    while True:
        print("\n[0]: Exit"
              "\n[1]: Add Vertex"
              "\n[2]: Add Edge"
              "\n[3]: Print"
              "\n[4]: Shortest Path"
              "\n[5]: Display Graph"
              "\n[6]: MST"
              "\n[7]: Release The Capacity"
              "\n[8]: Passenger List"
              "\n[9]: Print Passenger Queue"    
              "\n[10]: Check Passenger Queue"
              "\n[11]: Multi-Destination Shortest Path")

        command = input("--> ")

        if command == '0':
            break

        elif command == '1':
            graph.add_vertex()

        elif command == '2':
            if graph.current_size < 2:
                print("\nAdd Vertex first !!")
                continue

            try:
                src, dest = map(int, input("Source and Destination: ").split())

                if src not in graph.adj_list:
                    print("\nSource Vertex does not exist !!!")
                    continue

                if dest not in graph.adj_list:
                    print("\nDestination Vertex does not exist !!!")
                    continue

                if src == dest:
                    print("Self-loop !!")
                    continue

                try:
                    cost, capacity, start_time, end_time = (
                        map(int, input("Cost, Capacity, Start Time, End Time: ").split()))

                except ValueError:
                    print("\nINVALID !, Should Enter 4 Number")
                    continue

                if not cost > 0:
                    print("Edge's Cost Can not be less than/equal to zero")
                    continue

                if start_time < 0 or start_time > end_time or end_time > 23:
                    print("Invalid Start and End Time")
                    continue

                graph.add_edge(src, dest, cost, capacity, start_time, end_time)

                if graph.current_size < 5:
                    print(f"\nThe Edge {src} <--> {dest} added.")

                else:
                    if check_radius_bfs(graph, radius_limit):
                        graph.add_edge(src, dest, cost, capacity, start_time, end_time)

                        print(f"\nThe Edge {src} <--> {dest} added.")
                    else:
                        # removing the Edge
                        graph.adj_list[src] = [edge for edge in graph.adj_list[src] if edge.vertex != dest]
                        graph.adj_list[dest] = [edge for edge in graph.adj_list[dest] if edge.vertex != src]
                        graph.G.remove_edge(min(src, dest), max(src, dest))

                        print("\nThe Edge Can not be Added, (RADIUS LIMIT)")

            except ValueError:
                print("\nInvalid input! Please enter numbers only.")

        elif command == '3':
            graph.print_graph()


        elif command == '4':
            if graph.current_size < 2:
                print("\nAdd Vertex first!")
                continue

            try:
                src = int(input("Source vertex: "))
                dest = int(input("Destination vertex: "))
                s_time, e_time = map(int, input("Start and End Time: ").split())

                if src == dest:
                    print("Source and Destination can not be The same")
                    continue

                if s_time < 0 or s_time > e_time or e_time > 23:
                    print("Invalid Start and End Time")
                    continue

                shortest_path = graph.shortest_path(src, dest, s_time, e_time,False, True)
                # shortest_path is a tuple with 2 element,
                # the first element is edges in Shortest Path and the second is Vertices
                # Example: ([(0, 1), (0, 2)], [1, 0, 2])
                if shortest_path:
                    cmd = input("Wanna Reserve The Route? [y/n]: ")

                    if cmd == "y" or cmd.lower() == "yes":
                        name = input("Enter Name: ")
                        reserve_status = reserve_route(graph, name, shortest_path, passenger_queue)

                        # reserve_route() return Values:
                        # 0: Error Or "Return to Main Menu" is Selected, 1: Shortest Path Reserved,
                        # 2: Enqueued, 3: Alternative Shortest Path

                        if reserve_status == 1:
                            graph.highlight_edges(shortest_path[0])

                        elif reserve_status == 3:
                            alternative_shortest_path = graph.shortest_path(src,dest, s_time, e_time,
                                                                            True,
                                                                            True)
                            if alternative_shortest_path:
                                cmd2 = input("Wanna Reserve The Alternative Route? [y/n]: ")

                                if cmd2 == "y" or cmd2.lower() == "yes":
                                    reserve_status = reserve_route(graph, name, alternative_shortest_path, passenger_queue)

                                    if reserve_status == 1:
                                        graph.highlight_edges(alternative_shortest_path[0])

                    elif cmd == "n" or cmd.lower() == "no":
                        pass

                    else:
                        print("Invalid")


            except ValueError:
                print("\nInvalid Input")


        elif command == '5':
            graph.display_graph()

        elif command == '6':
            graph.mst_prim()

        elif command == '7':
            if not graph.passenger_info:
                print("No Passenger Yet.")
                continue

            name = input("Passenger Name:")
            release_route_capacity(graph, name)

        elif command == '8':
            graph.display_passengers()

        elif command == '9':
            passenger_queue.print_queue()

        elif command == '10':
            first_passenger = passenger_queue_process(graph, passenger_queue)

            if first_passenger:
                graph.highlight_edges(first_passenger.edges)

        elif command == '11':
            if graph.current_size <= 0:
                print("empty")
                continue

            start = int(input("Start node: "))
            destinations = list(map(int, input("Destinations: ").split()))

            cost_matrix, nodes, shortest_paths = build_cost_matrix(graph, start, destinations)

            min_cost, visiting_order, real_path = tsp_with_path(cost_matrix, shortest_paths,
                                                                nodes, start_index=0)

            if min_cost == float('inf'):
                print("\nThere is No path to visit all destinations.")
                continue

            edge_list = [(real_path[i], real_path[i + 1]) for i in range(len(real_path) - 1)]


            print("Cost:", min_cost)
            print("Visiting order:", visiting_order)
            print("Path:", edge_list)

            graph.highlight_edges(edge_list)


        else:
            print("\nInvalid")




if __name__ == '__main__':
    main()
