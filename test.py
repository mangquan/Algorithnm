import numpy as np
import copy


def print_shortest_records(record, start, destination):
    if start == destination:
        return 'You already there.'

    path = []

    previous_point = record[start][destination]

    while previous_point != start:
        path.append(previous_point)
        previous_point = record[start][previous_point]

    path.append(previous_point)

    for i in reversed(path):
        print(i, " -> ", end='')

    print(destination)


def return_index(i, j):
    # using the index starts with 1
    if i > j:
        return int(i * (i - 1) / 2 + j - 1)
    else:
        return int(j * (j - 1) / 2 + i - 1)


def find_min_index(dist, visited_vertexs):
    smallest = 99

    for v in range(len(dist)):
        if dist[v] < smallest and v not in visited_vertexs:
            smallest = dist[v]
            min_index = v

    return min_index


class Graph:
    def __init__(self, i, connectivity, adjmatrix):

        # Bug of i == 3, need to be fixed
        if connectivity == 'Sparse':
            self.num_of_vertex = i
            self.num_of_edges = int((i + 1) * i / 2)
            self.graphArray = np.full((1, self.num_of_edges), float('INF'))[0].tolist()

            edge_index = []

            do_not_insert = []

            for k in range(self.num_of_vertex):
                temp = return_index(k + 1, k + 1)
                do_not_insert.append(temp)
                self.graphArray[temp] = 0

            ramdon_length = np.random.randint((self.num_of_vertex - 1) * (self.num_of_vertex - 2) / 2 - 1,
                                              self.num_of_edges - len(do_not_insert) - 1)

            while len(edge_index) < ramdon_length:
                rindex = np.random.randint(0, self.num_of_edges - 1)
                if rindex not in edge_index and rindex not in do_not_insert:
                    edge_index.append(rindex)

            for i in edge_index:
                self.graphArray[i] = np.random.randint(1, 10)

        elif connectivity == 'Fully':
            self.num_of_vertex = i
            self.num_of_edges = (i + 1) * i / 2
            self.graphArray = (np.random.rand(1, int(self.num_of_edges)) * 10 + 1).astype(int)[0].tolist()

            for k in range(self.num_of_vertex):
                temp = return_index(k + 1, k + 1)
                self.graphArray[temp] = 0

        elif connectivity == 'Customize':
              self.num_of_vertex = len(adjmatrix)
              self.num_of_edges = int((self.num_of_vertex + 1)*self.num_of_vertex/2)
              self.graphArray = []

              for i in  range(len(adjmatrix)):
                for j in  range(i+1):
                    if i == j:
                        self.graphArray.append(0)
                    else:
                        self.graphArray.append(adjmatrix[i][j])


    def get_matrix(self):

        matrix = np.array([])

        for i in range(self.num_of_vertex):
            temp = []
            for j in range(self.num_of_vertex):

                if i == j:
                    temp.append(0)

                if j > i:
                    temp.append(self.graphArray[return_index(j + 1, i + 1)])

                if i > j:
                    temp.append(self.graphArray[return_index(i + 1, j + 1)])

            matrix = np.append(matrix, temp)

        return matrix.reshape(self.num_of_vertex, -1)

    def get_list(self):

        graphDict = []

        for i in range(self.num_of_vertex):
            temp = {}
            for j in range(self.num_of_vertex):

                if j > i and self.graphArray[return_index(j + 1, i + 1)] != float('INF'):
                    temp[j] = self.graphArray[return_index(j + 1, i + 1)]

                if i > j and self.graphArray[return_index(i + 1, j + 1)] != float('INF'):
                    temp[j] = self.graphArray[return_index(i + 1, j + 1)]

            graphDict.append(temp)

        return graphDict


def dijkstra_matrix(graph):
    shortest_path = []

    path_records = []

    number_of_vertexs = graph.num_of_vertex

    matrix = graph.get_matrix()

    for i in range(number_of_vertexs):

        visited_vertexs = []

        singe_path_record = [-1] * number_of_vertexs

        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if temp_path[dest_point] > matrix[v][dest_point] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[v][dest_point] + temp_path[v]
                    singe_path_record[dest_point] = v

        shortest_path.append(temp_path)
        path_records.append(singe_path_record)

    return shortest_path, path_records


def dijkstra_linkedList(graph):
    shortest_path = []

    path_records = []

    number_of_vertexs = graph.num_of_vertex

    matrix = graph.get_list()

    for i in range(len(matrix)):
        matrix[i][i] = 0

    for i in range(number_of_vertexs):

        visited_vertexs = []

        singe_path_record = [-1] * number_of_vertexs

        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if dest_point in matrix[v] and temp_path[dest_point] > matrix[v][dest_point] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[v][dest_point] + temp_path[v]
                    singe_path_record[dest_point] = v

        shortest_path.append(temp_path)
        path_records.append(singe_path_record)

    return shortest_path, path_records


def dijkstra_array(graph):
    shortest_path = []

    path_records = []

    number_of_vertexs = graph.num_of_vertex

    matrix = graph.graphArray

    for i in range(number_of_vertexs):

        visited_vertexs = []

        singe_path_record = [-1] * number_of_vertexs

        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if temp_path[dest_point] > matrix[return_index(v + 1, dest_point + 1)] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[return_index(v + 1, dest_point + 1)] + temp_path[v]
                    singe_path_record[dest_point] = v

        shortest_path.append(temp_path)
        path_records.append(singe_path_record)

    return shortest_path, path_records


def floyd_matrix(in_graph):
    graph = copy.deepcopy(in_graph)

    number_of_vertex = graph.num_of_vertex

    path_track = []

    for i in range(number_of_vertex):
        path_track.append(np.full((1, number_of_vertex), i)[0].tolist())

    shortest_path = graph.get_matrix()
    # starts here
    matrix = shortest_path

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if shortest_path[star][destination] > matrix[star][mid] + matrix[mid][destination]:
                    shortest_path[star][destination] = matrix[star][mid] + matrix[mid][destination]
                    path_track[star][destination] = mid

    return shortest_path, path_track


def floyd_array(graph):
    number_of_vertex = graph.num_of_vertex

    path_track = []

    for i in range(number_of_vertex):
        path_track.append(np.full((1, number_of_vertex), i)[0].tolist())

    shortest_path = copy.deepcopy(graph.get_matrix())
    # starts here
    matrix = copy.deepcopy(graph.graphArray)

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if shortest_path[star][destination] > matrix[return_index(star + 1, mid + 1)] + matrix[return_index(mid + 1, destination + 1)]:
                    shortest_path[star][destination] = matrix[return_index(star + 1, mid + 1)] + matrix[
                        return_index(mid + 1, destination + 1)]
                    matrix[return_index(star + 1, destination + 1)] = matrix[return_index(star + 1, mid + 1)] + matrix[
                        return_index(mid + 1, destination + 1)]
                    path_track[star][destination] = mid

    return shortest_path, path_track


def floyd_linkedList(graph):
    number_of_vertex = graph.num_of_vertex

    path_track = []

    for i in range(number_of_vertex):
        path_track.append(np.full((1, number_of_vertex), i)[0].tolist())

    shortest_path = copy.deepcopy(graph.get_matrix())
    # starts here
    matrix = copy.deepcopy(graph.get_list())
    for i in range(len(matrix)):
        matrix[i][i] = 0

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if mid in matrix[star] and destination in matrix[mid] and shortest_path[star][destination] > \
                        matrix[star][mid] + matrix[mid][destination]:
                    shortest_path[star][destination] = matrix[star][mid] + matrix[mid][destination]
                    matrix[star][destination] = matrix[star][mid] + matrix[mid][destination]
                    path_track[star][destination] = mid

    return shortest_path, path_track




testCase1 = [[float('INF'),  float('INF'),  float('INF'), 29,  float('INF'),  float('INF'),  float('INF'),  float('INF')],[float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'), 11, 11,  float('INF')],[float('INF'),  float('INF'),  float('INF'), 12,  float('INF'),  5,  5,  float('INF'),],[29,  float('INF'), 12,  float('INF'),  5,  float('INF'), 13,  float('INF')],[float('INF'),  float('INF'),  float('INF'),  5,  float('INF'),  float('INF'),  7, 11],[float('INF'), 11,  5,  float('INF'),  float('INF'),  float('INF'),  float('INF'), 17],[float('INF'), 11,  5, 13,  7,  float('INF'),  float('INF'),  float('INF')],[float('INF'),  float('INF'),  float('INF'),  float('INF'), 11, 17,  float('INF'),  float('INF')]]


testCase2 = [[float('INF'), 11, 14,  float('INF'),  8,  float('INF'), 29, 28,  float('INF'),  float('INF'), 14,  float('INF'),],
[11,  float('INF'), 12,  float('INF'),  6,  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),],
[14, 12,  float('INF'), 18, 13, 13,  float('INF'),  float('INF'), 25,  float('INF'),  float('INF'), 16,],
[float('INF'),  float('INF'), 18,  float('INF'),  float('INF'),  float('INF'), 27, 17,  9, 25,  float('INF'),  float('INF'),],
[8,  6, 13,  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'), 22],
[float('INF'),  float('INF'), 13,  float('INF'),  float('INF'),  float('INF'),  float('INF'), 15,  5,  float('INF'),  float('INF'),  float('INF'),],
[29,  float('INF'),  float('INF'), 27,  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),],
[28,  float('INF'),  float('INF'), 17,  float('INF'), 15,  float('INF'),  float('INF'),  5,  9,  float('INF'),  float('INF'),],
[float('INF'),  float('INF'), 25,  9,  float('INF'),  5,  float('INF'),  5,  float('INF'),  float('INF'), 25,  float('INF'),],
[float('INF'),  float('INF'),  float('INF'), 25,  float('INF'),  float('INF'),  float('INF'),  9,  float('INF'),  float('INF'),  float('INF'),  float('INF'),],
[14,  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'), 25,  float('INF'),  float('INF'),  float('INF'),],
[float('INF'),  float('INF'), 16,  float('INF'), 22,  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),  float('INF'),]]

testGraph1 = Graph(8, 'Customize', testCase1)

testGraph2 = Graph(12, 'Customize', testCase2)

print('Using the dijkstra with adject matrix:')
d1, p1 = dijkstra_matrix(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)


d2, p2 = dijkstra_matrix(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)
print('-----------------------------------------------------------------------')
print('Using the dijkstra with linkedlist:')
d1, p1  = dijkstra_linkedList(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)

d2, p2 = dijkstra_linkedList(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)
print('-----------------------------------------------------------------------')
print('Using the dijkstra with 1 d array:')
d1, p1  = dijkstra_array(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)

d2, p2 = dijkstra_array(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)
print('-----------------------------------------------------------------------')

print('Using the floyd with matrix:')
d1, p1  = floyd_matrix(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)

d2, p2 = floyd_matrix(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)
print('-----------------------------------------------------------------------')
print('Using the floyd with list:')
d1, p1  = floyd_linkedList(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)

d2, p2 = floyd_linkedList(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)
print('-----------------------------------------------------------------------')

print('Using the floyd with 1d array:')
d1, p1  = floyd_array(testGraph1)
print('The shortest distance of test case 1 is', d1)
print('The distance between v1 -v8 is', d1[0][7])
print('The path is')
print_shortest_records(p1, 0, 7)

d2, p2 = floyd_array(testGraph2)
print('The shortest distance of test case 2 is', d2)
print('The distance between v12 -v10 is', d2[11][9])
print('The path is')
print_shortest_records(p2, 11, 9)