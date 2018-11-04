import numpy as np
import copy
import time
import matplotlib.pyplot as plt

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
        print(i," -> ", end='')

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
    def __init__(self, i, connectivity):

        # Bug of i == 3, need to be fixed
        if connectivity == 'Sparse':
            self.num_of_vertex = i
            self.num_of_edges = int((i + 1)*i/2)
            self.graphArray = np.full((1, self.num_of_edges), float('INF'))[0].tolist()

            edge_index = []

            do_not_insert = []

            for k in range(self.num_of_vertex):
                temp = return_index(k + 1, k + 1)
                do_not_insert.append(temp)
                self.graphArray[temp] = 0

            ramdon_length = np.random.randint((self.num_of_vertex - 1)*(self.num_of_vertex - 2)/2 -1, self.num_of_edges - len(do_not_insert) -1)

            while len(edge_index) < ramdon_length:
                rindex = np.random.randint(0, self.num_of_edges-1)
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

    number_of_vertexs = graph.num_of_vertex

    matrix = graph.get_matrix()

    for i in range (number_of_vertexs):

        visited_vertexs = []


        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if temp_path[dest_point] > matrix[v][dest_point] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[v][dest_point] + temp_path[v]

        shortest_path.append(temp_path)




def dijkstra_linkedList(graph):
    shortest_path = []


    number_of_vertexs = graph.num_of_vertex

    matrix = graph.get_list()

    for i in range(len(matrix)):
        matrix[i][i] = 0

    for i in range (number_of_vertexs):

        visited_vertexs = []

        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if dest_point in matrix[v] and temp_path[dest_point] > matrix[v][dest_point] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[v][dest_point] + temp_path[v]


        shortest_path.append(temp_path)



def dijkstra_array(graph):

    shortest_path = []


    number_of_vertexs = graph.num_of_vertex

    matrix = graph.graphArray

    for i in range (number_of_vertexs):

        visited_vertexs = []

        temp_path = np.full((1, number_of_vertexs), float('INF'))[0].tolist()

        temp_path[i] = 0

        while len(visited_vertexs) < number_of_vertexs:
            v = find_min_index(temp_path, visited_vertexs)

            visited_vertexs.append(v)

            for dest_point in range(number_of_vertexs):
                if temp_path[dest_point] > matrix[return_index(v+1, dest_point+1)] + temp_path[v] and dest_point not in visited_vertexs:
                    temp_path[dest_point] = matrix[return_index(v+1, dest_point+1)] + temp_path[v]


        shortest_path.append(temp_path)


def floyd_matrix(in_graph):

    graph = copy.deepcopy(in_graph)

    number_of_vertex = graph.num_of_vertex
    shortest_path = graph.get_matrix()
    # starts here
    matrix = shortest_path

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if shortest_path[star][destination] > matrix[star][mid] + matrix[mid][destination]:
                    shortest_path[star][destination] = matrix[star][mid] + matrix[mid][destination]


def floyd_array(graph):

    number_of_vertex = graph.num_of_vertex

    shortest_path = copy.deepcopy(graph.get_matrix())
    # starts here
    matrix = copy.deepcopy(graph.graphArray)

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if shortest_path[star][destination] > matrix[return_index(star+1, mid+1)] + matrix[return_index(mid+1, destination+1)]:
                    shortest_path[star][destination] = matrix[return_index(star+1, mid+1)] + matrix[return_index(mid+1, destination+1)]
                    matrix[return_index(star+1, destination+1)] = matrix[return_index(star+1, mid+1)] + matrix[return_index(mid+1, destination+1)]


def floyd_linkedList(graph):

    number_of_vertex = graph.num_of_vertex

    shortest_path = copy.deepcopy(graph.get_matrix())
    # starts here
    matrix = copy.deepcopy(graph.get_list())
    for i in range(len(matrix)):
        matrix[i][i] = 0

    for mid in range(number_of_vertex):
        for star in range(number_of_vertex):
            for destination in range(number_of_vertex):
                if mid in matrix[star] and destination in matrix[mid] and shortest_path[star][destination] > matrix[star][mid] + matrix[mid][destination]:
                    shortest_path[star][destination] = matrix[star][mid] + matrix[mid][destination]
                    matrix[star][destination] = matrix[star][mid] + matrix[mid][destination]


testCase = [100, 200, 300]

y = []

for i in testCase:
    print('Now it is runing', i, ' nodes')
    temp = []
    testGraphFull = Graph(i, 'Sparse')
    start = time.time()
    floyd_array(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    start = time.time()
    # testGraphFull = Graph(i, 'Fully')
    floyd_matrix(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    start = time.time()
    # testGraphFull = Graph(i, 'Fully')
    floyd_linkedList(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    start = time.time()
    # testGraphFull = Graph(i, 'Fully')
    dijkstra_array(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    start = time.time()
    # testGraphFull = Graph(i, 'Fully')
    dijkstra_matrix(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    start = time.time()
    # testGraphFull = Graph(i, 'Fully')
    floyd_linkedList(testGraphFull)
    end = time.time()
    temp.append(end - start)
    print(end)

    y.append(temp)

for i in range(6):
    y_axis = []

    for k in y:
        y_axis.append(k[i])


    plt.plot(testCase, y_axis)

    plt.title(i+6)

    plt.xlabel('number of nodes')
    plt.ylabel('Time')

    plt.show()

print(y)