import numpy as np
import sys
def return_index(i, j):
    # using the index starts with 1
    # The graph class store the graph in compressed 1d array, this function will find the right index of that array.
    # It's basically a arithmetic progression( for the lower triangle)

    if i > j:
     return int(i * (i - 1) / 2 + j - 1)
    else:
        return int(j * (j - 1) / 2 + i - 1)



class Graph:
    #This graph class can output a ramdon graph in 3 diffrent kinds of storage ways. But all of them start with the compressed array.
     # 1. Matrix
     # 2. Compressed Array
     # 3. Dict
    def __init__(self, i, connectivity):

        # Bug of i == 3, need to be fixed
        if connectivity == 'Sparse':
            self.num_of_vertex = i

            # To initiate the compressed array.
            self.num_of_edges = int((i + 1)*i/2)
            self.graphArray = np.full((1, self.num_of_edges), float('INF'))[0].tolist()

            # To store the edge that will be initiated, since it's a sparse graph.
            edge_index = []

            # do not insert edges like 11 22 33
            do_not_insert = []

             # find the index for those 11 22 33
            for k in range(self.num_of_vertex):
                temp = return_index(k + 1, k + 1)
                do_not_insert.append(temp)
                self.graphArray[temp] = 0

            # decide how many edges we want ramdonly, but make sure it's connected, which means at least n-1 points are fully connected.
            ramdon_length = np.random.randint((self.num_of_vertex - 1)*(self.num_of_vertex - 2)/2 -1, self.num_of_edges - len(do_not_insert) -1)

            # ramdonly decide which edges shall we insert
            while len(edge_index) < ramdon_length:
                rindex = np.random.randint(0, self.num_of_edges-1)
                if rindex not in edge_index and rindex not in do_not_insert:
                    edge_index.append(rindex)

            #insert these edges
            for i in edge_index:
                self.graphArray[i] = np.random.randint(1, 10)

        elif connectivity == 'Fully':
            self.num_of_vertex = i
            self.num_of_edges = (i + 1) * i / 2
            self.graphArray = (np.random.rand(1, int(self.num_of_edges)) * 10 + 1).astype(int)[0].tolist()

            # for fully connected graph, each edge shall have a weight
            for k in range(self.num_of_vertex):
                temp = return_index(k + 1, k + 1)
                self.graphArray[temp] = 0

    def get_matrix(self):
        # This function will return the graph in matrix form


        matrix = np.array([])

        for i in range(self.num_of_vertex):
            temp = []
            for j in range(self.num_of_vertex):

                if i == j:
                    temp.append(99)

                if j > i:
                    temp.append(self.graphArray[return_index(j + 1, i + 1)])

                if i > j:
                    temp.append(self.graphArray[return_index(i + 1, j + 1)])

            matrix = np.append(matrix, temp)

        return matrix.reshape(self.num_of_vertex, -1)

    def get_list(self):
        # This function will return the graph in list form
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


def prim(graph):
    # This function will find the minial spanning tree for the graph using prim
    # Prim will establish two set, one to store the points that already be merged in tree, one to store the points waiting for insert.
    # For each iteration, we will find the shortest edge that connect these two sets.

    adjMatrix = graph.get_matrix()

    # mstSet is the set to store the points that have been merged.
    #distance records the minimal distance between two sets.
    mstSet = [False]*graph.num_of_vertex
    distance = [99]*graph.num_of_vertex

    # source will record the start point to the end point.
    source = [-1]*graph.num_of_vertex

    #start from the 0 point
    distance[0] = 0
    source[0] = 0

    while False in mstSet:

        shortest = 99

        nextPoint = -1

        # find the shortest edge
        for i in range(len(distance)):
            if shortest > distance[i] and mstSet[i] == False:
                shortest = distance[i]
                nextPoint = i

        mstSet[nextPoint] = True

        # Update the distance between two sets, since new point has been joined.
        for j in range(len(adjMatrix[nextPoint])):
            if distance[j] > adjMatrix[nextPoint][j] and mstSet[j]== False:
                distance[j] = adjMatrix[nextPoint][j]
                source[j] = nextPoint


    for dest in range(len(source)):
        print(source[dest],' -> ', dest, ' distance:', distance[dest])

    print('total cost:', np.sum(distance))


test = Graph(10,'Sparse')

print(test.get_matrix())

prim(test)






