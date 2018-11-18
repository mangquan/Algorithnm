import numpy as np


def return_index(i, j):
    # using the index starts with 1
    if i > j:
        return int(i * (i - 1) / 2 + j - 1)
    else:
        return int(j * (j - 1) / 2 + i - 1)


class Graph:
    def __init__(self, i, connectivity):

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


def promising(graph, circleRecord, nextpoint, totalWeight):
    if nextpoint == 0:
        return True

    switch = True

    tempWeight = 0

    if nextpoint == graph.num_of_vertex - 1:
        if graph.get_matrix()[0][nextpoint] == 99:
            switch = False

    if nextpoint < graph.num_of_vertex - 1:
        if graph.get_matrix()[circleRecord[nextpoint - 1]][circleRecord[nextpoint]] == 99:
            switch = False

    for i in range(nextpoint):
        if circleRecord[i] == circleRecord[nextpoint]:
            switch = False

    for i in range(nextpoint + 1):
        if i == graph.num_of_vertex - 1:
            tempWeight = tempWeight + graph.get_matrix()[circleRecord[i]][circleRecord[0]]
        else:
            tempWeight = tempWeight + graph.get_matrix()[circleRecord[i]][circleRecord[i + 1]]

    if tempWeight > totalWeight[0]:
        switch = False

    return switch


def hamitonian(graph, circleRecord, totalWeight, index):
    if (promising(graph, circleRecord, index, totalWeight)):
        if index == graph.num_of_vertex - 1:
            tempWeight = 0
            for i in range(len(circleRecord)):
                if i == graph.num_of_vertex - 1:
                    tempWeight = tempWeight + graph.get_matrix()[circleRecord[i]][circleRecord[0]]
                else:
                    tempWeight = tempWeight + graph.get_matrix()[circleRecord[i]][circleRecord[i + 1]]
                print(circle[i], '->', end='')
            totalWeight[0] = tempWeight
            print(totalWeight[0])

        else:
            for i in range(1, graph.num_of_vertex):
                circleRecord[index + 1] = i
                hamitonian(graph, circleRecord, totalWeight, index + 1)

# assume the unconnected pair of vertexs has weight equal to 99
test = Graph(5, 'Fully')

circle = [0] * 5

totalWeight = [99]

hamitonian(test, circle, totalWeight, 0)
