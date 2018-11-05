class Node:
    def __init__(self, index, weight):
        self.index = index
        self.next = None
        self.weight = weight


class Graph:
    # Store the graph in Linked list
    # I included the head point, which will make the who process more intuitive.
    def __init__(self, vertexs):
        self.graphlist = []
        for i in range(vertexs):
            self.graphlist.append(Node(i, 0))

    def add_edges(self, src, dest, weight):
        # always keep the headpoint at the first position and the new points will be directly added to the headpoint
        node = Node(dest, weight)
        node.next = self.graphlist[src].next
        self.graphlist[src].next = node

        node = Node(src, weight)
        node.next = self.graphlist[dest].next
        self.graphlist[dest].next = node

    def print_graph(self):
        for i in self.graphlist:
            temp = i.next
            while temp:
                print(temp.index, '->', end='')
                temp = temp.next
            print('')


def dijktra(graph):
    # Dijkstra is very similar to to prim except the standard of shortest path is based on the start point
    # instead of the whole subset like prim did.

    numOfvertex = len(graph.graphlist)

    # use to record the whole distance
    wholeDistance = []

    # record the transmitting point, which means by passing this point the distance will be shorten
    shortestPathRecord = []

    # find the shortest distance for every point
    for source in range(numOfvertex):
        subSet = []
        # record the distance from start point to destination point
        distance = [99] * numOfvertex
        # start from itself
        distance[source] = 0

        # Transmitting point
        pathRecord = [-1] * numOfvertex

        # loop until all points are emerged into the whole set
        while len(subSet) != numOfvertex:
            shortest = 99
            nextVisit = -1

            # find the next visit point
            for i in range(len(distance)):
                if distance[i] < shortest and i not in subSet:
                    shortest = distance[i]
                    nextVisit = i

            subSet.append(nextVisit)
            distance[nextVisit] = shortest

            nextVisitPoint = graph.graphlist[nextVisit]

            # to see if the adding point will help to shorten the distance between the start point and the rest
            while nextVisitPoint:
                if distance[nextVisitPoint.index] > distance[nextVisit] + nextVisitPoint.weight:
                    distance[nextVisitPoint.index] = distance[nextVisit] + nextVisitPoint.weight
                    pathRecord[nextVisitPoint.index] = nextVisit
                nextVisitPoint = nextVisitPoint.next

        wholeDistance.append(distance)
        shortestPathRecord.append(pathRecord)

    for i in wholeDistance:
        print(i)










graph = Graph(8)

graph.add_edges(0, 1, 1)
graph.add_edges(0, 2, 2)
graph.add_edges(0, 3, 3)

graph.add_edges(1, 4, 4)

graph.add_edges(3, 6, 3)

graph.add_edges(2, 4, 5)
graph.add_edges(2, 5, 6)
graph.add_edges(2, 6, 7)

graph.add_edges(6, 7, 8)

dijktra(graph)
