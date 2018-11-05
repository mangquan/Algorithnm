class Node:
    def __init__(self,index):
        self.index = index
        self.next = None


class Graph:
    # Store the graph in Linked list
    # I included the head point, which will make the who process more intuitive.
    def __init__(self, vertexs):
        self.graphlist = []
        for i in range(vertexs):
            self.graphlist.append(Node(i))

    def add_edges(self, src, dest):
        node = Node(dest)
        node.next = self.graphlist[src].next
        self.graphlist[src].next = node

        node = Node(src)
        node.next = self.graphlist[dest].next
        self.graphlist[dest].next = node


    def print_graph(self):
        for i in self.graphlist:
            temp = i.next
            while temp:
             print(temp.index, '->',end='')
             temp = temp.next
            print('')


def dfs(nextnode, visited, graphlist):
    visited[nextnode.index] = True
    print(nextnode.index)

    temp = graphlist[nextnode.index].next

    while temp and not visited[temp.index]:
        dfs(temp,visited,graphlist)
        temp = temp.next



graph = Graph(8)

graph.add_edges(0, 1)
graph.add_edges(0, 2)
graph.add_edges(0, 3)

graph.add_edges(1, 4)

graph.add_edges(2, 4)
graph.add_edges(2, 5)
graph.add_edges(2, 6)

graph.add_edges(6, 7)

graph.print_graph()

visited = [False]*8

dfs(graph.graphlist[0],visited,graph.graphlist)





