class queen:
    def __init__(self, row, col, layer):
        self.layer = layer
        self.row = row
        self.col = col


def promising(arr, insertNode):
    if insertNode.layer == 0:
        return True

    for previousNode in range(insertNode.layer):
        if arr[previousNode].row == insertNode.row and arr[previousNode].col == insertNode.col:
            return False
        if insertNode.layer - arr[previousNode].layer == abs(
                insertNode.row - arr[previousNode].row) and insertNode.layer - arr[previousNode].layer == abs(
            insertNode.col - arr[previousNode].col):
            return False
        if insertNode.row == arr[previousNode].row and insertNode.layer - arr[previousNode].layer == abs(
                insertNode.col - arr[previousNode].col):
            return False
        if insertNode.col == arr[previousNode].col and insertNode.layer - arr[previousNode].layer == abs(
                insertNode.row - arr[previousNode].row):
            return False

    return True


def nqueens(currentArray, currentLayer, n, totalNumber):
    if currentLayer < n:
        for row in range(n):
            for col in range(n):
                insertNode = queen(row, col, currentLayer)
                if promising(arr=currentArray, insertNode=insertNode):
                    currentArray[insertNode.layer] = insertNode

                    if insertNode.layer == n - 1:
                        totalNumber[0] = totalNumber[0] + 1
                        print(totalNumber[0])
                        # for node in currentArray:
                        #     print(node.layer, node.row, node.col)
                    else:
                        nqueens(currentArray, currentLayer + 1, n, totalNumber)


arr = [0] * 8

totalNumber = [0]

nqueens(currentArray=arr, currentLayer=0, n=8, totalNumber=totalNumber)
