import copy


class node:
    def __init__(self, nodeWeight, level, nodeProfit, item):
        self.nodeWeight = nodeWeight
        self.level = level
        self.nodeProfit = nodeProfit
        self.bound = 0
        self.item = item

    def calculatebound(self, totalweight, weight, value):
        if self.level>len(weight) or self.nodeWeight > totalweight:
            self.bound = 0

        else:
            tempWeight = self.nodeWeight
            tempLevel = self.level
            self.bound = self.nodeProfit
            while tempLevel + 1 < len(weight) and tempWeight + weight[tempLevel + 1] < totalweight:
                self.bound = self.bound + value[tempLevel + 1]

                tempWeight = tempWeight + weight[tempLevel + 1]

                tempLevel = tempLevel + 1

            if tempLevel + 1 < len(weight):
                self.bound = self.bound + (totalweight - tempWeight) * (value[tempLevel + 1] / weight[tempLevel + 1])


def knapsack_best(totalweight, weight, value):
    item = [False] * len(weight)

    maxProfit = 0

    item[0] = True

    root = node(0, 0, 0, item)

    queue = [root]

    root.calculatebound(totalweight, weight, value)

    takeItem = []

    while len(queue):

        biggestBound = 0
        nextVisit = -1

        for nextPoint in range(len(queue)):
            if queue[nextPoint].bound > biggestBound:
                biggestBound = queue[nextPoint].bound
                nextVisit = nextPoint

        nextVisitPoint = queue.pop(nextVisit)

        if nextVisitPoint.bound > maxProfit:
            level = nextVisitPoint.level + 1
            itemArryTake = copy.deepcopy(nextVisitPoint.item)
            if level < len(weight):
                itemArryTake[level] = True
                takeNextItem = node(nextVisitPoint.nodeWeight + weight[level], level,
                                    nextVisitPoint.nodeProfit + value[level], itemArryTake)
                takeNextItem.calculatebound(totalWeight, weight, value)
            if level >= len(weight):
                takeNextItem = node(nextVisitPoint.nodeWeight, level,
                                    nextVisitPoint.nodeProfit, itemArryTake)
                takeNextItem.calculatebound(totalWeight, weight, value)

            if takeNextItem.nodeWeight <= totalWeight and takeNextItem.nodeProfit > maxProfit:
                maxProfit = takeNextItem.nodeProfit
                takeItem = takeNextItem.item

            if takeNextItem.bound > maxProfit:
                queue.append(takeNextItem)

            itemArryNotTake = copy.deepcopy(nextVisitPoint.item)

            notTakeNextItem = node(nextVisitPoint.nodeWeight, level, nextVisitPoint.nodeProfit, itemArryNotTake)

            notTakeNextItem.calculatebound(totalWeight, weight, value)

            if notTakeNextItem.bound > maxProfit:
                queue.append(notTakeNextItem)

    return maxProfit, takeItem


weight = [0, 2, 5, 7, 3, 1]

value = [0, 20, 30, 35, 12, 3]

totalWeight = 13

maxProfit, takenItem = knapsack_best(totalWeight, weight, value)

print(maxProfit)

for i in takenItem:
    print(i)
