def promising(index, weight, value, maxProfit, totalWeight, currentWeight, currentValue):
    if weight[index] + currentWeight[0] > totalWeight:
        return False
    temp = currentValue[0]
    while index <= len(weight) - 1 and weight[index] + currentWeight[0] < totalWeight:
        temp = temp + value[index]
        currentWeight[0] = currentWeight[0] + weight[index]
        index = index + 1

    if index < len(weight):
        temp = temp + (totalWeight - currentWeight[0]) * (value[index] / weight[index])

    print(temp, maxProfit)

    return temp > maxProfit[0]


def knapsack(weight, value, totalWeight, carryItem, maxProfit, currentWeight, currentValue, root):
    if root > len(weight) - 1 or weight[root] + currentWeight[0] > totalWeight:
        return currentValue

    if currentValue[0] >= maxProfit[0]:
        maxProfit[0] = currentValue[0]

    temp = [currentValue[0]]
    temp2 = [currentValue[0]]

    tempWeight1 = [currentWeight[0]]
    tempWeight2 = [currentWeight[0]]

    if promising(root, weight, value, maxProfit, totalWeight, currentWeight, currentValue):
        temp[0] = temp[0] + value[root]
        tempWeight1[0] = tempWeight1[0] + weight[root]

        if temp[0] >= maxProfit[0] and tempWeight1[0] < totalWeight:
            maxProfit[0] = temp[0]

        temp = knapsack(weight, value, totalWeight, carryItem, maxProfit, tempWeight1, temp, root + 1)

        temp2 = knapsack(weight, value, totalWeight, carryItem, maxProfit, tempWeight2, temp2, root + 1)

    if temp[0] > temp2[0]:
        carryItem[root] = True
        return temp

    else:
        carryItem[root] = False
        return temp2


weight = [0, 2, 5, 7, 3, 1]

value = [0, 20, 30, 35, 12, 3]

totalWeight = 9

maxProfit = [0]

currentWeight = [0]

currentValue = [0]

carryItem = [False]*len(weight)

w = knapsack(weight, value, totalWeight, carryItem, maxProfit, currentWeight, currentValue, 0)

# print(promising(index=0, weight=weight, value=value, maxProfit=maxProfit, totalWeight=totalWeight,
#                 currentWeight=currentWeight, currentValue=currentValue))

print(w[0], carryItem)
