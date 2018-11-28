def partition(arr, low, high, piviot):
    i = low
    j = high
    location = low

    while i != j:

        while arr[j] >= piviot and i < j:
            if arr[j] == piviot:
                temp = arr.pop(j)
                arr.insert(low, temp)
                i = i + 1
            else:
                j = j - 1

        while arr[i] <= piviot and i < j:
            if arr[i] == piviot:
                temp = arr.pop(i)
                arr.insert(low, temp)
            else:
                i = i + 1

        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    arr[location], arr[i] = arr[i], arr[location]
    return i


def pair_nuts(nuts, bolts, low, high):
    if high > low:
        i = partition(nuts, low, high, bolts[high])

        partition(bolts, low, high, nuts[i])

        pair_nuts(nuts, bolts, low, i - 1)

        pair_nuts(nuts, bolts, i + 1, high)


nuts = [1, 2, 3, 4, 5]

bolts = [5, 2, 3, 1, 4]

pair_nuts(nuts, bolts, 0, 4)

print(nuts)

print(bolts)
