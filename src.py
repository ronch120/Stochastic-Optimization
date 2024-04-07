"""
Nitay Vilner
Tomer Gal
Guy Green
Ron Chechik
Tal Dasht
"""

from typing import List
from datetime import datetime
import pandas as pd
import numpy as np #for the graph
import matplotlib.pyplot as plt # for graph

def maximizeWeights(points: List[List[int]]) -> List[int]:
    # idea: for each point, find heaviest route that runs through it. maximize route's weight by raising point's weight.

    # data used in algorithm
    weights = [1] * len(points)

    # step 1 - sort points by x and then y - O(nlogn)
    points.sort(key = lambda point: (point[0], -point[1]))

    # step 2 - find longest route using LIS algorithm - O(nlogn)
    reversed_points = [(p[1],p[0]) for p in points] # changing between x and y for the lis algorithm
    heaviest_chain = [(p[1],p[0]) for p in LIS(reversed_points)] # again replace the x and y back to its right place
    longestRoute = len(heaviest_chain)

    # step 3 - find line that bestly approximates longest route by using Least Squares method.
    line = least_squares(heaviest_chain)

    # step 4 - sort points by their distance from the line
    indexedPoints = list(enumerate(points))
    indexedPoints.sort(reverse= True, key=lambda point: np.abs(line['slope'] * point[1][0] - point[1][1] + line['intercept']) / np.sqrt(line['slope']**2 + 1))

    """This graph illustrates all the points that make up the longest route, as well as the line resulting
    from the least squares method"""
    # printGraph(points,heaviest_chain,  [point[1] for point in indexedPoints[:250]], line=line)

    # step 5 - for each point, find heaviest route that runs through it using Guy Jacobson and Kiem-Phong Vo's HIS algorithm - O(n**2logn)
    ran = [point[0] for point in indexedPoints]

    time4Hundred = datetime.now()
    for i, p in enumerate(ran):
        # test run-time
        # if i % 100 == 0:
        #     now = datetime.now()
        #     print("index: " + str(i) + " || " + str(now - time4Hundred))
        #     time4Hundred = now

        heaviestRoute = HIS(points, weights, p)
        weights[p] = longestRoute - heaviestRoute + 1

    return weights,points


#######################################################################################################################
        
def LIS(nums):
    # Initialize an array to store LIS indices
    lis_indices = [0] * len(nums)
    # Initialize an array to store predecessor indices
    predecessors = [-1] * len(nums)
    # Initialize variables to keep track of LIS length and ending index
    lis_length = 1
    lis_end_index = 0

    # Function to perform binary search to find the position to insert the element
    def binary_search(val, end):
        left, right = 0, end
        while left < right:
            mid = (left + right) // 2
            if nums[lis_indices[mid]] < val:
                left = mid + 1
            else:
                right = mid
        return left

    # Iterate through the array and update LIS indices and predecessors
    for i in range(1, len(nums)):
        if nums[i] > nums[lis_indices[lis_length - 1]]:
            lis_indices[lis_length] = i
            predecessors[i] = lis_indices[lis_length - 1]
            lis_length += 1
            lis_end_index = i
        else:
            # Find the position to insert num in lis_indices
            index = binary_search(nums[i], lis_length - 1)
            lis_indices[index] = i
            if index > 0:
                predecessors[i] = lis_indices[index - 1]

    # Reconstruct the LIS
    lis_sequence = []
    index = lis_end_index
    while index != -1:
        lis_sequence.append(nums[index])
        index = predecessors[index]

    return lis_sequence[::-1]


#######################################################################################################################

def HIS(points:List[List[int]], weights: List[int], p: int) -> int:
    """
    Because the algorithm finds the heaviest subsequence for all points array,
    giving to a specific point a big weights promises that the algorithm finds
    heaviest subsequence going through that point.
    We can then return point's weight back to 1.
    """
    MAX_VAL = 1000
    weights[p] = MAX_VAL

    # HIS holds y values with accumulating heaviest route
    his = [(points[0][1], weights[0])]
    for i in range(1, len(points)):
        y = points[i][1]
        
        # initialize variables used in algorithm
        next = findPositionHIS(his, y)      # find position of already visited point 'next' such that next'y >= p'y
        prev = next - 1
        prevWeight = 0
        nextY, nextWeight = float('inf'), 0
        
        if prev >= 0:
            prevWeight = his[prev][1]
        if next < len(his):
            nextY, nextWeight = his[next]
        
        # Guy Jacobson and Kiem-Phong Vo's HIS algorithm
        while next < len(his) and prevWeight + weights[i] >= nextWeight:
            his.pop(next)

            if next < len(his):
                nextY, nextWeight = his[next]
        
        if next == len(his) or y < nextY:
            his.insert(next, (y, prevWeight + weights[i]))
    
    return his[-1][1] - MAX_VAL + 1


#######################################################################################################################

def findPositionHIS(his: List[List[int]], y: int) -> int:
    # Find position using binary search
    L, R = 0, len(his) - 1
    while L <= R:
        m = (L + R) // 2

        if his[m][0] == y:
            return m

        if his[m][0] < y:
            L = m + 1
        
        else:
            R = m - 1

    return L


#######################################################################################################################

def least_squares(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"slope" : m,"intercept": c}


#######################################################################################################################

def readData(sheetName, filePath = "data.xlsx"):
    data = pd.read_excel(filePath, sheet_name=sheetName)
    return list(zip(data['X'], data['Y']))


def writeData(points, weights, sheetName, filePath = "data.xlsx"):
    table =dict(zip(points, weights)) # mapping between the point to the weight we got for it.
    data = pd.read_excel(filePath, sheet_name=sheetName)
    new_weights = []
    for i in range(len(data['X'])):
        new_weights.append(table[(data['X'][i],data['Y'][i])]) # adding the weight of the point according the way it on the excel
    data['W'] = new_weights
    with pd.ExcelWriter(filePath, engine='openpyxl', mode='a') as writer:
        data.to_excel(writer, sheet_name=sheetName + "_result", index=False)


#######################################################################################################################

def printGraph(*groups_of_points, line=None):
    colors = ['yellow', 'red', 'green', 'black', 'pink', 'nocolorfinish']
    i = -1
    for points in groups_of_points:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        if i == -1:
            plt.scatter(x, y, alpha=0.3)
        else:
            plt.scatter(x, y, color=colors[i])
        i = i + 1
    plt.xlabel('X')
    plt.ylabel('Y')
    if line != None:
        slope = line['slope']
        intercept = line['intercept']

        # Generate x values for the line
        line_x = np.linspace(0, 1, 100)

        # Calculate y values for the line
        line_y = slope * line_x + intercept

        # Plot the line
        plt.plot(line_x, line_y, color='purple', linewidth=3, label='Least Squares line')
    plt.show()


#######################################################################################################################

SQUARESNAMES = ["square_10000_samples_V0", "square_10000_samples_V1", "square_10000_samples_V2","square_10000_samples_V3", "square_10000_samples_V4"]
NAMES = ["rhombus_10000_samples_V0","rhombus_10000_samples_V1", "rhombus_10000_samples_V2","rhombus_10000_samples_V3","rhombus_10000_samples_V4"]

for i in range(len(NAMES)):
    name1, name2 = SQUARESNAMES[i], NAMES[i]
    data1, data2 = readData(name1), readData(name2)

    output_w1 ,output_p1 = maximizeWeights(data1)
    print(f"name: {name1}, sum: {sum(output_w1)}, sum of squares: {sum(w**2 for w in output_w1)}")
    
    output_w2 ,output_p2 = maximizeWeights(data2)
    print(f"name: {name2}, sum: {sum(output_w2)}, sum of squares: {sum(w**2 for w in output_w2)}")
    # writeData(output_p,output_w,name, "data.xlsx")