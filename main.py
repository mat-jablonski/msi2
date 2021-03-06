import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import operator

SIMPLE_TEST_FILE = 'data.simple.test.10000.csv'
SIMPLE_TRAIN_FILE = 'data.simple.train.10000.csv'

THREE_GAUSS_TEST_FILE = 'data.three_gauss.test.10000.csv'
THREE_GAUSS_TRAIN_FILE = 'data.three_gauss.train.10000.csv'

K = 70
CLASSES = 2

def main():
    startAlgorithm(SIMPLE_TRAIN_FILE, SIMPLE_TEST_FILE, "Simple_data", K, 2)
    startAlgorithm(THREE_GAUSS_TRAIN_FILE, THREE_GAUSS_TEST_FILE, "Three_gauss", K, 3)

def startAlgorithm(trainDataFile, testDataFile, algorithmName, kPoints, classesCount):
    if os.path.isfile(trainDataFile) and os.path.isfile(testDataFile):
        loadAndCompute(trainDataFile, testDataFile, algorithmName, kPoints, classesCount)
    else:
        print("Files missing for {0}".format(algorithmName))
    
def loadAndCompute(trainDataFile, testDataFile, algorithmName, kPoints, classesCount):
    print("{0} computing...".format(algorithmName))
    train_data = readFile(trainDataFile)
    test_data = readFile(testDataFile)
    predictions, accuracy = compute(train_data, test_data, kPoints, classesCount)

    show_data(test_data, predictions, '{0}_results.png'.format(algorithmName), accuracy)

def readFile(fileName):
    xs = []
    with open(fileName, newline='') as file:
        lines = csv.reader(file)
        next(lines)
        for record in lines:
            parsed = [float(record[0]), float(record[1]), int(record[2])]
            xs.append(parsed)
    print("File {0} has been read successfully".format(fileName))
    return np.array(xs)

def compute(train_data, test_data, kPoints, classesCount):
    predictions=[]
    test_length = len(test_data)
    for x in range(test_length):
        testInstance = test_data[x]
        sortedPoints = getSortedPointsByDistances(train_data, testInstance)
        pointWithClass = getResponse(sortedPoints, testInstance, kPoints, classesCount)
        predictions.append(pointWithClass)
        
        if(x % 1000 == 0):
            print("{0} out of {1} points processed".format(x, test_length))
    
    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    return (predictions, accuracy)

def getSortedPointsByDistances(train_data, testInstance):
    xArray = train_data[0:len(train_data), 0]
    yArray = train_data[0:len(train_data), 1]
    xDiff = xArray - testInstance[0]
    yDiff = yArray - testInstance[1]
    xSquare = np.square(xDiff)
    ySquare = np.square(yDiff)
    sumArray = xSquare + ySquare
    sqrtArray = np.sqrt(sumArray)
    changedSqrt = np.array([sqrtArray])
    mergedArray = np.concatenate((train_data, changedSqrt.T), axis=1)
    sortedAscendingArray = mergedArray[mergedArray[:,-1].argsort(kind='mergesort')]
    return sortedAscendingArray

def getResponse(sortedPoints, testInstance, kPoints, classesCount):
    if kPoints>len(sortedPoints):
        kPoints = len(sortedPoints)
    selectedPoints = sortedPoints[0:kPoints,:]
    classesVotes = np.zeros((classesCount,1))
    for point in selectedPoints:
        index = int(point[2]) - 1
        classesVotes[index] = classesVotes[index] + 1

    return [testInstance[0], testInstance[1], np.argmax(classesVotes) + 1]
def getAccuracy(test_data, predictions):
	correct = 0
	for x in range(len(test_data)):
		if test_data[x][-1] == predictions[x][-1]:
			correct += 1
	return (correct/float(len(test_data))) * 100.0

def show_data(test_data, predictions, result_filename, accuracy):
    plt.gcf().set_size_inches(10, 10)
    plt.subplot(221)
    plt.title("Test Data")
    plt.scatter(list(map(lambda x: x[0], test_data)), list(map(lambda y: y[1], test_data)), c=list(map(lambda z: z[2], test_data)))
    plt.subplot(222)
    plt.title("Predictions, K={0}, accuracy: {1} %".format(K, accuracy))
    plt.scatter(list(map(lambda x: x[0], predictions)), list(map(lambda y: y[1], predictions)), c=list(map(lambda z: z[2], predictions)))
    plt.savefig(result_filename)
    plt.show()

if __name__ == "__main__":
    main()
