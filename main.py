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

K = 3

def load_and_compute():
    print("hello")

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


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(train_data, testInstance):
	distances = []
	length = len(testInstance)-1
	for x in range(len(train_data)):
		dist = euclideanDistance(testInstance, train_data[x], length)
		distances.append((train_data[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(K):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	max_class = max(classVotes, key=classVotes.get)
	return max_class
 
def getAccuracy(test_data, predictions):
	correct = 0
	for x in range(len(test_data)):
		if test_data[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(test_data))) * 100.0

def split_in_out(data):
    return data[:, 0:2], data[:, 2:3].astype('int')

def simple_data_load_and_compute():
    print("simple data computing...")
    train_data = readFile(SIMPLE_TRAIN_FILE)
    test_data = readFile(SIMPLE_TEST_FILE)
    predictions = compute(train_data, test_data)

    test_points, test_classess= split_in_out(test_data)
    
    show_data(test_points, test_classess, predictions)
   
def three_gauss_data_load_and_compute():
    print("three gauss data computing...")
    train_data = readFile(THREE_GAUSS_TRAIN_FILE)
    test_data = readFile(THREE_GAUSS_TEST_FILE)
    predictions = compute(train_data, test_data)

    test_points, test_classess= split_in_out(test_data)

    show_data(test_points, test_classess, predictions)

def compute(train_data, test_data):
    predictions=[]
    test_length = len(test_data)
    for x in range(test_length):
        neighbors = getNeighbors(train_data, test_data[x])
        result = getResponse(neighbors)
        predictions.append(result)
        
        if(x % 100 == 0):
            print("{0} out of {1} points processed".format(x, test_length))
    
    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    return predictions


def show_data(xs, vals, predictions):
    def map_c(v):
        if v == 1:
            return [1, 0, 0]
        elif v == 2:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

  
    plt.gcf().set_size_inches(30, 30)
    plt.subplot(221)
    plt.title("Test Data")
    plt.scatter(list(map(lambda x: x[0], xs)), list(map(lambda y: y[1], xs)), c=list(map(map_c, vals)))
    plt.subplot(222)
    plt.title("Predictions")
    plt.scatter(list(map(lambda x: x[0], xs)), list(map(lambda y: y[1], xs)), c=list(map(map_c, predictions )))
    plt.show()


if os.path.isfile(SIMPLE_TEST_FILE) and os.path.isfile(SIMPLE_TRAIN_FILE):
    simple_data_load_and_compute()
else:
    print("Files missing for simple data")

if os.path.isfile(THREE_GAUSS_TEST_FILE) and os.path.isfile(THREE_GAUSS_TRAIN_FILE):
    three_gauss_data_load_and_compute()
else:
    print("Files missing for three gauss data")
