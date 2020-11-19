import numpy as np
import sys
from pyspark import SparkConf, SparkContext

class CustomLabeledPoint:
    def __init__(self, x, y):
        self.x = np.array(x, float)
        self.y = y

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def sign(x):
    if x < 0.52:
        return 0
    else:
        return 1

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    values.insert(0, 1)
    return CustomLabeledPoint(values[:-1], values[-1]) 

def initialize_parameter():
    seed = 100
    mu = 0.0
    sigma = 0.01
    random_obj = np.random.RandomState(seed)
    return random_obj.normal(loc=mu, scale=sigma, size=5)


sc = SparkContext.getOrCreate(); 

# Load and parse the data
data = sc.textFile("s3://cloud-computing-spark/data_banknote_authentication.txt")
parsedData = data.map(parsePoint)

#initialize
w = initialize_parameter()
epoch = 500

#logistic regression
for i in range(epoch): 
    grad = parsedData.map(lambda p: p.x * (sigmoid(np.dot(p.x, w))-p.y)).reduce(lambda x, y: x+y)
    w -= grad

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.y, sign(sigmoid(np.dot(p.x, w)))))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))