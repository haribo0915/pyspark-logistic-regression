from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[-1], values[:-1])    

sc = SparkContext.getOrCreate();

# Load and parse the data
data = sc.textFile("s3://cloud-computing-spark/data_banknote_authentication.txt")
parsedData = data.map(parsePoint)

# Train model
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))