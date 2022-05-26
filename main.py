import network
from data import loadData

# [attributes, hidden neurons, output]
net = network.Network([13,10,5, 3])
trainData, testData = loadData()
# (training_data, epochs, batch_size, eta, test_data)
net.SGD(trainData, 10000, 10, 0.4, testData)
