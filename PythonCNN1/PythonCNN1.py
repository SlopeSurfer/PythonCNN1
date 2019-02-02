#!/usr/bin/python3

import os
import struct
import numpy as np
from imagesAndLabels import imagesAndLabels
from CNNStructure import CNNStructure

#trainingData = imagesAndLabels('training','./data/')
#imagesAndLabels = list(trainingData.getImagesAndLabels())
#labels = list(trainingData.getLabels())
#node6 = trainingData.getInputNodes(3)
#label6 = trainingData.getLabel(3)

#print("nodes size ",node6.size,"node[784] ",node6[784])
#print("label ",label6)
#print("Number of sets ",trainingData.getNumSets())
#print("Input dimension = ",trainingData.getInputDimension(),"Output dimension = ",trainingData.getOutputDimension())

print("")
print("Set up weights structure")
#testStructure = CNNStructure('training')
#tempList  = [trainingData.getInputDimension(),16,16,trainingData.getOutputDimension()]
tempList = [5,3,3,2]
tempInput = np.array([0 for col in range(5)],dtype=float).reshape(5,1)
tempDesired = np.array([0 for col in range(2)],dtype=float).reshape(2,1)
tempDesired[0] = 1.;
testStructure2 = CNNStructure(tempList,.5,.5)
for x in range(5):
    tempInput[x] = x
tempInput[4] = 1.
print("tempInput")
print(tempInput)
#tempInput = np.array[1.,2.,3.,4.,5.]

testStructure2.updateLayerNodes(tempInput)

tempGradHoldStructure = CNNStructure(tempList)
print("tempGradHoldStructure",tempGradHoldStructure.weights[0], tempGradHoldStructure.weights[1])
testStructure2.makeGradPass(tempGradHoldStructure,tempDesired)
#trainingData.displayImage(3)
#trainingData.asciiShow(3)
for layer in range(len(tempList)-1):
    print("Layer = ",layer)
    print(testStructure2.weights[layer])

print("Cost = ",testStructure2.calcCost(tempInput,tempDesired))

