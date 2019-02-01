#!/usr/bin/python3

import os
import struct
import numpy as np
from imagesAndLabels import imagesAndLabels
from CNNStructure import CNNStructure

trainingData = imagesAndLabels('training','./data/')
#imagesAndLabels = list(trainingData.getImagesAndLabels())
#labels = list(trainingData.getLabels())
node6 = trainingData.getInputNodes(3)
label6 = trainingData.getLabel(3)

print("nodes size ",node6.size,"node[784] ",node6[784])
print("label ",label6)
print("Number of sets ",trainingData.getNumSets())
print("Input dimension = ",trainingData.getInputDimension(),"Output dimension = ",trainingData.getOutputDimension())


print("")
print("")
print("Set up weights structure")
testStructure = CNNStructure('training')
#tempList  = [trainingData.getInputDimension(),16,16,trainingData.getOutputDimension()]
tempList = [5,4,3,2]
testStructure2 = CNNStructure(tempList,.5,.5)
#trainingData.displayImage(3)
#trainingData.asciiShow(3)

