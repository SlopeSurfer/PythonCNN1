#!/usr/bin/python3

import os
import struct
import numpy as np
from imagesAndLabels import imagesAndLabels

trainingData = imagesAndLabels('training','./data/')
imagesAndLabels = list(trainingData.getImagesAndLabels())
labels = list(trainingData.getLabels())
image = trainingData.getImage(5)
label = trainingData.getLabel(5)
print(label)
print (len(labels))
pixels = image
trainingData.displayImage(pixels)

