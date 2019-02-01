import numpy as np
from imagesAndLabels import imagesAndLabels
import random

class CNNStructure(object):
    def __init__(self, ftOrL, w = 0, b = 0):
        print("Type of input = ",type(ftOrL))
        
        if type(ftOrL) is str:
            print("Choosing constructor from file ", ftOrL)
            self.constructorFromFile(ftOrL)

        elif type(ftOrL) is list:
            print("Choosing constructor from list ", ftOrL," w = ",w,"b = ",b)
            self.constructorFromList(ftOrL,w,b)

        else:
            print("Input to CNNStructure constructor not recognized")
        
    def constructorFromFile(self,inFileType):
        print("Entering Constructor from file with file type **Not Implemented Yet ",inFileType)

    def constructorFromList(self,List,w,b):
        print("Entering constructor from list with list = ",List," with w and b = ",w,b)
        """
        Construct weighting matrices to take the input nodes and calculate the output nodes. The length of the list
        (minus one) gives the number of matrices. Considering consecutive pairs from the list, the first number
        determines the number of columns, the second number determines the number of rows. Going to construct 
        a list of numpy arrays. Note, each matrix gets one extra row of 0s ending in a 1.
        """
        numLayers = len(List)
        random.seed(0)
        self.weights = []
        print("numLayers ",numLayers)
        for layerCount in range(numLayers-1):
            numRows = List[layerCount+1]+1
            numCols = List[layerCount]
            tWeights = np.array([[0 for col in range(numCols)] for row in range(numRows)],dtype=float) 
            if(w != 0 or b !=0):    #w and b = zero is a special case to construct a zero filled matrix
                for rows in range(numRows-1):
                    for cols in range(numCols):
                        tWeights[rows,cols] = random.uniform(0,1)

            tWeights[numRows-1,numCols-1] = 1.
            self.weights.append(tWeights)
        print(self.weights)
        self.addLayerNodes(List)

# Make space for the layer nodes.
    def addLayerNodes(self,List):
        numLayers = len(List)
        self.layerNodes = []
        for layerCount in range(numLayers):
            numRows = List[layerCount]+1
            tempLayer = np.array([0 for col in range(numRows)],dtype=float)
            tempLayer[numRows-1] = 1.
            self.layerNodes.append(tempLayer)

        print(self.layerNodes)

        