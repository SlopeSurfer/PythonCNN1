import numpy as np
from imagesAndLabels import imagesAndLabels
import random

class CNNStructure(object):
    def __init__(self, ftOrL, w = 0, b = 0):
        print("Type of input = ",type(ftOrL))
        self.layerNodes = [] 
        self.weights = []
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
        a list of numpy arrays. Note, each matrix gets a row of 0s ending in a 1 for the last row.
        """
        numLayers = len(List)
        random.seed(0)

        for layerCount in range(numLayers-1):
            numRows = List[layerCount+1]
            numCols = List[layerCount]
            tWeights = np.array([[0 for col in range(numCols)] for row in range(numRows)],dtype=float) 
            if(w != 0 or b !=0):    #w and b = zero is a special case to construct a zero filled matrix
                for rows in range(numRows-1):
                    for cols in range(numCols):
                        tWeights[rows,cols] = random.uniform(0,1)

            tWeights[numRows-1,numCols-1] = 1.
            self.weights.append(tWeights)

        self.addLayerNodes(List)

# Make space for the layer nodes.
    def addLayerNodes(self,List):
        numLayers = len(List)

        for layerCount in range(numLayers):
            numRows = List[layerCount]
            tempLayer = np.array([0 for col in range(numRows)],dtype=float).reshape(numRows,1)
            tempLayer[numRows-1] = 1.
            self.layerNodes.append(tempLayer)
        print("Show all the layerNodes")
        print(self.layerNodes)

    def updateLayerNodes(self,inputLayer):
        self.layerNodes[0] = inputLayer
        print("len(self.weights) ",len(self.weights))
        for layerCount in range (len(self.weights)):
            if layerCount is 0:
                tempLayer = inputLayer
            
            print("The matrix")
            print(self.weights[layerCount])
            print("The vector")
            print(tempLayer)
            
            self.layerNodes[layerCount+1] = np.dot(self.weights[layerCount],tempLayer)
            print("After matrix mult")
            print(self.layerNodes[layerCount+1])
#Apply sigma
           
            for rowCount in range(len(self.layerNodes[layerCount+1])):
                if self.layerNodes[layerCount+1][rowCount]< 0:
                    self.layerNodes[layerCount+1][rowCount] = 0
            
            tempLayer = self.layerNodes[layerCount+1]
 #           print(tempLayer)

    def makeGradPass(self,tempGradStruct,desired):
        print("len(self.weights)",len(self.weights))
        for layerCount in range(len(self.weights),0,-1):
            print("layerCount ",layerCount)
            partRelu = np.dot(self.weights[layerCount-1],self.layerNodes[layerCount-1])
            if layerCount == len(self.weights):
                pCpA = 2.*(self.layerNodes[layerCount] - desired)
                print("pCpA")
                print(pCpA)

            for rowCount in range(len(self.weights[layerCount - 1])-1):

                if partRelu[rowCount] < 0.: 
	                partRelu[rowCount] = 0.
                
                else: 
	                partRelu[rowCount] = 1.
                
                print("len(self.weights[layerCount-1][0]) ",len(self.weights[layerCount-1][0]))
                for colCount in range(len(self.weights[layerCount-1][0])-1):
                    print("colCount",colCount)
                    tempGradStruct.weights[layerCount - 1][rowCount][colCount] =self.layerNodes[layerCount - 1][colCount] * partRelu[rowCount] *pCpA[rowCount];
			
#Each row also has a bias term at the end of the row.
                tempGradStruct.weights[layerCount - 1][rowCount][len(self.weights[layerCount - 1][0])-1] =partRelu[rowCount] * pCpA[rowCount];

            temppCpA = np.array
            temppCpA = np.zeros(([len(self.weights[layerCount - 1][0])-1,1]),dtype=np.float)
    #		//Calculate the pCpA vector for the next round.
            for colCount in range(len(self.weights[layerCount - 1][0])-1): 
	                tempSum = 0.;
	                for rowCount in range(len(self.weights[layerCount - 1]) - 1):
		                tempSum += self.weights[layerCount - 1][rowCount][colCount]*partRelu[rowCount]*pCpA[rowCount];
				
	                temppCpA[colCount] =tempSum
            pCpA =temppCpA;

    def calcCost(self,input,desired):
        self.updateLayerNodes(input)
#The cost only depends on the last layer's nodes. And there is no addition term added to the vector.
        costSum = 0.
        numLayers = len(self.layerNodes)
        for iCnt in range(len(desired)-1): 	# Cut down by one because desired has the extra 1.
										    #It doesn't really matte r since both have 1 at the end.
            costSum += (self.layerNodes[numLayers-1][iCnt] - desired[iCnt])**2
	
        return(costSum);