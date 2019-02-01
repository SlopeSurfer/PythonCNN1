#!/usr/bin/python3

import os
import struct
import numpy as np

class imagesAndLabels(object):
    """
    Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    which is GPL licensed.
    """
    def __init__(self, dataset = "training", path = "."):

        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.    
        """

        if dataset is "training":
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')

        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.lbl = np.fromfile(flbl, dtype=np.int8)  

#Construct the set of output nodes. These are lists of 11 elements, all zeros except for the one
#corresponding to the label (set to one) and the one at the end, which is set to one.                       
        tempNode = [0,0,0,0,0,0,0,0,0,0,1]
        self.outputNodes = []
        for x in self.lbl:
            thisIndex = self.lbl[x]
            tempNode[thisIndex] =1
            self.outputNodes.append(tempNode)
            tempNode[thisIndex] =0

        with open(fname_img, 'rb') as fimg:
            magic, num, self.rows, self.cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.lbl), self.rows*self.cols)    
                            
#Each layer of nodes gets one extra element which will be multiplied by the bias terms of the weighting matrix.

        tempCols = np.ones(([len(self.lbl),1]),dtype=np.uint8)
        img = np.append(img, tempCols)
        self.img = img.reshape(len(self.lbl), self.rows*self.cols+1)
        for x in range(len(self.lbl)):  #I do not know why this step is needed.
            self.img[x,784] = 1
        self.cols+=1

# Create an iterator that returns each image and label in turn
    def getImagesAndLabels(self):
        get_img = lambda idx: (self.lbl[idx], self.img[idx])
        for i in range(len(self.lbl)):
            yield get_img(i)

# Create an iterator that returns each label in turn
    def getLabels(self):
        for i in range(len(self.lbl)):
            yield self.lbl[i]

#Return a single image
    def getInputNodes(self,i):
        return self.img[i]

#Return a single label
    def getLabel(self,i):
        return self.lbl[i]

#Return the number of sets
    def getNumSets(self):
        return len(self.lbl)

#Return the input dimension
    def getInputDimension(self):
        return (len(self.img[0]))

#Return the output dimension
    def getOutputDimension(self):
        return (len(self.outputNodes[0])) 

#Return a set of output nodes.  

    def displayImage(self, imageNum):
        """
        Render a given numpy.uint8 2D array of pixel data. It is kept as a one dimensional array. So, reshape it here.
        """
        image = self.img[imageNum]
#knock the extra one off before you reshape it back into a 2-D array.
        tempCopy = image[:self.rows*(self.cols-1)].copy()
        tempCopy = np.reshape(tempCopy,(self.rows,self.cols-1))
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(tempCopy, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def asciiShow(self, imageNum):
        image = self.img[imageNum]
        #knock the extra one off before you reshape it back into a 2-D array.
        tempCopy = image[:self.rows*(self.cols-1)].copy()
        tempCopy = np.reshape(tempCopy,(self.rows,self.cols-1))
#        image.resize(self.rows,self.cols-1)
        for y in tempCopy:
            row = ""
            for x in y:
                row += '{0: <4}'.format(x)
            print(row)




