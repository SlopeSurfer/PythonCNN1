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
        print("Initializing imagesAndLabels")
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
 
        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.img = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.lbl), rows, cols)

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
    def getImage(self,i):
        return self.img[i]

#Return a single label
    def getLabel(self,i):
        return self.lbl[i]

    def displayImage(self, image):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def ascii_show(self, image):
        for y in image:
            row = ""
            for x in y:
                row += '{0: <4}'.format(x)
            print(row)




