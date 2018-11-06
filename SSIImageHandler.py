import numpy as np
import struct
import math
from matplotlib import pyplot as plt
from array import array


class SSIImageHandler:
    def __init__(self):
        print('SSIImageReader.__init__()')

    def readImage(self, imagePath):
        with open(imagePath, "rb") as fin:
            imsize = struct.unpack('iii', fin.read(3*4))
            imtype = str(fin.read(10*1))
            retIm = np.fromfile(fin, dtype=np.float32).reshape([imsize[0], imsize[1],imsize[2]])
        return retIm
    def writeImage(self, imArr, imagePath):
        imsize = [1,1,1]
        dims = imArr.shape
        for ii in range(min(len(dims), 3)):
            imsize[ii] = dims[ii]
        with open(imagePath, 'wb') as fout:
            # write image size to file:
            array('i', imsize).tofile(fout)
            # write 'single' in a 10 byte array length
            singleArr = array('b')
            singleArr.frombytes('single'.encode())
            singleArr.tofile(fout)
            numZeros = 10 - len('single')
            array('b',[0]*numZeros).tofile(fout)
            # write the image contents to file
            imArr.tofile(fout)





    def visualizeImageGray(self, Im, layer=-1):
        if (layer >= 0):
            Im2 = np.squeeze(Im[:,:,layer])
        else:
            Im2 = np.squeeze(Im);

        plt.imshow(Im2, cmap='gray', interpolation='bilinear')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()






