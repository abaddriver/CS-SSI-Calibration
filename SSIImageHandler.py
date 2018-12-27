import numpy as np
import struct
from matplotlib import pyplot as plt
from array import array

def readImage(imagePath):
    with open(imagePath, "rb") as fin:
        imsize = struct.unpack('iii', fin.read(3*4))
        imtype = str(fin.read(10*1))
        retIm = np.fromfile(fin, dtype=np.float32).reshape([imsize[0], imsize[1],imsize[2]])
    return retIm

def writeImage(imArr, imagePath):
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

def filterImageLines(inIm, Filt):
    # filter image with line kernel
    # inputs:
    #   inIm -  input image, containing L color channels
    #   Filt -  filter, containing L filters that represent 1d filters across lines
    # outputs:
    #   outIm - output image, that is the filtering result of Filt on inIm.
    #           output dimensions: height(inIm) x (width(inIm) + width(Filt) - 1)

    # sizes:
    convSize = inIm.shape[1] + Filt.shape[1] - 1
    pad_size = int((convSize - inIm.shape[1])/2)
    L = Filt.shape[0]
    assert(inIm.shape[2] == L)
    Height = inIm.shape[0]

    # make padded input:
    inImPad = np.pad(inIm, [(0,0), (pad_size, pad_size), (0,0)], 'constant', constant_values=(0, 0))
    # allocate output:
    outIm = np.zeros([inIm.shape[0], convSize])

    # convolution manualy by channel:
    for ii in range(Height):  # go over height of input image
        for kk in range(L):  # go over each color channel
            outIm[ii, :] = outIm[ii, :] + \
                        np.convolve(np.squeeze(inImPad[ii, :, kk]), np.flip(np.squeeze(Filt[kk, :])), 'same')

    return outIm


def visualizeImageGray(Im, layer=-1):
    if layer >= 0:
        Im2 = np.squeeze(Im[:,:,layer])
    else:
        Im2 = np.squeeze(Im);

    plt.imshow(Im2, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

