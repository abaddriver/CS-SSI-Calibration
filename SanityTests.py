import numpy as np
import SystemDimensions
from CalibEstimator import CalibEstimator



## define sizes for tests:
sysDims = SystemDimensions.getSystemDimensions()

NCube = sysDims.NCube  # Cube [y, x, lambda] image size
NDD = sysDims.NDD  # DD [y,x] image size
NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300
NChannels = NCube[2]
NDD[1] = DDx_new # directly use DDx_new instead of the original size which is too big
# pad size:
pad_size = int((DDx_new - NCube[1])/2)

## sanity test 1: create a random filter and Cube, Generate DD and check the loss:
# create random Cube
Cube = np.random.standard_normal(NCube)
Filts = np.random.standard_normal((NChannels,NFilt))

# add (NFilt-1) zeros to cube in beginning and end of x direction:
Cube1 = np.lib.pad(Cube, ((0,0), (pad_size, pad_size), (0,0)), 'constant', constant_values=(0, 0))

# create output as zeros:
DD = np.zeros(NDD, dtype=np.float)

# do convolution manualy:
for ii in range(NCube[0]): # go over height of input image
    for kk in range(NChannels): # go over each color channel
        # cube_stripe =
        # filt_stripe =
        # val_res =
        #val_res = np.convolve(np.squeeze(Cube[ii, :, kk]), np.flip(np.squeeze(Filts[:, kk])), 'valid')
        DD[ii, :] = DD[ii, :] + np.convolve(np.squeeze(Cube1[ii, :, kk]), np.flip(np.squeeze(Filts[kk, :])), 'same')

# check the loss:
calibEst = CalibEstimator(NX=NCube, NY=NDD, L=NChannels, NFilt=NFilt, learningRate=0.01, batchSize=32,
                          a0 = Filts)

Cube = Cube.reshape([-1, 1, NCube[1], NCube[2]])
DD = DD.reshape([-1, 1, NDD[1], 1])

calibEst.buildModel()
endloss = calibEst.CalcLoss(Cube, DD)

print('end of test, loss is: {}'.format(endloss))