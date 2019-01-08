import numpy as np
import SystemSettings
from CalibEstimator import CalibEstimator
import SSIImageHandler as imhand
from DatasetCreator import DatasetCreator
from os.path import join


## sanity test 1: create a random filter and Cube, Generate DD and check the loss:
## this sanity test verifies that the model defined in CalibEstimator is correct
def calibEstimatorSanityTest1():
    # define sizes for tests:
    sysDims = SystemSettings.getSystemDimensions()
    NCube = sysDims.NCube  # Cube [y, x, lambda] image size
    NDD = list(sysDims.NDD)  # DD [y,x] image size
    NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
    DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300
    NChannels = NCube[2]
    NDD[1] = DDx_new # directly use DDx_new instead of the original size which is too big
    #  pad size:
    pad_size = int((DDx_new - NCube[1])/2)


    Cube = np.random.standard_normal(NCube)
    Filts = np.random.standard_normal((NChannels,NFilt))

    # add (NFilt-1) zeros to cube in beginning and end of x direction:
    Cube1 = np.lib.pad(Cube, ((0,0), (pad_size, pad_size), (0,0)), 'constant', constant_values=(0, 0))

    # create output as zeros:
    DD = np.zeros(NDD, dtype=np.float)

    # do convolution manualy:
    for ii in range(NCube[0]): # go over height of input image
        for kk in range(NChannels): # go over each color channel
            DD[ii, :] = DD[ii, :] + np.convolve(np.squeeze(Cube1[ii, :, kk]), np.flip(np.squeeze(Filts[kk, :])), 'same')

    # check the loss:
    calibEst = CalibEstimator(NX=NCube, NY=NDD, L=NChannels, NFilt=NFilt, learningRate=0.01, batchSize=32,
                              a0 = Filts)

    Cube = Cube.reshape([-1, 1, NCube[1], NCube[2]])
    DD = DD.reshape([-1, 1, NDD[1], 1])
    DD_eval = np.zeros_like(DD)

    calibEst.setModeEval()
    calibEst.createNPArrayDatasets()
    calibEst.buildModel()

    DD_eval = calibEst.eval(Xeval=Cube, Yeval=DD_eval)

    diff = np.squeeze(DD_eval) - np.squeeze(DD)
    maxAbsDiff = np.max(np.abs(diff))
    loss = np.sum(np.square(diff))
    print('l2 loss is {0:.4f}, max abs diff is {1:.4f}'.format(loss, maxAbsDiff))


# create random data (for sanity tests):
# create random Cubes and random filters, and then create DD images
# DD images are created using the model that was verified in sanity test 1
def calibEstimatorSanityTest2_createData():
    logfiledir = '/home/amiraz/Documents/CS SSI/TestFiles/SanityTest2/'
    validDir = join(logfiledir, 'Valid')
    trainDir = join(logfiledir, 'Train')

    # define sizes for tests:
    sysDims = SystemSettings.getSystemDimensions()
    NCube = sysDims.NCube  # Cube [y, x, lambda] image size
    NDD = list(sysDims.NDD)  # DD [y,x] image size
    NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
    DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300
    NChannels = NCube[2]
    NDD[1] = DDx_new # directly use DDx_new instead of the original size which is too big


    numTrainExamples = 1000
    numValidExamples = 200

    NCube_train = (numTrainExamples*NCube[0], 1, NCube[1], NCube[2])
    NCube_valid = (numValidExamples * NCube[0], 1, NCube[1], NCube[2])
    Cube_train = np.random.standard_normal(NCube_train).astype(dtype=np.float32)
    Cube_valid = np.random.standard_normal(NCube_valid).astype(dtype=np.float32)
    Filts_GT = np.random.standard_normal((NChannels,NFilt)).astype(dtype=np.float32)

    DD_train = np.zeros((NCube_train.shape[0], 1, NDD[1], 1), np.float32)
    DD_valid = np.zeros((NCube_valid.shape[0], 1, NDD[1], 1), np.float32)

    # create the DD (Y) image:
    cEst = CalibEstimator(NX=NCube, NY=NDD, L=NChannels, NFilt=NFilt, learningRate=0.01, batchSize=1, a0=Filts_GT)
    cEst.setModeEval()
    cEst.createNPArrayDatasets()
    cEst.buildModel()

    DD_train = cEst.eval(Xeval=Cube_train, Yeval=DD_train)
    DD_valid = cEst.eVal(Xeval=Cube_valid, Yeval=DD_valid)

    cEst.resetModel()

    # save results:
    # filters:
    filters_str = join(logfiledir, 'filters_GT.rawImage')
    imhand.writeImage(Filts_GT, filters_str)

    # save training data:
    for ii in range(numTrainExamples):
        cube_str = trainDir + 'Img_{}_Cube.rawImage'.format(ii)
        DD_str = trainDir + 'Img_{}_DD.rawImage'.format(ii)
        imhand.writeImage(np.squeeze(Cube_train[ii*256:(ii+1)*256, :, :, :]), cube_str)
        imhand.writeImage(np.squeeze(DD_train[ii * 256:(ii + 1) * 256, :, :, :]), DD_str)

    # save validation data:
    for ii in range(numValidExamples):
        cube_str = validDir + 'Img_{}_Cube.rawImage'.format(ii)
        DD_str = validDir + 'Img_{}_DD.rawImage'.format(ii)
        imhand.writeImage(np.squeeze(Cube_valid[ii*256:(ii+1)*256, :, :, :]), cube_str)
        imhand.writeImage(np.squeeze(DD_valid[ii * 256:(ii + 1) * 256, :, :, :]), DD_str)


## sanity test 2: train a network from a sinthesized random data and compare to the known filters
def calibEstimatorSanityTest2():
    logfiledir = '/home/amiraz/Documents/CS SSI/TestFiles/SanityTest2/'
    validDir = join(logfiledir, 'Valid')
    trainDir = join(logfiledir, 'Train')

    # define sizes for tests:
    sysDims = SystemSettings.getSystemDimensions()
    NCube = sysDims.NCube  # Cube [y, x, lambda] image size
    NDD = list(sysDims.NDD)  # DD [y,x] image size
    NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
    DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300
    NChannels = NCube[2]
    NDD[1] = DDx_new  # directly use DDx_new instead of the original size which is too big

    # get train database
    myCreator = DatasetCreator(trainDir, NCube=NCube, NDD=NDD, maxNExamples=-1)
    train_database = myCreator.getDataset()

    # get validation database
    myCreator = DatasetCreator(validDir, NCube=NCube, NDD=NDD, maxNExamples=-1)
    valid_database = myCreator.getDataset()

    Filts_GT = imhand.readImage(join(logfiledir, 'filters_GT.rawImage'))

    # run a training network and check the output weights
    # estimate calibration:
    cEst = CalibEstimator(NX=NCube,
                          NY=NDD,
                          L=NChannels,
                          NFilt=NFilt,
                          learningRate=0.01,
                          batchSize=100,
                          numEpochs=10,
                          logfiledir=logfiledir)

    cEst.buildModel()
    cEst.train(Xtrain=train_database['Cubes'], Ytrain=train_database['DDs'],
               Xvalid=valid_database['Cubes'], Yvalid=valid_database['DDs'])

    Filts_Calib = cEst.getCalibratedWeights()
    imhand.writeImage(Filts_Calib, logfiledir + 'sanity_test_2_Filters_Calib.rawImage')
    imhand.writeImage(Filts_GT, logfiledir + 'sanity_test_2_Filters_GT.rawImage')

    diff = np.squeeze(Filts_Calib) - np.squeeze(Filts_GT)
    maxAbsDiff = np.max(np.abs(diff))
    error = np.sum(np.square(diff))/diff.size
    print('error norm: {}, max abs error: {}'.format(error, maxAbsDiff))



#calibEstimatorSanityTest1()
#calibEstimatorSanityTest2()
#calibEstimatorSanityTest2_createData()