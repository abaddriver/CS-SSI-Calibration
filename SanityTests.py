import numpy as np
import SystemSettings
from CalibEstimator import CalibEstimator
import SSIImageHandler as imhand
from DatasetCreator import DatasetCreator
from os.path import join
from os import mkdir


_LOG_FILE_DIR = '/home/amiraz/Documents/CS SSI/TestFiles/SanityTest2/'
_FILTERS_GT_PATH = '/home/amiraz/Documents/CS SSI/ImageDatabase/OpticalFilters.rawImage'

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
    Filts = np.random.standard_normal((NChannels, NFilt))

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
    logfiledir = _LOG_FILE_DIR
    validDir = join(logfiledir, 'Valid')
    trainDir = join(logfiledir, 'Train')

    # define sizes for tests:
    sysPaths = SystemSettings.getSystemPaths('Server')
    sysDims = SystemSettings.getSystemDimensions()
    NCube = sysDims.NCube  # Cube [y, x, lambda] image size
    NDD = list(sysDims.NDD)  # DD [y,x] image size
    NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
    DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300
    NChannels = NCube[2]

    numTrainExamples = 1000
    numValidExamples = 200

    NCube_train = (numTrainExamples*NCube[0], 1, NCube[1], NCube[2])
    NCube_valid = (numValidExamples * NCube[0], 1, NCube[1], NCube[2])

    # Cube_train = np.random.standard_normal(NCube_train).astype(dtype=np.float32)
    # Cube_valid = np.random.standard_normal(NCube_valid).astype(dtype=np.float32)

    dataCreator = DatasetCreator(directory=sysPaths.trainPath, NCube=NCube, NDD=NDD)
    dataCreator.cropDDWidth(DDx_crop=DDx_new)
    train_dataset = dataCreator.getDataset()
    Cube_train = train_dataset['Cubes']
    # Cube_std = np.std(Cube_train)
    #Cube_train = Cube_train / Cube_std
    del train_dataset

    # print('calibEstimatorSanityTest2_createData: Cube: std: {}, mean: {}, min: {}, max: {}'.format(
    #     np.std(Cube_train), np.mean(Cube_train), np.min(Cube_train), np.max(Cube_train)
    # ))

    dataCreator = DatasetCreator(directory=sysPaths.validPath, NCube=NCube, NDD=NDD)
    dataCreator.cropDDWidth(DDx_crop=DDx_new)
    valid_dataset = dataCreator.getDataset()
    Cube_valid = valid_dataset['Cubes']
    # Cube_valid = Cube_valid / Cube_std
    del valid_dataset

    Filts_GT = np.squeeze(imhand.readImage(_FILTERS_GT_PATH))
    # crop Filts_GT to the shape of NFilt
    crop_remove_size = int((Filts_GT.shape[1] - NFilt)/2)
    Filts_GT = Filts_GT[1:32, crop_remove_size:crop_remove_size+NFilt]
    # Filts_GT = np.random.normal(loc=0.0, scale=1.0, size=(31, 301)).astype(dtype=np.float32)

    print('calibEstimatorSanityTest2_createData: Filters size: ({}x{})'.format(Filts_GT.shape[0], Filts_GT.shape[1]))

    NDD[1] = DDx_new  # directly use DDx_new instead of the original size which is too big

    DD_train = np.zeros((NCube_train[0], 1, NDD[1], 1), np.float32)
    DD_valid = np.zeros((NCube_valid[0], 1, NDD[1], 1), np.float32)

    # create the DD (Y) image:
    cEst = CalibEstimator(NX=NCube, NY=NDD, L=NChannels, NFilt=NFilt, learningRate=0.01, batchSize=128, a0=Filts_GT)
    cEst.setModeEval()
    cEst.createNPArrayDatasets()
    cEst.buildModel()

    DD_train = cEst.eval(Xeval=Cube_train, Yeval=DD_train)
    DD_valid = cEst.eval(Xeval=Cube_valid, Yeval=DD_valid)

    cEst.resetModel()

    # save results:
    # filters:
    filters_str = join(logfiledir, 'filters_GT.rawImage')
    imhand.writeImage(Filts_GT, filters_str)

    # save training data:
    for ii in range(numTrainExamples):
        cube_str = join(trainDir, 'Img_{}_Cube.rawImage'.format(ii))
        DD_str = join(trainDir, 'Img_{}_DD.rawImage'.format(ii))
        imhand.writeImage(np.squeeze(Cube_train[ii*256:(ii+1)*256, :, :, :]), cube_str)
        imhand.writeImage(np.squeeze(DD_train[ii * 256:(ii + 1) * 256, :]), DD_str)

    # save validation data:
    for ii in range(numValidExamples):
        cube_str = join(validDir, 'Img_{}_Cube.rawImage'.format(ii))
        DD_str = join(validDir, 'Img_{}_DD.rawImage'.format(ii))
        imhand.writeImage(np.squeeze(Cube_valid[ii*256:(ii+1)*256, :, :, :]), cube_str)
        imhand.writeImage(np.squeeze(DD_valid[ii * 256:(ii + 1) * 256, :]), DD_str)

# sanity test 2: train a network from a sinthesized random data and compare to the known filters
def calibEstimatorSanityTest2(subfold=None):
    logfiledir = _LOG_FILE_DIR
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

    train_dict = {'Xtrain': train_database['Cubes'], 'Ytrain': train_database['DDs'],
                  'Xvalid': valid_database['Cubes'], 'Yvalid': valid_database['DDs']}

    Cube_train = train_dict['Xtrain']
    print('calibEstimatorSanityTest2_createData: Cube: std: {}, mean: {}, min: {}, max: {}'.format(
        np.std(Cube_train), np.mean(Cube_train), np.min(Cube_train), np.max(Cube_train)
    ))

    if subfold is None:
        outFold = logfiledir
    else:
        outFold = join(logfiledir, subfold)
        mkdir(outFold)

    # run a training network and check the output weights
    # estimate calibration:
    cEst = CalibEstimator(NX=NCube,
                          NY=NDD,
                          L=NChannels,
                          NFilt=NFilt,
                          learningRate=0.01,
                          batchSize=100,
                          numEpochs=10,
                          logfiledir=outFold,
                          optimizer='gd')
    cEst.createNPArrayDatasets()
    cEst.buildModel()
    cEst.train(DBtype='NPArray', DBargs=train_dict)
    Filts_Calib = cEst.getCalibratedWeights()
    imhand.writeImage(Filts_Calib, join(outFold, 'Filters_Calib.rawImage'))


    diff = np.squeeze(Filts_Calib) - np.squeeze(Filts_GT)
    maxAbsDiff = np.max(np.abs(diff))
    error = np.sum(np.square(diff))/diff.size
    print('error norm: {}, max abs error: {}'.format(error, maxAbsDiff))
    cEst.resetModel()

#calibEstimatorSanityTest1()
calibEstimatorSanityTest2_createData()
#calibEstimatorSanityTest2(subfold='test1')
# calibEstimatorSanityTest2(subfold='test2')
