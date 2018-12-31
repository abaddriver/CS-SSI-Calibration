from DatasetCreator import DatasetCreator
from CalibEstimator import CalibEstimator
import SSIImageHandler as imhand
from datetime import datetime
import SystemSettings
import itertools
from os.path import join
from os import mkdir
import tensorflow as tf
import numpy as np


# debug or actual run?
debug_tests = 0

filtsize = 351
lossWeights = 'None'
loss_function = 'l2_loss'
regFactor = 0.001  # dummy, unused here
minConvCoeffs = 0

testfoldername = 'EstSensitivity_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# paths:
system = 'Server'
sysPaths = SystemSettings.getSystemPaths(system=system)
trainPath = sysPaths.trainPath
validPath = sysPaths.validPath
calibOutputBaseDir = join(sysPaths.outputBaseDir, testfoldername)
mkdir(calibOutputBaseDir)
OrigFilterPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/OpticalFilters.rawImage'

AllNoiseSigmas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
AllNoiseReceivers = [(True, True), (True, False), (False, True)]

if debug_tests == 1:
    maxNExamples = 10
    numEpochs = 2
    batchSize = 2
else:
    maxNExamples = -1
    numEpochs = 40
    batchSize = 100

# get Original Filter:
F_orig = np.squeeze(imhand.readImage(OrigFilterPath))
# remove one filter.. optical filter has 1 filter more
F_orig = F_orig[1:32, :]
NFilt_orig = F_orig.shape[1]

# calculate errors:
calib_diffs = []

for (noiseSigma, noiseReceivers) in itertools.product(AllNoiseSigmas, AllNoiseReceivers):

    addNoiseCube = noiseReceivers[0]
    addNoiseDD = noiseReceivers[1]

    # sizes:
    # set defined sizes:
    SystemSettings.setNFilt(filtsize)
    SystemSettings.setMinConvCoeffs(minConvCoeffs)
    sysDims = SystemSettings.getSystemDimensions()
    NCube = sysDims.NCube  # Cube [y, x, lambda] image size
    NDD = list(sysDims.NDD) # DD [y,x] image size
    NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
    DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300


    # paths:
    calibOutputDir = join(calibOutputBaseDir, 'noise_{}_nCube_{}_nDD_{}'.format(noiseSigma, addNoiseCube, addNoiseDD))
    mkdir(calibOutputDir)
    calibOutputPath = join(calibOutputDir, 'outputFilters.rawImage')

    # get train database
    myCreator = DatasetCreator(trainPath, NCube=NCube, NDD=NDD, maxNExamples=maxNExamples, FiltersSynthetic=F_orig)
    myCreator.cropDDWidth(DDx_new)
    train_database = myCreator.getDataset()


    # get validation database
    myCreator = DatasetCreator(validPath, NCube=NCube, NDD=NDD, maxNExamples=maxNExamples, FiltersSynthetic=F_orig)
    myCreator.cropDDWidth(DDx_new)
    valid_database = myCreator.getDataset()

    assert(train_database['Cubes'].shape[0] % batchSize == 0)
    assert(valid_database['Cubes'].shape[0] % batchSize == 0)

    # Add noise if needed:
    if addNoiseCube:
        train_database['Cubes'] = imhand.addGaussianNoise(train_database['Cubes'], noiseSigma)
    if addNoiseDD:
        train_database['DDs'] = imhand.addGaussianNoise(train_database['DDs'], noiseSigma)


    NDD[1] = DDx_new
    mylearningrate = [0.01]*100

    # estimate calibration:
    cEst = CalibEstimator(NX=NCube,
                          NY=NDD,
                          L=NCube[2],
                          NFilt=NFilt,
                          learningRate=mylearningrate,
                          batchSize=batchSize,
                          numEpochs=numEpochs,
                          logfiledir=calibOutputDir,
                          useLossWeights=lossWeights,
                          lossFunc=loss_function,
                          regularizationFactor=regFactor)


    cEst.createNPArrayDatasets()
    cEst.buildModel()
    cEst.train(DBtype='NPArray', DBargs={'Xtrain': train_database['Cubes'],'Ytrain': train_database['DDs'],
                                  'Xvalid': valid_database['Cubes'], 'Yvalid': valid_database['DDs']})

    # get calibration and save to file:
    calibRes = cEst.getCalibratedWeights()
    imhand.writeImage(calibRes, calibOutputPath)

    ind_begin = int(0.5*abs(NFilt_orig-NFilt))
    ind_end = ind_begin + NFilt
    calib_diffs.append(np.sum(np.square((calibRes - F_orig[:, ind_begin:ind_end]))))

    # make new estimation available:
    tf.reset_default_graph()
