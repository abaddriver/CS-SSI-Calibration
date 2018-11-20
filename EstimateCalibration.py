from DatasetCreator import DatasetCreator
from CalibEstimator import CalibEstimator
from SSIImageHandler import SSIImageHandler
import SSITFRecordHandler as recordhandler
from datetime import datetime
from SystemDimensions import getSystemDimensions
from os.path import join
import time

# debug or actual run?
debug_tests = 0

use_tfrecords = False

# paths:
trainPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/Train'
validPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/Valid'
calibOutputDir = '/home/amiraz/Documents/CS SSI/CalibOutputs/'
logfiledir = '/home/amiraz/Documents/CS SSI/CalibOutputs/'
calibOutputPath = calibOutputDir + 'Filters_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.rawImage'

if debug_tests == 1:
    maxNExamples = 10
    numEpochs = 2
    batchSize = 2
else:
    maxNExamples = -1
    numEpochs = 1000
    batchSize = 100

# sizes:
sysDims = getSystemDimensions()
NCube = sysDims.NCube  # Cube [y, x, lambda] image size
NDD = list(sysDims.NDD) # DD [y,x] image size
NFilt = sysDims.NFilt  # number of coefficients to be estimated for each lambda filter
DDx_new = sysDims.DDx_new  # the amount of Data influenced by a filter of size 300


if use_tfrecords:
    trainFilePath = recordhandler.ConvertDatabaseToTFRecords(trainPath, join(trainPath, 'tfrecords'), maxExamples=maxNExamples)
    validFilePath = recordhandler.ConvertDatabaseToTFRecords(validPath, join(validPath, 'tfrecords'), maxExamples=maxNExamples)
else:
    # get train database
    myCreator = DatasetCreator(trainPath, NCube=NCube, NDD=(256,2592),maxNExamples=maxNExamples)
    myCreator.cropDDWidth(DDx_new)
    train_database = myCreator.getDataset()

    # get validation database
    myCreator = DatasetCreator(validPath, NCube=NCube, NDD=(256,2592),maxNExamples=maxNExamples)
    myCreator.cropDDWidth(DDx_new)
    valid_database = myCreator.getDataset()

    assert(train_database['Cubes'].shape[0] % batchSize == 0)
    assert(valid_database['Cubes'].shape[0] % batchSize == 0)

NDD[1] = DDx_new

mylearningrate = [0.01]*100 + [0.001]*900

# estimate calibration:
cEst = CalibEstimator(NX=NCube,
                      NY=NDD,
                      L=NCube[2],
                      NFilt=NFilt,
                      learningRate=mylearningrate,
                      batchSize=batchSize,
                      numEpochs=numEpochs,
                      logfiledir=logfiledir)

if use_tfrecords:
    cEst.createTFRecordDatasets(trainFilenames=[trainFilePath], validFilenames=[validFilePath])
    cEst.buildModel()
    cEst.train()
else:
    cEst.createNPArrayDatasets()
    cEst.buildModel()
    cEst.train(DBtype='NPArray', DBargs={'Xtrain': train_database['Cubes'],'Ytrain': train_database['DDs'],
                                  'Xvalid': valid_database['Cubes'], 'Yvalid': valid_database['DDs']})

# get calibration and save to file:
calibRes = cEst.getCalibratedWeights()
imHand = SSIImageHandler()
imHand.writeImage(calibRes, calibOutputPath)