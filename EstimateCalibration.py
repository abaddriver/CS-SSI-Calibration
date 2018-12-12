from DatasetCreator import DatasetCreator
from CalibEstimator import CalibEstimator
from SSIImageHandler import SSIImageHandler
import SSITFRecordHandler as recordhandler
from datetime import datetime
import SystemSettings
import itertools
from os.path import join
from os import mkdir
import tensorflow as tf

# debug or actual run?
debug_tests = 0
# use tfrecords - tests show it doesnt speed up
use_tfrecords = False

filterSizes = list(range(21, 300, 50)) + [301, 351]
#filterSizes = [301]
lossWeights = ['None', 'proportional', 'squared', 'quad', 'exp']  # {None, 'None', 'proportional', 'squared', 'quad', 'exp'}
# lossWeights = ['exp']
allminConvCoeffs = [0]
loss_functions = ['l1_loss']

testfoldername = 'Est_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# paths:
system = 'Server'
sysPaths = SystemSettings.getSystemPaths(system=system)
trainPath = sysPaths.trainPath
validPath = sysPaths.validPath
calibOutputBaseDir = join(sysPaths.outputBaseDir, testfoldername)
mkdir(calibOutputBaseDir)

for (filtsize, lossWeights, minConvCoeffs, loss_function) in itertools.product(filterSizes, lossWeights, allminConvCoeffs, loss_functions):

    if debug_tests == 1:
        maxNExamples = 10
        numEpochs = 2
        batchSize = 2
    else:
        maxNExamples = -1
        numEpochs = 100
        batchSize = 200

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
    calibOutputDir = join(calibOutputBaseDir, 'NFilt_{}_weights_{}_DDW_{}_{}'.format(filtsize, lossWeights, DDx_new, loss_function))
    mkdir(calibOutputDir)
    calibOutputPath = join(calibOutputDir, 'outputFilters.rawImage')


    if use_tfrecords:
        trainFilePaths = recordhandler.ConvertDatabaseToTFRecords(trainPath, join(trainPath, 'tfrecords'), maxExamples=maxNExamples)
        validFilePaths = recordhandler.ConvertDatabaseToTFRecords(validPath, join(validPath, 'tfrecords'), maxExamples=maxNExamples)
    else:
        # get train database
        myCreator = DatasetCreator(trainPath, NCube=NCube, NDD=NDD,maxNExamples=maxNExamples)
        myCreator.cropDDWidth(DDx_new)
        train_database = myCreator.getDataset()

        # get validation database
        myCreator = DatasetCreator(validPath, NCube=NCube, NDD=NDD,maxNExamples=maxNExamples)
        myCreator.cropDDWidth(DDx_new)
        valid_database = myCreator.getDataset()

        assert(train_database['Cubes'].shape[0] % batchSize == 0)
        assert(valid_database['Cubes'].shape[0] % batchSize == 0)

    NDD[1] = DDx_new
    mylearningrate = [0.01]*100 + [0.001]*100

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
                          lossFunc=loss_function)

    if use_tfrecords:
        cEst.createTFRecordDatasets(trainFilenames=trainFilePaths, validFilenames=validFilePaths)
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

    # make new estimation available:
    tf.reset_default_graph()
