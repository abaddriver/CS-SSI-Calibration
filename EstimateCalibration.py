from DatasetCreator import DatasetCreator
from CalibEstimator import CalibEstimator
from SSIImageHandler import SSIImageHandler
from datetime import datetime
import time

# debug or actual run?
debug_tests = 0

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
NCube = [256, 256, 31]  # Cube [y, x, lambda] image size
NDD = [256, 2592]  # DD [y,x] image size
NFilt = 301  # number of coefficients to be estimated for each lambda filter
DDx_new = (NFilt-1)*2 + 256  # the amount of Data influenced by a filter of size 300

# get train database
myCreator = DatasetCreator(trainPath, NCube=(256,256,31), NDD=(256,2592),maxNExamples=maxNExamples)
myCreator.cropDDWidth(DDx_new)
train_database = myCreator.getDataset()
NDD[1] = DDx_new

# get validation database
myCreator = DatasetCreator(validPath, NCube=(256,256,31), NDD=(256,2592),maxNExamples=maxNExamples)
myCreator.cropDDWidth(DDx_new)
valid_database = myCreator.getDataset()
NDD[1] = DDx_new

assert(train_database['Cubes'].shape[0] % batchSize == 0)
assert(valid_database['Cubes'].shape[0] % batchSize == 0)

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
cEst.buildModel()
cEst.train(Xtrain=train_database['Cubes'],Ytrain=train_database['DDs'],
           Xvalid = valid_database['Cubes'], Yvalid = valid_database['DDs'])

# get calibration and save to file:
calibRes = cEst.getCaliratedWeights()
imHand = SSIImageHandler()
imHand.writeImage(calibRes, calibOutputPath)