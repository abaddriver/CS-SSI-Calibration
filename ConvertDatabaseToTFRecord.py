from os.path import join
from SSIImageHandler import SSIImageHandler
import tensorflow as tf
from DatasetCreator import DatasetCreator
from SystemDimensions import getSystemDimensions

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def ConvertDatabaseToTFRecord(inFolder, outFile, isTest=False):
    print('ConvertDatabaseToTFRecord:')
    print('input folder: ' + inFolder)
    print('output file: ' + outFile)
    print('is Test: ' + str(isTest))

    sysdims = getSystemDimensions()
    maxExamples = -1
    if isTest:
        maxExamples = 5

    dataCreator = DatasetCreator(directory=inFolder,
                                 NCube=sysdims.NCube,
                                 NDD=sysdims.NDD,
                                 maxNExamples=maxExamples)

    CubeFiles, DDFiles = dataCreator.getFileLists()

    imhand = SSIImageHandler()
    writer = tf.python_io.TFRecordWriter(outFile)

    for cubepath,ddpath in zip(CubeFiles, DDFiles):

        # read images:
        cubeim = imhand.readImage(cubepath)
        ddim = imhand.readImage(ddpath)
        cubeheight = cubeim.shape[0]
        cubewidth = cubeim.shape[1]
        cubechannels = cubeim.shape[2]
        ddheight = ddim.shape[0]
        ddwidth = ddim.shape[1]

        # convert images to string:
        cube_raw = cubeim.tostring()
        dd_raw = ddim.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'cubeheight': _int64_feature(cubeheight),
            'cubewidth': _int64_feature(cubewidth),
            'cubechannels': _int64_feature(cubechannels),
            'ddheight': _int64_feature(ddheight),
            'ddwidth': _int64_feature(ddwidth),
            'Cube': _bytes_feature(cube_raw),
            'DD': _bytes_feature(dd_raw)}))

        writer.write(example.SerializeToString())

    writer.close()

## paths for examples:
## define paths:
inFolder = ''
outFolder = ''
tfrecords_filename = join(outFolder, 'ImageDatabase.tfrecords')

# option for testing on 5 examples:
isTest=True

# run the process:
ConvertDatabaseToTFRecord(inFolder=inFolder, outFile=tfrecords_filename, isTest=isTest)

