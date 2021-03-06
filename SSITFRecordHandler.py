from os.path import join, isfile, isdir
from os import mkdir
import SSIImageHandler as imhand
import tensorflow as tf
from DatasetCreator import DatasetCreator
from SystemSettings import getSystemDimensions

# utilities for ConvertDatabaseToTFRecords:
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# function ConvertDatabaseToTFRecords
# -----------------------------------
# this function converts a database of cube and dd image files of type .rawImage to a single .tfrecord file containing all examples
# inputs:
#   inFolder - path to folder in which there are pairs of cube and dd images.
#   outFolder - path to the output file that would be created.
#   maxExamples - maximum number of examples to store in the output files.
# outputs:
#   outFilePath - path to the output tfrecords file created
def ConvertDatabaseToTFRecords(inFolder, outFolder, maxExamples=-1):
    print('ConvertDatabaseToTFRecord:')
    print('input folder: ' + inFolder)
    print('output folder: ' + outFolder)
    print('maxExamples: ' + str(maxExamples))

    # get the cube and dd file lists:
    sysdims = getSystemDimensions()

    if not isdir(outFolder):
        mkdir(outFolder)

    dataCreator = DatasetCreator(directory=inFolder,
                                 NCube=sysdims.NCube,
                                 NDD=sysdims.NDD,
                                 maxNExamples=maxExamples)
    CubeFiles, DDFiles, Filenames = dataCreator.getFileLists()

    # crop dd image indices:
    x_dd_start = int((sysdims.NDD[1] - sysdims.DDx_new)/2)
    x_dd_end = x_dd_start + sysdims.DDx_new

    # initialize output
    outFiles = []

    outFilePath = join(outFolder, 'database_DDW{}.tfrecords'.format(sysdims.DDx_new))
    if isfile(outFilePath):
        # add file to file list and continue:
        return [outFilePath]

    writer = tf.python_io.TFRecordWriter(outFilePath)

    # iterate over all paths:
    for cubepath,ddpath, filename in zip(CubeFiles, DDFiles, Filenames):

        # read images:
        cubeim = imhand.readImage(cubepath)
        ddim = imhand.readImage(ddpath)[:,x_dd_start:x_dd_end, :]
        cubeheight = cubeim.shape[0]
        cubewidth = cubeim.shape[1]
        cubechannels = cubeim.shape[2]
        ddheight = ddim.shape[0]
        ddwidth = ddim.shape[1]

        for ii in range(cubeheight):
            # convert image stripes to string:
            cube_raw = cubeim[ii, :, :].tostring()
            dd_raw = ddim[ii, :, :].tostring()

            # create a feature:
            example = tf.train.Example(features=tf.train.Features(feature={
                'cubewidth': _int64_feature(cubewidth),
                'cubechannels': _int64_feature(cubechannels),
                'ddwidth': _int64_feature(ddwidth),
                'Cube': _bytes_feature(cube_raw),
                'DD': _bytes_feature(dd_raw)}))

            # write feature to file:
            writer.write(example.SerializeToString())

    # close file:
    writer.close()

    return [outFilePath]

# utilities for ConvertTFRecordsToDatset:
_sysdims = getSystemDimensions()

def _get_tfrecords_features():
    return {
        'cubewidth': tf.FixedLenFeature([1], tf.int64),
        'cubechannels': tf.FixedLenFeature([1], tf.int64),
        'ddwidth': tf.FixedLenFeature([1], tf.int64),
        'Cube': tf.FixedLenFeature([1], tf.string),
        'DD': tf.FixedLenFeature([1], tf.string)
        }

def _feature_retrieval(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features=_get_tfrecords_features()
    )
    # feature -> cube, dd
    _cubewidth = tf.cast(features['cubewidth'], tf.int64)[0]
    _cubechannels = tf.cast(features['cubechannels'], tf.int64)[0]
    _ddwidth = tf.cast(features['ddwidth'], tf.int64)[0]
    cube = tf.reshape(tf.decode_raw(tf.cast(features['Cube'], tf.string)[0], tf.float32),(1, _sysdims.NCube[1], _sysdims.NCube[2])) #_cubewidth, _cubechannels))
    dd = tf.reshape(tf.decode_raw(tf.cast(features['DD'], tf.string)[0], tf.float32),(1, _sysdims.DDx_new, 1))

    return cube, dd


# function ConvertTFRecordsToDatset
# _____________________________
# this function creates SSI datasets for train and validation from SSI tfrecord files and a string handle
# inputs:
#   handle - string handle
#   filesTrain -                list of tfrecord filenames for training
#   filesValid -                list of tfrecord filenames for validation
#   batchSize -                 batch size
#   trainShuffleBufferSize -    buffer size for shuffling training data.
# outputs:
#   next_elem -                 (cube, dd) tensors to build model from
#   train_init_iter -           initializer for the training dataset
#   valid_init_iter -           initializer for the validation dataset
def ConvertTFRecordsToDatset(handle, filesTrain, filesValid, batchSize, trainShuffleBuffSize=-1):

    # Training data
    train_dataset = tf.data.TFRecordDataset(filesTrain)
    train_dataset = train_dataset.map(_feature_retrieval)
    train_dataset = train_dataset.batch(batch_size=batchSize)
    # shuffle train dataset:
    if trainShuffleBuffSize>0:
        train_dataset = train_dataset.shuffle(trainShuffleBuffSize, reshuffle_each_iteration=True)

    # Validation data
    valid_dataset = tf.data.TFRecordDataset(filesValid)
    valid_dataset = valid_dataset.map(_feature_retrieval)
    valid_dataset = valid_dataset.batch(batch_size=batchSize)

    # Initializable iterator
    iterator = tf.data.Iterator.from_string_handle(
        handle,
        train_dataset.output_types,
        train_dataset.output_shapes)

    next_elem = iterator.get_next()

    train_init_iter = train_dataset.make_initializable_iterator()
    valid_init_iter = valid_dataset.make_initializable_iterator()

    return next_elem, train_init_iter , valid_init_iter
