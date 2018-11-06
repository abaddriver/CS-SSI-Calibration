import numpy as np
import tensorflow as tf
import math
import logging
from datetime import datetime
import time
from SSIImageHandler import SSIImageHandler


# Class CalibEstimator
# -------------
# describes an estimator for calibration of the A matrix as filters
# there are L filters on each spatial cube level
# each level fits a different wavelength image
# each image from X is being filtered and the sum is Y
class CalibEstimator:
    def __init__(self,
                 NX,  # dimensions of cube image (X)
                 NY,  # dimensions of DD image (Y)
                 L,  # number of wavelengths
                 NFilt,  # number of coeffs in each wavelength's filter
                 learningRate,  # learning rate for backprop algorithm
                 batchSize, # number of examples in each batch
                 numEpochs = 1, # number of times to go over the dataset
                 logfiledir = '',  # filename to print log to
                 a0=0):  # initial guess for the filters. default: empty
        # init function for estimator
        # inputs:
        #   NX - x,y dimensions of X image (spatial cube, with dimensions y and z joined)
        #   NY - x,y dimensions of Y image (diffused and dispersed)
        #   filterSize - the number of elements in each filter
        #   a0 - matrix of L*filterSize, specifying the fist guess for the matrix A (default: 0 - no guess)

        print('CalibEstimator.__init__()')

        # dimensions:
        self.dims = {'NX': NX, 'NY': NY, 'L': L, 'NFilt': NFilt}
        self.dims['NA'] = [self.dims['NY'][0], self.dims['NX'][0]]
        # other inputs:
        self.learningRate = learningRate
        self.a0 = a0
        self.batchSize = batchSize
        self.updatedWeights = a0
        self.numEpochs = numEpochs
        self.logfiledir = logfiledir

        if self.logfiledir == '':
            self.logfiledir = './'

        if not isinstance(self.learningRate, list):
            self.learningRate = [self.learningRate]*self.numEpochs

        assert(len(self.learningRate) >= self.numEpochs)

    def buildModel(self):
        # this function creates the graph: tensors and operations
        print('CalibEstimator.initModel()')

        # initializer for filters matrix
        if (self.a0==0):
            kernelInit=tf.initializers.random_normal
        else:
            # transpose and reshape a0 matrix so that dimensions fit tensorflow
            self.a0_ndarr=np.array(self.a0).T.reshape((1, self.dims['NFilt'], 1, self.dims['L']))
            kernelInit=tf.constant_initializer(self.a0_ndarr)

        # find the padded value to match the output width:
        paddW = int((self.dims['NY'][1] - self.dims['NX'][1])/2)

        self.tensors = {}
        # input:
        self.tensors['x_data'] = tf.placeholder(tf.float32, shape=[None, 1, self.dims['NX'][1], self.dims['NX'][2]], name='x')
        # output:
        self.tensors['y_data_GT'] = tf.placeholder(tf.float32, shape=[None,1, self.dims['NY'][1],1], name='y_GT')
        # dataset:
        self.tensors['train_dataset'] = tf.data.Dataset.from_tensor_slices((self.tensors['x_data'], self.tensors['y_data_GT'])).\
            repeat(self.numEpochs).batch(self.batchSize)
        self.tensors['valid_dataset'] = tf.data.Dataset.from_tensor_slices((self.tensors['x_data'], self.tensors['y_data_GT'])).\
            batch(self.batchSize)
        self.tensors['iter'] = tf.data.Iterator.from_structure(self.tensors['train_dataset'].output_types,
                                                               self.tensors['train_dataset'].output_shapes)
        self.tensors['x'], self.tensors['y_GT'] = self.tensors['iter'].get_next()
        self.tensors['train_init_op'] = self.tensors['iter'].make_initializer(self.tensors['train_dataset'])
        self.tensors['valid_init_op'] = self.tensors['iter'].make_initializer(self.tensors['valid_dataset'])

        # network:
        self.tensors['x_padded'] = tf.pad(self.tensors['x'], [[0, 0], [0, 0], [paddW, paddW], [0, 0]], "CONSTANT")
        # convolutional layer:
        self.tensors['y_est'] = tf.layers.conv2d(inputs=self.tensors['x_padded'],
                                                      filters=1,
                                                      kernel_size=(1, self.dims['NFilt']),
                                                      kernel_initializer=kernelInit,
                                                      padding = 'same',
                                                      activation=None,
                                                      use_bias=False,
                                                      name='x_filtered')

        #loss function:
        self.tensors['loss']=tf.losses.mean_squared_error(labels=self.tensors['y_GT'], predictions=self.tensors['y_est'])

    def train(self, Xtrain, Ytrain, Xvalid, Yvalid):
        print('CalibEstimator.train()')
        self.tensors['learningRate'] = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.GradientDescentOptimizer(self.tensors['learningRate'])
        trainOp = optimizer.minimize(self.tensors['loss'])

        logfilename = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        logfilename = self.logfiledir + logfilename + '.log'

        with tf.Session() as sess:

            # start log with basic info:
            logging.basicConfig(filename=logfilename, level=logging.DEBUG)
            logging.info("starting sessions:")
            logging.info("training size: {0}".format(Xtrain.shape[0]))
            logging.info("Validation size: {0}".format(Xvalid.shape[0]))
            logging.info("filter length: {0}, numFilters: {1}".format(self.dims['NFilt'], self.dims['L']))
            logging.info("epochs: {0}".format(self.numEpochs))
            # init gloval variables and init
            sess.run(tf.global_variables_initializer())

            numBatchesTrain = int(math.floor(Xtrain.shape[0] / self.batchSize))
            numBatchesValid = int(math.floor(Xvalid.shape[0] / self.batchSize))

            # loop over epochs and examples:
            for epoch in range(self.numEpochs):

                train_loss = 0.0
                valid_loss = 0.0
                start = time.time()

                # init train database:
                sess.run(self.tensors['train_init_op'], feed_dict={self.tensors['x_data']: Xtrain,
                                                                   self.tensors['y_data_GT']: Ytrain})

                # run all train examples in batch:
                for _ in range(numBatchesTrain):
                    _, loss_val = sess.run([trainOp, self.tensors['loss']], feed_dict={self.tensors['learningRate']: self.learningRate[epoch]})
                    train_loss += loss_val

                # init validation database
                sess.run(self.tensors['valid_init_op'], feed_dict={self.tensors['x_data']: Xvalid,
                                                                   self.tensors['y_data_GT']: Yvalid})

                # run all valid examples in batch:
                for _ in range(numBatchesValid):
                    loss_val = sess.run(self.tensors['loss'])
                    valid_loss += loss_val

                # print loss:

                dur = time.time() - start
                printstr = "Iter: {0}, time: {1:.4f}, TrainLoss: {2:.4f}, ValidLoss: {3:.4f}".format(epoch,
                                                                                                     dur,
                                                                                                     train_loss / numBatchesTrain,
                                                                                                     valid_loss / numBatchesValid)
                print(printstr)
                logging.info(printstr)

                if (epoch % 20) == 0:
                    # get calibration and save to file:
                    weights_var = tf.trainable_variables()[0]
                    self.updatedWeights = np.squeeze(sess.run(weights_var)).T
                    imHand = SSIImageHandler()
                    imHand.writeImage(self.updatedWeights, self.logfiledir + "Filter_temp_epoch{}".format(epoch))

            # save weights:
            weights_var = tf.trainable_variables()[0]
            self.updatedWeights = np.squeeze(sess.run(weights_var)).T

    # run a forward pass:
    def forwardPass(self, X):
        print('CalibEstimator.forwardPass()')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y_est=sess.run(self.tensors['y_est'], feed_dict={self.tensors['x']:X})
        return np.sqeeze(y_est)

    def getCaliratedWeights(self):
        print('CalibEstimator.getCaliratedWeights()')
        return self.updatedWeights