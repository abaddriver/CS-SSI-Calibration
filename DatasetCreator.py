from SSIImageHandler import SSIImageHandler
import numpy as np
from os import listdir
from os.path import isfile, join

# Class DatasetCreator
# -------------
# this is a tool designed to load the image dataset
# the output treats every row of every image as a different example
# the dataset contains the following dimensions:
# dataset['Cubes'] -  (NCube_y * #Images, 1, NCube_x, L)
# dataset['DDs'] - (NDD_y * #Images, 1, NDD_x, 1)
class DatasetCreator:
    def __init__(self,
                 directory,
                 NCube, # the dimensions of the spectral cube as [Nx, Ny, L] format
                 NDD,# the dimensions of the DD image, as [Nx, Ny] format
                 maxNExamples = -1): # for testing: if specified, dataset will contain a maximum of maxNExamples images
        print('DatasetCreator.__init__()')
        self.filenames = {}
        self.dataset = {}
        self.maxNExamples = maxNExamples

        if NDD[0] != NCube[0]:
            print('Error: DatasetCreator init(): Cube and DD image differ in height')
            self.directory = ''
        else:
            self.directory = directory
            self.NCube = tuple(NCube)
            self.NDD = tuple(NDD)
            self.DDx_cropInds = (0, NDD[1])
            self.DDx_crop = NDD[1]

    def getFileLists(self):
        print('DatasetCreator.getFileLists()')
        # find all the filenames that have both .rawCube and .rawImage endings for dataset
        dirCubeFiles = [f for f in listdir(self.directory) if
                        (isfile(join(self.directory, f)) and f.endswith('_Cube.rawImage'))]
        dirImgNames = list(map(lambda f: f.partition('_Cube')[0], dirCubeFiles))
        filenames = list(filter(lambda f: isfile(join(self.directory, f+'_DD.rawImage')), dirImgNames))

        # keep up to maxNExamples images in database:
        if ((self.maxNExamples != -1) and (len(filenames) > self.maxNExamples)):
            filenames = filenames[0:self.maxNExamples]

        # create the image lists
        CubeList = list(map(lambda f: join(self.directory, f + '_Cube.rawImage'), filenames))
        DDList = list(map(lambda f: join(self.directory, f + '_DD.rawImage'), filenames))

        return (CubeList, DDList)

    def buildDataset(self):
        print('DatasetCreator.buildDataset()')

        # get file lists:
        (CubeList, DDList) = self.getFileLists()

        # iterate over filenames and read Cube and DD images
        h = SSIImageHandler()
        self.filenames = []
        h_index_start = 0
        Cubes = np.empty([self.NCube[0]*len(CubeList), 1, self.NCube[1], self.NCube[2]], dtype=float)
        DDs = np.empty([self.NDD[0]*len(CubeList), 1, self.DDx_crop, 1])

        for cubepath, ddpath in zip(CubeList, DDList):
            cube = h.readImage(cubepath)
            ddIm = h.readImage(ddpath)

            #  check that image dimensions match the specification
            # if it does, inset to database
            if ((tuple(cube.shape) == self.NCube) and (tuple(ddIm.shape[0:2]) == tuple(self.NDD[0:2]))):

                # crop DD image if needed:
                ddIm = ddIm[:, self.DDx_cropInds[0]:self.DDx_cropInds[1]]

                # add images to database
                self.filenames.append((cubepath, ddpath))
                Cubes[h_index_start:h_index_start+self.NCube[0],:, :, :] = np.reshape(cube, (cube.shape[0],1,cube.shape[1],cube.shape[2]))
                DDs[h_index_start:h_index_start + self.NDD[0], :, :, :] = np.reshape(ddIm, (ddIm.shape[0], 1, ddIm.shape[1], 1))
                h_index_start += self.NDD[0]

        # if there are images with mismatching sizes - remove unused rows:
        if len(self.filenames)!= len(CubeList):
            print('warning: DatasetCreator.buildDataset() - some images contained different sizes than expected')
            Cubes = np.delete(Cubes, range(h_index_start, self.NCube[0]*len(CubeList)), 0)
            DDs = np.delete(DDs, range(h_index_start, self.NDD[0] * len(CubeList)), 0)
        self.dataset['Cubes'] = Cubes
        self.dataset['DDs'] = DDs

    def cropDDWidth(self, DDx_crop):
        assert (DDx_crop % 2 == 0) # new size must be even
        self.DDx_crop = DDx_crop
        ind_begin = int((self.NDD[1] - DDx_crop)/2)
        ind_end = ind_begin + DDx_crop
        self.DDx_cropInds = (ind_begin, ind_end)
        if self.dataset:
            self.dataset['DDs'] = self.dataset['DDs'][:, :, ind_begin:ind_end, :]


    def getDataset(self):
        print('DatasetCreator.getDataset()')
        if not self.dataset:
            self.buildDataset()
        return self.dataset