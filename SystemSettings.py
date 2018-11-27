import collections
DEFAULT_NFILT = 301
DEFAULT_MINCONVCOEFFS = 0


# return system dimensions for all data in the system
_NFilt_glob = DEFAULT_NFILT  # default number of coefficients for each wavelength
_MinConvCoeffs_glob = DEFAULT_MINCONVCOEFFS  # default minimum number of coefficients for equations

def setNFilt(nfilt):
    if nfilt is not None:
        _NFilt_glob = nfilt


def setMinConvCoeffs(MinConvCoeffs):
    if MinConvCoeffs is not None:
        _MinConvCoeffs_glob = MinConvCoeffs

def getSystemDimensions():
    SystemDimensions = collections.namedtuple('SystemDimensions', 'NCube NDD NFilt MinConvCoeffs DDx_new')
    NCube = (256, 256, 31)  # Cube image (height x width x channels (wavelengths))
    NDD = (256, 2592)  # DD image (height x width)
    NFilt = _NFilt_glob  # number of coefficients for each wavelength
    MinConvCoeffs = _MinConvCoeffs_glob  # minimum number of nonzero coefficients for each convolution output
    DDx_new = NCube[1] + NFilt - 1 - 2*MinConvCoeffs
    sysDims = SystemDimensions(NCube=NCube, NDD=NDD, NFilt=NFilt, MinConvCoeffs=MinConvCoeffs, DDx_new=DDx_new)
    return sysDims


# return the system paths in the system
def getSystemPaths(system='Server'):
    SystemPaths = collections.namedtuple('SystemPaths', 'outputBaseDir trainPath validPath testPath')

    if system=='Server':
        trainPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/Train'
        validPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/Valid'
        testPath = '/home/amiraz/Documents/CS SSI/ImageDatabase/Test'
        outputBaseDir = '/home/amiraz/Documents/CS SSI/CalibOutputs'

    sysPaths = SystemPaths(outputBaseDir=outputBaseDir, trainPath=trainPath, validPath=validPath, testPath=testPath)
    return sysPaths