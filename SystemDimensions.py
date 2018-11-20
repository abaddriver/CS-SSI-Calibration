import collections

# return system dimensions for all data in the system
def getSystemDimensions():
    SystemDimensions = collections.namedtuple('SystemDimensions', 'NCube NDD NFilt MinConvCoeffs DDx_new')
    NCube = (256, 256, 31)  # Cube image (height x width x channels (wavelengths))
    NDD = (256, 2592)  # DD image (height x width)
    NFilt = 301  # number of coefficients for each wavelength
    MinConvCoeffs = 0  # minimum number of nonzero coefficients for each convolution output
    DDx_new = NCube[1] + NFilt - 1 - 2*MinConvCoeffs
    sysDims = SystemDimensions(NCube=NCube, NDD=NDD, NFilt=NFilt, MinConvCoeffs=MinConvCoeffs, DDx_new=DDx_new)
    return sysDims
