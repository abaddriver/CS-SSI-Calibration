import collections

# return system dimensions for all data in the system
def getSystemDimensions():
    SystemDimensions = collections.namedtuple('SystemDimensions', 'NCube NDD NFilt DDx_new')
    NCube = (256, 256, 31)
    NDD = [256, 2592]
    NFilt = 301
    DDx_new = NCube[1] + NFilt - 1
    sysDims = SystemDimensions(NCube=NCube, NDD=NDD, NFilt=NFilt, DDx_new=DDx_new)
    return sysDims