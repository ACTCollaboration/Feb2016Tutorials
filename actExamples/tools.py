import time
import numpy as np
import numba as nb

def listFromConfig(Config,section,name):
    return [float(x) for x in Config.get(section,name).split(',')]

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__,te-ts)
        return result

    return timed


@timeit
#@nb.jit("f8[:,:](f8[:,:],f8,i8,i8,i8,f8,i8)",nopython=True)
def makeMask(liteMapData,pixScaleX,Nx,Ny,nHoles,holeSize,lenApodMask):


    pixScaleArcmin=pixScaleX*60*360/np.pi
    holeSizePix=int(holeSize/pixScaleArcmin)
    
    mask=liteMapData*0.+1.
    holeMask=mask*0.+1.
    
    xList=np.random.rand(nHoles)*Nx
    yList=np.random.rand(nHoles)*Ny
    
    for k in range(nHoles):
        holeMask[:]=1
        for i in range(Nx):
            for j in range(Ny):
            	rad=(i-int(xList[k]))**2+(j-int(yList[k]))**2
            	
            	if rad < holeSizePix**2:
                    holeMask[j,i]=0
                for pix in range(lenApodMask):
                	
                    if rad <= (holeSizePix+pix)**2 and rad > (holeSizePix+pix-1)**2:
                        holeMask[j,i]=1./2*(1-np.cos(-np.pi*float(pix)/lenApodMask))
        mask[:]*=holeMask[:]
    data=mask[:]


    return mask


