

@timeit
@nb.jit("f8[:,:](f8[:,:],f8,i8,i8,i8,f8,i8)",nopython=True)
def makeMask(liteMapData,pixScaleX,Nx,Ny,nHoles,holeSize,lenApodMask):

    #143x speedup with Numba

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


def addWhiteNoise(map,rmsArcmin):
    """
        Adds white noise to a given map; returns a new map
        """
    noisyMap = map.copy()
    if rmsArcmin == 0.0:
        pass
    else:
        radToMin = 180/np.pi*60
        pixArea = radToMin**2 * map.pixScaleX*map.pixScaleY
        rms = rmsArcmin/np.sqrt(pixArea)
        
        noise = np.random.normal( scale = rms, size = map.data.shape )
        
        noisyMap.data[:] += noise[:]
    
    return noisyMap

def makeTemplate(m, wl, ell, maxEll, outputFile = None):
    """
        For a given map (m) return a 2D k-space template from a 1D specification wl
        ell = 2pi * i / deltaX
        (m is not overwritten)
        """
    
    ell = np.array(ell)
    wl  = np.array(wl)
    
    
    fT = fftTools.fftFromLiteMap(m)
    print "max_lx, max_ly", fT.lx.max(), fT.ly.max()
    print "m_dx, m_dy", m.pixScaleX, m.pixScaleY
    print "m_nx, m_ny", m.Nx, m.Ny
    l_f = np.floor(fT.modLMap)
    l_c = np.ceil(fT.modLMap)
    fT.kMap[:,:] = 0.
    
    for i in xrange(np.shape(fT.kMap)[0]):
        for j in xrange(np.shape(fT.kMap)[1]):
            if l_f[i,j] > maxEll or l_c[i,j] > maxEll:
                continue
            w_lo = wl[l_f[i,j]]
            w_hi = wl[l_c[i,j]]
            trueL = fT.modLMap[i,j]
            w = (w_hi-w_lo)*(trueL - l_f[i,j]) + w_lo
            fT.kMap[i,j] = w
    
    m = m.copy()
    m.data = abs(fT.kMap)
    if outputFile != None:
        m.writeFits(outputFile, overWrite = True)
    return m


def fillWithGaussianRandomField(self,ell,Cell,bufferFactor = 1):

    ft = fftTools.fftFromLiteMap(self)
    Ny = self.Ny*bufferFactor
    Nx = self.Nx*bufferFactor
    bufferFactor = int(bufferFactor)
    realPart = np.zeros([Ny,Nx])
    imgPart  = np.zeros([Ny,Nx])
    ly = fftfreq(Ny,d = self.pixScaleY)*(2*np.pi)
    lx = fftfreq(Nx,d = self.pixScaleX)*(2*np.pi)
    modLMap = np.zeros([Ny,Nx])
    iy, ix = np.mgrid[0:Ny,0:Nx]
    modLMap[iy,ix] = np.sqrt(ly[iy]**2+lx[ix]**2)
    s = splrep(ell,Cell,k=3)
    ll = np.ravel(modLMap)
    kk = splev(ll,s)
    id = np.where(ll>ell.max())
    kk[id] = Cell[-1] 

    area = Nx*Ny*self.pixScaleX*self.pixScaleY
    p = np.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2
        
    realPart = np.sqrt(p)*np.random.randn(Ny,Nx)
    imgPart = np.sqrt(p)*np.random.randn(Ny,Nx)
    kMap = realPart+1j*imgPart
    data = np.real(ifft2(kMap))
        
    b = bufferFactor
    self.data = data[(b-1)/2*self.Ny:(b+1)/2*self.Ny,(b-1)/2*self.Nx:(b+1)/2*self.Nx]
    return(self)



