from actExamples.tools import makeMask
import numpy as np

liteMapData = np.ones([1200,1200])
pixScaleX = 0.5
Ny, Nx = liteMapData.shape
nHoles = 20
holeSize = 5
lenApodMask = 0

retMap = makeMask(liteMapData,pixScaleX,Nx,Ny,nHoles,holeSize,lenApodMask)
