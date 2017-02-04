import numpy as np


def logLike(theory,data,invcovmat):
    """Calculate lnLike = -0.5*chisquare for a data vector
    
    Parameters
    ----------
    theory : (N,) array_like
             A 1-d vector of the expected model
             values corresponding to each data value.

    data : (N,) array_like
           A 1-d vector of the measured data corresponding
           values corresponding to each data value.

    invcovmat : (N,N)
                An NxN matrix corresponding to the inverse
                of the covariance of the data points.


    Returns
    -------
    loglike : float
              The logarithm of the likelihood assuming it 
              is gaussian. Equal to -0.5 chisquare
              or (theory-data)^T invcovmat (theory-data)

    Examples
    --------
    
    >>> logLike(2.,1.,1.)
    -0.5

    >>> a = np.array([1.,2.])
    >>> amat = np.array([[2.,3.],[3.,2.]])
    >>> theory = np.array([1.5,2.])
    >>> logLike(theory,a,amat)
    -0.25
    
    
    """
    theory = np.asarray(theory)
    data = np.asarray(data)
    invcovmat = np.asarray(invcovmat)
    assert theory.shape==data.shape
    len_data = data.size
    if len_data>1:
        assert invcovmat.shape==(len_data,len_data)
    else:
        assert invcovmat.size==1
        
    diff = theory-data
    interm = np.dot(diff,invcovmat)
    return -0.5*np.dot(interm,diff)
