
import numpy as np
import sys

def read_qgrids(pars):
    """
    Function to read in the spherical grids generated from getsubgrids.f90.
    Note: convert co-latitude generated from the fortran routine to normal latitude.
    """

    # Read the grids
    deg = 180.0 / np.pi
    start = True
    for ii in range(pars.qmin, pars.qmax+1):
        scl = 'q%02d' % ii
        dat = np.loadtxt(pars.grddir + '/thph_' + scl + '.dat')
        if start:
            lat = 0.5*np.pi - dat[:,0]
            lon = dat[:,1]
            q = ii * np.ones((lat.size,), dtype=float)
            start = False
        else:
            lat = np.hstack((lat, 0.5*np.pi - dat[:,0]))
            lon = np.hstack((lon, dat[:,1]))
            q = np.hstack((q, ii*np.ones((dat[:,0].size,), dtype=float)))

    # Pre-calculated angular support of wavelets
    if pars.wvt == 'dog':
        ang_support = np.array([82.44146828403895, 47.31035849423593, 24.707514995576712, 
                                12.499019815878315, 6.268153313525013, 3.1364227674948486, 
                                1.5685051418745237, 0.7842893061821807, 0.39214924548067165, 
                                0.19607519680420418, 0.09803767016053697, 0.04901884405008221, 
                                0.024509423146266033])
    elif pars.wvt == 'poisson':
        ang_support = np.array([35.70358244703732, 27.853737635854497, 18.5467216726992, 
                                11.0249226026565, 6.087439077124019, 3.2133751600836407, 
                                1.6532754803935261, 0.8388870313498137, 0.4225871551557125, 
                                0.21209006663400015, 0.10624550521101161, 0.053173041029848656, 
                                0.02659911403396257])
    qthresh = np.cos(ang_support / deg)

    # Return Ngrd x 3 matrix
    return np.transpose(np.vstack((lon,lat,q))), qthresh
