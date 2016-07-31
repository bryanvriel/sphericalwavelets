
import numpy as np
import matplotlib.pyplot as plt
from . import DogWavelet as dwvt
from . import PoissonWavelet as pwvt
import sys

def construct_Gmat(frame, dlon, dlat, pars, secderivs=False):
    """
    Constructs the design matrix G for given q-grids and observation locations.

    Inputs:
    frame                   N x 3 array with columns lon-center, lat-center, q
    dlon, dlat              lon/lat of observation points
    pars                    CfgParser instance containing processing parameters
    secderivs               boolean to compute second derivatives

    Outputs:
    G, Gdph, Gdth [Gdphdph, Gdthdth, Gdthdph]
    """

    # Initialize arrays
    ndat = dlon.size
    ngrid = frame.shape[0]
    G = np.zeros((ndat,ngrid), dtype=float)
    Gdph = np.zeros((ndat,ngrid), dtype=float)
    Gdth = np.zeros((ndat,ngrid), dtype=float)
   
    # If requested, intitialize arrays for second derivatives
    nderiv = 1
    if secderivs:
        nderiv = 2
        Gdphdph = np.zeros((ndat,ngrid), dtype=float)
        Gdthdth = np.zeros((ndat,ngrid), dtype=float)
        Gdthdph = np.zeros((ndat,ngrid), dtype=float)

    # Fill each column of G with a basis function evaluated at all the datapoints
    for jj in range(ngrid):
        if pars.wvt == 'dog':
            fw = dwvt.dogsph_vals(frame[jj,0], frame[jj,1], frame[jj,2], dlon, dlat, nderiv=nderiv)
        else:
            fw = pwvt.poisson_vals(frame[jj,0], frame[jj,1], frame[jj,2], dlon, dlat, 
                                   pars.mporder, nderiv=nderiv)
        G[:,jj] = fw[:,0]
        Gdph[:,jj] = fw[:,1]
        Gdth[:,jj] = fw[:,2]
        if secderivs:
            Gdphdph[:,jj] = fw[:,3]
            Gdthdth[:,jj] = fw[:,4]
            Gdthdph[:,jj] = fw[:,5]

    # Construct the regularization matrix: norm of the gradient
    S = np.diag(2.0 ** (frame[:,2])) 

    if secderivs:
        return G, Gdph, Gdth, S, Gdphdph, Gdthdth, Gdthdph
    else:
        return G, Gdph, Gdth, S

def wavelet_thresh(frame, qthresh, nthresh, dlon, dlat):
    """
    Performs wavelet thresholding. First restricts the lowest-scale wavelet based
    on the size of the GPS network. Then loops through the gridpoints and determines 
    which ones have n > nthresh data points that are within arccos(qthresh).

    Inputs:
    frame                   Full Ngrid x 3 frame
    qthresh                 threshold for cos(angular_distance)
    nthresh                 minimum number of stations within qthresh
    dlon, dlat              data lon/lat

    Output:
    oframe                  output reduced frame
    nkeep                   number of data points per output frame point
    """

    # Estimate outer length scale of network
    Atot = latlon_area(dlon, dlat)
    Lscale = 2.0 * np.sqrt(Atot / np.pi)

    # Lower bound on wavelets
    ang_support_meters = np.arccos(qthresh) * 6378137.0
    qmin = np.amin((ang_support_meters < 2.0*Lscale).nonzero()[0])
    print(' - Minimum q:', qmin)
    ind = frame[:,2] >= qmin
    oframe = frame[ind,:]
    
    ngrid = oframe.shape[0]
    
    # Data points
    theta = 0.5*np.pi - dlat # make co-latitude
    phi = dlon
    stheta = np.sin(theta)
    ctheta = np.cos(theta)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # Q-grid points
    theta_grd = 0.5*np.pi - oframe[:,1] # make co-latitude
    phi_grd = oframe[:,0]
    stheta_grd = np.sin(theta_grd)
    ctheta_grd = np.cos(theta_grd)
    sphi_grd = np.sin(phi_grd)
    cphi_grd = np.cos(phi_grd)

    # Convert to XYZ
    xyz_dat = np.vstack((stheta*cphi, stheta*sphi, ctheta))                     # 3 x Ndat
    xyz_grd = np.vstack((stheta_grd*cphi_grd, stheta_grd*sphi_grd, ctheta_grd)) # 3 x Ngrid

    # Loop through the grid points and check station support
    ikeep = []
    nkeep = []
    for i in range(ngrid):
        cos_dist = np.dot(xyz_grd[:,i], xyz_dat)
        ind = int(oframe[i,2])
        in_support = (cos_dist > qthresh[ind]).nonzero()[0]
        if in_support.size > nthresh:
            ikeep.append(i)
            nkeep.append(in_support.size)

    # Done
    return oframe[ikeep,:], nkeep

def latlon_area(dlon, dlat, radius=6378137.0):
    """
    Utility function to compute the area of a square patch on the sphere 
    described by two bounding latitude lines and two bounding longitude lines.
    """

    lonmin = np.amin(dlon)
    lonmax = np.amax(dlon)
    latmin = np.amin(dlat)
    latmax = np.amax(dlat)

    # Order points counter-clockwise
    Plat = np.array([latmin, latmin, latmax, latmax])
    Plon = np.array([lonmin, lonmax, lonmax, lonmin])
    Pxyz = llh2xyz_spherical(Plat, Plon, np.zeros(Plat.shape), radius=1.0)

    # Divide square into a NW triangular patch and a SE triangular patch
    Anw = triarea(Pxyz[:,0], Pxyz[:,2], Pxyz[:,3])
    Ase = triarea(Pxyz[:,0], Pxyz[:,1], Pxyz[:,2])

    # Done
    return radius**2 * (Anw + Ase)

def triarea(v1, v2, v3):
    """
    Given three vectors on the unit sphere, compute the area of the spherical
    triangular patch formed by the three vectors.
    """
    
    a = np.arccos(np.dot(v3, v2))
    b = np.arccos(np.dot(v1, v2))
    c = np.arccos(np.dot(v1, v3))
    s = 0.5 * (a + b + c)
    
    return (4.0 * np.arctan(np.sqrt(np.tan(0.5*s) * np.tan(0.5*(s - a)) 
          * np.tan(0.5*(s - b)) * np.tan(0.5*(s - c)))))

def llh2xyz_spherical(lat, lon, h, radius=6378137.0):
    """
    Convert lat/lon/h to spherical coordinates for a given Earth radius.
    """
    
    x = np.zeros((3,lat.size), dtype=float)
    x[0,:] = radius * np.cos(lat) * np.cos(lon)
    x[1,:] = radius * np.cos(lat) * np.sin(lon)
    x[2,:] = radius * np.sin(lat)

    return x
