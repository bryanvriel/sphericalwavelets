
import numpy as np
from . import PoissonWavelet as pwvt
from . import DogWavelet as dwvt
import sys

def generate_vel_tensors(lon, lat, G, Gdph, Gdth, m):
    """
    Computes the spatial velocity gradient L from the multiscale velocity field.
    From this, we can decompose L into a symmetric part, D, and an antisymmetric
    part, W. D = strain-rate tensor and W = rotation-rate tensor. Here, everything 
    is defined in spherical coordinates. We store the 3 x 3 tensors as 9 x 1 vectors.

    Input:
    lat                     latitudes of grid points (radians)
    lon                     longitudes of grid points (radians)
    Gdph                    derivative of design matrix with respect to longitude
    Gdth                    derivative of design matrix with respect to latitude
    m                       length 3 array of coefficients

    Output:
    D                       strain-rate tensor
    Ws                      spherical (tangential) rotation-rate vector
    Wt                      tilt-like rotation vector
    """

    # Convert lat/lon to theta/phi
    theta = (0.5*np.pi - lat) # make co-latitude
    phi = lon
    npts = phi.size
    r = 6371009.0 # earth mean radius
    rinv = 1.0 / r

    # Compute the velocity field (convert mm/yr to m/yr)
    mn,me,mu = [0.001 * coeff for coeff in m]
    vr = np.dot(G, mu)
    vphi = np.dot(G, me)
    vtheta = np.dot(G, -mn) # minus sign b/c theta points to the south

    # Compute the gradient of the velocity field
    dvrdph = np.dot(Gdph, mu)
    dvrdth = np.dot(Gdth, mu)
    dvphdph = np.dot(Gdph, me)
    dvthdph = np.dot(Gdph, -mn)
    dvphdth = np.dot(Gdth, me)
    dvthdth = np.dot(Gdth, -mn)

    # Allocate arrays
    L = np.zeros((npts,9), dtype=float)
    D = np.zeros((npts,9), dtype=float)
    Ws = np.zeros((npts,3), dtype=float)
    Wt = np.zeros((npts,2), dtype=float)

    # Compute L terms, assuming an isotropic linear elastic, Poisson earth
    # with a free surface condition imposed
    L[:,1] = rinv * (-vtheta + dvrdth)
    L[:,2] = rinv * (-vphi + 1.0 / np.sin(theta) * dvrdph)
    L[:,4] = rinv * (vr + dvthdth)
    L[:,5] = rinv * (-vphi / np.tan(theta) + 1.0 / np.sin(theta) * dvthdph)
    L[:,7] = rinv * dvphdth
    L[:,8] = rinv * (vr + vtheta / np.tan(theta) + 1.0 / np.sin(theta) * dvphdph)
    L[:,0] = -1.0/3.0 * (L[:,4] + L[:,8])
    L[:,3] = -L[:,1]
    L[:,6] = -L[:,2]

    # Transpose of L
    LT = np.zeros(L.shape, dtype=float)
    LT[:,0] = L[:,0]
    LT[:,1] = L[:,3]
    LT[:,2] = L[:,6]
    LT[:,3] = L[:,1]
    LT[:,4] = L[:,4]
    LT[:,5] = L[:,7]
    LT[:,6] = L[:,2]
    LT[:,7] = L[:,5]
    LT[:,8] = L[:,8]

    # Compute strain tensor
    D = 0.5 * (L + LT)

    # Compute rotation vectors; divide into spherical (Ws) and tilt (Wt)
    Ws[:,0] = 0.5 * rinv * (vphi / np.tan(theta) - dvthdph / np.sin(theta) + dvphdth)
    Ws[:,1] = -rinv * vphi
    Ws[:,2] = rinv * vtheta
    Wt[:,0] = rinv * (dvrdph / np.sin(theta))
    Wt[:,1] = rinv * dvrdth

    # Done
    return D,Ws,Wt,L

def compute_scalar_fields(D, Ws, Wt):
    """
    Computes dilatation rate, strain-rate, shear-strain-rate, and magnitude of 
    rotation rate from the strain-rate tensor and rotation-rate tensor.
    """
    Drr = D[:,0]
    Drth = D[:,1]
    Drph = D[:,2]
    Dthth = D[:,4]
    Dthph = D[:,5]
    Dphph = D[:,8]

    # Compute dilatation-rate
    dilat = Drr + Dthth + Dphph

    # Compute strain-rate
    a1 = Drth**2 + Drph**2 + Dthph**2
    a2 = Drr**2 + Dthth**2 + Dphph**2
    a3 = Drr*Dthth + Drr*Dphph + Dthth*Dphph
    strain = np.sqrt(2.0*a1 + a2)
    shear = np.sqrt(2.0*a1 + 2.0/3.0*a2 - 2.0/3.0*a3)

    # Compute sphere and tilt rotation scalars
    rot_sphere = np.sqrt(Ws[:,0]**2 + Ws[:,1]**2 + Ws[:,2]**2)
    rot_tilt = np.sqrt(Wt[:,0]**2 + Wt[:,1]**2)

    # Done
    return {'dilatation': dilat, 'strain': strain, 'shear': shear, 
            'sphericalRotation': rot_sphere, 'tiltRotation': rot_tilt}

