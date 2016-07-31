
import numpy as np

def dogsph_vals(clon, clat, q, lon_vec, lat_vec, nderiv=1):
    """
    Adopted from Pablo Muse, April 3, 2007.
    Given a gridpoint (clon,clat), return the value of the Difference-Of-Gaussian wavelet of scale
    1/2**(q-2) centered at this point, and its first derivative evaluated at all
    the input datapoints.

    Input:
    clon, clat, q                   lon/lat/scale of the DOG wavelet. q is the folding
                                    order of the triangular mesh (must be q > 2).
    nderiv                          number of derivatives to compute.
    lon_vec, lat_vec                datapoints to evaluate the wavelet.

    Output:
    fw                              output wavelet
    """

    # Perform some conversions
    phi = clon
    theta = 0.5 * np.pi - clat # make co-latitude
    phi_vec = lon_vec
    theta_vec = 0.5 * np.pi - lat_vec # make co-latitude
    ndat = lon_vec.size

    alpha = 1.25
    a0 = 1.0
    aj = a0 / 2**q
    a_alpha = aj * alpha
    norm_cst_j = 1.0 # L2-normalization

    # Data points
    ctheta_vec = np.cos(theta_vec)
    stheta_vec = np.sin(theta_vec)
    cphi_vec = np.cos(phi_vec)
    sphi_vec = np.sin(phi_vec)
    X = stheta_vec * cphi_vec
    Y = stheta_vec * sphi_vec
    Z = ctheta_vec

    # Wavelet center
    cX = np.sin(theta) * np.cos(phi)
    cY = np.sin(theta) * np.sin(phi)
    cZ = np.cos(theta)

    # Columns of fw
    # [0]   f, function value (Eq. A1)
    # [1]   df/dphi
    # [2]   df/dtheta
    # [3]   surf_del2 -- depends only on delta
    # [4]   |del f|   -- depends only on delta
    if nderiv == 0:
        ncol = 1
    elif nderiv == 1:
        ncol = 3
    elif nderiv == 2:
        ncol = 6
    else:
        assert False, 'Invalid number of derivatives to compute wavelet.'

    # Evaluate the function first
    fw = np.zeros((ndat, ncol), dtype=float)
    sqr_dist = (X - cX)**2 + (Y - cY)**2 + (Z - cZ)**2
    tan2_halfangle = sqr_dist / (4.0 - sqr_dist)
    sqrt_lambda_a = 2.0*aj / (2.0*aj*aj + 0.5*(1.0 - aj*aj)*sqr_dist)
    sqrt_lambda_a_alpha = 2.0*a_alpha / (2.0*a_alpha**2 + 0.5*(1.0 - a_alpha**2)*sqr_dist)
    fw[:,0] = (sqrt_lambda_a * np.exp(-tan2_halfangle / aj**2)
             - sqrt_lambda_a_alpha / alpha * np.exp(-tan2_halfangle / a_alpha**2))

    # Optionally, compute the derivatives
    if ncol >= 2:
        deriv_tan2_halfangle = 4.0 / (4.0 - sqr_dist)**2
        tmp = 2.0*aj*aj + 0.5*(1.0 - aj*aj)*sqr_dist
        deriv_sqrt_lambda_a = -aj*(1.0 - aj*aj) / tmp**2
        tmp_alpha = 2*a_alpha**2 + 0.5*(1.0 - a_alpha**2)*sqr_dist
        deriv_sqrt_lambda_a_alpha = -a_alpha*(1.0 - a_alpha**2) / tmp_alpha**2

        # Derivative of the pair of dilated gaussian wrt sqr_dist
        deriv_ga = ((deriv_sqrt_lambda_a - sqrt_lambda_a*deriv_tan2_halfangle / aj**2)
                  * np.exp(-tan2_halfangle / aj**2))
        deriv_ga_alpha = ((deriv_sqrt_lambda_a_alpha - sqrt_lambda_a_alpha*deriv_tan2_halfangle 
                          / a_alpha**2) * np.exp(-tan2_halfangle / a_alpha**2))

        # Derivative of the whole DOG wavelet wrt sqr_dist
        deriv_gaussians = norm_cst_j * (deriv_ga - deriv_ga_alpha / alpha)
        
        # Derivative of sqr_dist wrt colatitude theta
        deriv_sqr_dist_dth = -2.0*(cX*cphi_vec + cY*sphi_vec)*ctheta_vec + 2.0*cZ*stheta_vec

        # Derivative of sqr_dist wrt longitude phi
        deriv_sqr_dist_dph = 2.0*(cX*sphi_vec - cY*cphi_vec)*stheta_vec

        # Final result by chain rule
        fw[:,1] = deriv_gaussians * deriv_sqr_dist_dph # df/dphi
        fw[:,2] = deriv_gaussians * deriv_sqr_dist_dth # df/dtheta

    # Safety for a point on the opposite side of the sphere
    ind = (np.abs(sqr_dist - 4.0) < 1.0e-5).nonzero()[0]
    fw[ind,:] = 0.0

    # Done
    return fw


