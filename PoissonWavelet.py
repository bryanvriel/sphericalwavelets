
import numpy as np
import matplotlib.pyplot as plt
import sys

def poisson_vals(clon, clat, q, lon_vec, lat_vec, mporder, nderiv=1):
    """
    Given a gridpoint (clon,clat), return the value of the Poisson wavelet
    centered at this point, and its first derivative evaluated at all the
    input datapoints.

    Input:
    clon, clat, q                   lon/lat/scale of the wavelet. q is the folding
                                    order of the triangular mesh (must be q > 2).
    mporder                         multipole order
    nderiv                          number of derivatives to compute.
    lon_vec, lat_vec                datapoints to evaluate the wavelet.

    Output:
    fw                              output wavelet
    """

    if nderiv == 0:
        ncol = 1
    elif nderiv == 1:
        ncol = 3
    elif nderiv == 2:
        ncol = 6
    else:
        assert False, 'invalid number of derivatives: must be 0, 1, or 2'

    # Compute beta table
    beta_table = compute_beta_table(2*mporder+1)

    # Data points
    phi = clon
    theta = 0.5 * np.pi - clat # make co-latitude
    phi_vec = lon_vec
    theta_vec = 0.5 * np.pi - lat_vec # make co-latitude

    # Some constants
    ndat = lon_vec.size
    a0 = 2.33
    a_q = a0 * 0.5**q
    u = (np.sin(theta) * np.sin(theta_vec) * np.cos(phi_vec - phi) 
       + np.cos(theta) * np.cos(theta_vec))
    alpha = np.arccos(u)

    # Compute 1-D poisson wavelet
    pwvt = poisson_wvt_1D(mporder, a_q, alpha, beta_table, nderiv=nderiv)

    # Construct output matrix
    fw = np.zeros((ndat,ncol), dtype=float)
    fw[:,0] = pwvt[:,0]
    if nderiv >= 1:
        h1 = -pwvt[:,1] / np.sqrt(1.0 - u**2)
        dudth =  (np.sin(theta) * np.cos(theta_vec) * np.cos(phi_vec - phi)
               -  np.cos(theta) * np.sin(theta_vec))
        dudph =  -np.sin(theta) * np.sin(theta_vec) * np.sin(phi_vec - phi)
        fw[:,1] = h1 * dudph
        fw[:,2] = h1 * dudth
    if nderiv == 2:
        h2 = (1.0 / (1.0 - u**2)) * (pwvt[:,2] + u * h1)
        d2u_dth2 = -u
        d2u_dph2 = -np.sin(theta) * np.sin(theta_vec) * np.cos(phi_vec - phi)
        d2u_dthdph = -np.sin(theta) * np.cos(theta_vec) * np.sin(phi_vec - phi)
        fw[:,3] = h2 * dudph**2 + h1 * d2u_dph2
        fw[:,4] = h2 * dudth**2 + h1 * d2u_dth2
        fw[:,5] = h2 * du_dth * du_dph + h1 * d2u_dthdph

    # Done
    return fw

def poisson_wvt_1D(mporder, scale, colat, beta_table, nderiv=0):
    """
    Computes a 1D profile of a poisson wavelet centered at the north pole.
    Parameterized only w.r.t. the colatitude.

    Inputs:
    mporder                 multipole order
    scale                   multipole located at exp(-scale) from the origin
    colat                   evaluation points (phi = 0) (can be array)
    beta_table              coefficients table for thte poisson wavelet computation
    nderiv                  number of derivatives to compute (default 0)

    Output:
    fw                      wavelet and derivatives
    """

    n = mporder
    a = scale
    npts = colat.size
    theta = colat
    fw = np.zeros((npts,nderiv+1), dtype=float)

    # Center at north pole
    cX = np.sin(theta)
    cY = 0.0
    cZ = np.cos(theta)

    # Compute wavelet
    invnorm_deriv = diff_invnorm(n+1, np.exp(-a), cX, cY, cZ)
    tmp = np.zeros((npts,), dtype=float)
    coeff = np.zeros((n+1,), dtype=float)
    for k in range(n+1):
        beta1 = beta_table[n+1,k]
        beta2 = beta_table[n,k]
        coeff[k] = np.exp(-k*a) * (2.0*beta1 + beta2)
        tmp += coeff[k] * invnorm_deriv[k,:]

    # Compute the L2-norm
    invnorm_deriv_np = diff_invnorm(2*n+1, np.exp(-2*a), 0.0, 0.0, 1.0)
    tmp2 = 0.0
    for k in range(2*n+1):
        beta1 = beta_table[2*n+1,k]
        beta2 = beta_table[2*n,k]
        coeff2 = np.exp(-2.0*k*a) * (2.0*beta1 + beta2)
        tmp2 += coeff2 * invnorm_deriv_np[k]
    norm = np.sqrt(4.0 * np.pi) * 2.0**(-n) * np.sqrt((2.0*a)**(2*n) * tmp2)

    # Store the normalized wavelet in the first output column
    psi = a**n * tmp / norm
    fw[:,0] = psi

    # Compute derivatives using finite differences
    if nderiv >= 1:
        h = np.pi / 4000.0
        X = np.sin(theta + h)
        Y = 0.0
        Z = np.cos(theta + h)
        invnorm_deriv_dtheta = diff_invnorm(n+1, np.exp(-a), X, Y, Z)
        tmp_dtheta = np.zeros((npts,), dtype=float)
        for k in range(n+1):
            tmp_dtheta += coeff[k] * invnorm_deriv_dtheta[k,:]
        psi_dtheta = a**n * tmp_dtheta / norm
        dpsidtheta = (psi_dtheta - psi) / h
        fw[:,1] = dpsidtheta
    if nderiv == 2:
        X = np.sin(theta - h)
        Y = 0.0
        Z = np.cos(theta - h)
        invnorm_deriv_dtheta_minus = diff_invnorm(n+1, np.exp(-a), X, Y, Z)
        tmp_dtheta_minus = np.zeros((npts,), dtype=float)
        for k in range(n+1):
            tmp_dtheta_minus += coeff[k] * invnorm_deriv_dtheta_minus[k,:]
        psi_dtheta_minus = a**n * tmp_dtheta_minus / norm
        d2psidtheta2 = (psi_dtheta - 2.0*psi + psi_dtheta_minus) / h**2
        fw[:,2] = d2psidtheta2

    # Done
    return fw

def compute_beta_table(n):
    """
    Constructs the beta table for Poisson wavelets.
    """

    beta = np.zeros((n+1,n+1), dtype=float)
    beta[0,0] = 1.0
    for i in range(1,n+1):
        for j in range(1,i+1):
            beta[i,j] = beta[i-1,j-1] + (j - 1.0) * beta[i-1,j]

    # Done
    return beta

def diff_invnorm(n, r, X, Y, Z):
    """
    Computes n-th order derivative for Poisson wavelet.
    """

    dz = Z - r
    d2 = X**2 + Y**2 + dz**2

    if isinstance(X, np.ndarray):
        f = np.zeros((n,X.size), dtype=float)
    else:
        f = np.zeros((n,1), dtype=float)

    if n >= 1:
        # 1st derivative
        f[0,:] = dz / d2**1.5
    if n >= 2:
        # 2nd derivative
        f[1,:] = 3*dz**2 / d2**2.5 - 1.0 / d2**1.5
    if n >= 3:
        # 3rd derivative
        f[2,:] = 15*dz**3 / d2**(7.0/2) - 9.0*dz / d2**2.5
    if n >= 4:
        # 4th derivative
        f[3,:] = 105*dz**4 / d2**(9.0/2) - 90*dz**2 / d2**(7.0/2) + 9 / d2**2.5
    if n >= 5:
        # 5th derivative
        f[4,:] = 945*dz**5 / d2**(11.0/2) - 1050*dz**3 / d2**(9.0/2) + 225*dz / d2**3.5
    if n >= 6:
        # 6th derivative
        f[5,:] = (10395*dz**6 / d2**(13.0/2) - 14175*dz**4 / d2**(11.0/2)
                + 4725*dz**2 / d2**(9.0/2) - 225 / d2**(7.0/2))
    if n >= 7:
        # 7th derivative
        f[6,:] = (135135*dz**7 / d2**(15.0/2) - 218295*dz**5 / d2**(13.0/2) 
                + 99225*dz**3 / d2**(11.0/2) - 11025*dz / d2**(9.0/2))
    if n >= 8:
        # 8th derivative
        f[7,:] = (2027025*dz**8 / d2**(17.0/2) - 3783780*dz**6 / d2**(15.0/2) 
                + 2182950*dz**4 / d2**(13.0/2) - 396900*dz**2 / d2**(11.0/2) 
                + 11025 / d2**(9.0/2))
    if n >= 9:
        # 9th derivative
        f[8,:] = (34459425*dz**9 / d2**(19.0/2) - 72972900*dz**7 / d2**(17.0/2) 
                + 51081030*dz**5 / d2**(15.0/2) - 13097700*dz**3 / d2**(13.0/2) 
                + 893025*dz / d2**(11.0/2))
    if n >= 10:
        # 10th derivative
        f[9,:] = (654729075*dz**10 / d2**(21.0/2) - 1550674125*dz**8 / d2**(19.0/2)
               + 1277025750*dz**6 / d2**(17.0/2) - 425675250*dz**4 / d2**(15.0/2)
               + 49116375*dz**2 / d2**(13.0/2) - 893025 / d2**(11.0/2))
    if n >= 11:
        # 11th derivative
        f[10,:] = (13749310575*dz**11 / d2**(23.0/2) - 36010099125*dz**9 / d2**(21.0/2)
                 + 34114830750*dz**7 / d2**(19.0/2) - 14047283250*dz**5 / d2**(17.0/2) 
                 + 2341213875*dz**3 / d2**(15.0/2) - 108056025*dz / d2**(13.0/2))

    # Done
    return f

