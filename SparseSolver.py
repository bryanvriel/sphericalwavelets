# -*- coding: utf-8 -*-

"""
Class definitions for two different sparse-regularization solvers. Uses CVXOPT for convex 
optimization.

.. author:
    Bryan Riel <briel@caltech.edu>

.. dependencies:
    numpy, scipy, cvxopt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv2, schur, rsf2csf#, all_mat
#from scipy.weave import inline, converters
from sklearn.cross_validation import KFold


class GenericClass:
    """
    Dummy class that will act as a C struct.
    """
    pass


class BaseOpt:
    """
    Base optimization class for monotonic time-series, i.e. for GPS-type data.
    Successive calls are made to CVXOPT for re-weighting operations. K-fold
    cross-validation is implemented.
    """

    def __init__(self, cutoff=0, maxiter=1):
        """
        Minimizes the cost function:

        J(m) = ||A*x - b||_2^2 + lambda * ||F*x||_1

        where the subscripts denote L2- or L1-norm. F is a diagonal matrix that penalizes the 
        amplitude of the elements in m. Casts the L1-regularized cost function as a second-order 
        cone quadratic problem and solves the problem using CVXOPT. Iterative re-weighting is 
        performed to update the diagonals of F.

        Arguments:
        cutoff                  number of parameters which we DO NOT regularize in inversion. 
                                Matrix A must be structured such that these parameters are the
                                first columns of A (or parameters of x)
        maxiter                 maximum number of re-weighting iterations
        """
        self.cutoff = cutoff
        self.maxiter = maxiter

    def invert(self, Ain, bin, penalty, eps=1.0e-4, pconst=0.0):
        """
        Calls CVXOPT Cone quadratic program solver.

        Arguments:
        Ain                     input design array of size (M x N)
        bin                     input data array of size (M)
        penalty                 floating point penalty parameter (lambda)
        eps                     small number to provide stability for re-weighting.

        Returns:
        x                       regularized least-squares solution of size (N)
        q                       array of weights used in regularization (diagonals of F)
        """

        from cvxopt import matrix, spdiag, mul, div, sqrt, log
        from cvxopt import blas, lapack, solvers, sparse, spmatrix

        solvers.options['show_progress'] = False
        arrflag = isinstance(penalty, np.ndarray)

        # Convert Numpy arrays to CVXOPT matrices
        b = matrix(bin)
        A = matrix(Ain)
        m,n = A.size
        nspl = n - self.cutoff

        # Fill q (will modify for re-weighting)
        q = matrix(0.0, (2*n,1))
        q[:n] = -A.T * b
        q[n+self.cutoff:] = penalty

        # Fill h
        h = matrix(0.0, (2*n,1))

        # Fill P
        P = matrix(0.0, (2*n,2*n))
        P[:n,:n] = A.T * A
        P[list(range(n)),list(range(n))] += pconst
        P = sparse(P)

        # Fill G
        G = matrix(0.0, (2*n,2*n))
        eye = spmatrix(1.0, range(n), range(n))
        G[:n,:n] = eye
        G[:n,n:] = -1.0 * eye
        G[n:,:n] = -1.0 * eye
        G[n:,n:] = -1.0 * eye
        G = sparse(G)

        # Perform re-weighting by calling solvers.coneqp()
        for iters in range(self.maxiter):
            x = solvers.qp(P, q, G=G, h=h)['x'][:n]
            xspl = x[self.cutoff:]
            wnew = log(div(blas.asum(xspl) + nspl*eps, abs(xspl) + eps))
            if arrflag: # if outputting array, use only 1 re-weight iteration
                q[n+self.cutoff:] = wnew
            else:
                q[n+self.cutoff:] = penalty * wnew
        
        return np.array(x).squeeze(), np.array(q[n:]).squeeze()

    def xval(self, kfolds, lamvec, A, b, random_state=None):
        """
        Define K-fold cross-validation scheme. Can choose to define training
        and testing sets using the aquisition dates ('sar') or the actual
        interferogram dates ('ifg').

        Arguments:
        kfolds                      number of folds to perform
        lamvec                      array of penalty parameters to test
        Ain                         input design array of size (M x N)
        bin                         input data of size (M)

        Returns:
        lam_min                     penalty corresponding to lowest mean square error
        error                       array of shape (lamvec.shape) containing mean square error for
                                    each penalty in lamvec.
        """

        # Separate the indices into testing and training subsets
        n, npar = A.shape
        kf = KFold(n, n_folds=kfolds, indices=False, shuffle=True, random_state=random_state)

        # Loop over k-folds
        err = np.zeros((kfolds, lamvec.size), dtype=float)
        nlam = lamvec.size
        ii = 0
        offset = 0.0
        for itrain, itest in kf:

            # Grab data based on indices
            Atrain = A[itrain,:]
            btrain = b[itrain]
            Atest  = A[itest,:]
            btest  = b[itest]

            #print Atrain.shape, btrain.shape, Atest.shape, btest.shape
            #continue

            # Loop over regularization parameters
            for jj in range(nlam):
                penalty = lamvec[jj]
                x = self.invert(Atrain, btrain, penalty)[0]
                misfit = btest - np.dot(Atest, x)
                err[ii,jj] = np.dot(misfit, misfit) 

            ii += 1

        #assert False

        # Collapse error vector by summing over all k-fold experiments
        toterr = np.sum(err, axis=0)

        # Find lambda that minimizes the error
        ind = np.argmin(toterr)
        return lamvec[ind], toterr



def MatrixRoot(Z):
    """
    Performs a robust matrix square root of input array Z. Returns only a real-valued array.
    Uses Scipy algorithm by Nicholas J. Higham.
    """
    try:
        T = np.linalg.cholesky(Z)
    except np.linalg.LinAlgError:
        # Matrix not positive-definite. Cannot use Cholesky reduction.
        #  - using sqrtm instead
        T = np.real(cSqrtm(Z))
    # Check error
    if np.amax(np.dot(T.T,T) - Z) > 1.0e-8:
        print('WARNING: approximated square root is inexact')

    return T


def cSqrtm(A):
    """
    Computes custom matrix square root using Scipy algorithm with inner loop written in C.
    """

    # Schur decomposition and cast to complex array
    T, Z = schur(A)
    T, Z = rsf2csf(T,Z)
    n,n = T.shape

    # Inner loop of sqrtm algorithm -> call C code
    R = np.zeros((n,n), dtype=T.dtype)
    stat = sqrtm_loop(R, T, n)
    R, Z = all_mat(R,Z)
    X = (Z * R * Z.H)

    return X.A


def sqrtm_loop(Rarr, Tarr, n):
    """
    Inner loop in sqrtm algorithm written in C; compiled with weave.inline
    """
    # C code
    code = """
    int j = 0, i = 0, k = 0;
    std::complex<double> s;
    for (j = 0; j < n; ++j) {
        Rarr(j,j) = sqrt(Tarr(j,j));
        for (i = j-1; i > -1; --i) {
            s = 0.0;
            for (k = i+1; k < j; ++k) {
                s += Rarr(i,k) * Rarr(k,j);
            }
            Rarr(i,j) = (Tarr(i,j) - s) / (Rarr(i,i) + Rarr(j,j));
        }
    }
    return_val = 0;
    """
    
    # Return compiled function
    return inline(code,['Rarr','Tarr', 'n'], type_converters=converters.blitz)



############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
