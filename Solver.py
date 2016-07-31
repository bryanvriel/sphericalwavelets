
import numpy as np
import matplotlib.pyplot as plt
from . import SparseSolver as sp
from sklearn.cross_validation import KFold
from multiprocessing import Array, Process
import sys


class Solver():
    """
    Class to contain all the routines needed to estimate the coefficients of the
    spherical wavelets given some 3-component deformation field. Cross-validation
    routines include k-fold cross-validation and generalized cross-validation.
    """

    def __init__(self, G, S, frame, pars, reg='l2', up=True):
        """
        Initialize the solver for a given design matrix G and regularization matrix S.
         Store cross-validation methods and penalty paramters.
        """
        
        self.G = G
        self.S = S
        self.frame = frame
        self.xval = pars.xval
        self.penalty = pars.penalty
        self.ncomp = 3
        if not up:
            self.ncomp = 2
        if reg == 'l2':
            self.solver = self.l2_invert
        elif reg == 'l1':
            self.solver = self.l1_invert
        elif reg == 'l2_pos':
            self.solver = self.l2pos_invert
        elif reg == 'l2_robust':
            self.solver = self.l2_robust
        return

    def solve_multi(self, data, sigma):
        """
        Performs least-squares inversion for all three components given in gdat
        GPS data class.
        """

        G = self.G
        S = self.S
        # If requested, solve for the penalty parameter for each component
        if self.xval == 'kfold':
            penalties = self.kfold_xval(data, sigma)
        elif self.xval == 'gcv':
            penalties = self.gcv(data, sigma)
        else:
            penalties = self.penalty * np.ones(len(data))

        # Loop over the components and perform the inversions
        self.m = []
        for i in range(self.ncomp):
            self.m.append(self.solver(G, data[i], sigma[i], S, penalties[i], maxiter=5))
        if self.ncomp < 3:
            self.m.append(np.zeros_like(self.m[0]))

        # Done
        return

    def multiscale_fields(self, qmin_p, qmax_p):
        """
        Computes the predicted velocity fields for a given scale range.
        """

        # Find indices of valid scales and subset G
        ind =  self.frame[:,2] >= qmin_p
        ind *= self.frame[:,2] <= qmax_p
        G = self.G[:,ind]

        # Compute the velocity field
        out = []
        for m in self.m:
            msub = m[ind]
            out.append(np.dot(G, msub))

        # Done
        return out

    @staticmethod
    def l2_invert(G, d, Cd, S, penalty, maxiter=1):
        """
        Computes l2-regularized least-squares.
        """

        W = 1.0 / Cd
        Cm = np.linalg.inv(np.dot(G.T, dmultl(W, G)) + penalty**2 * S)
        return np.dot(Cm, np.dot(G.T, dmultl(W, d)))

    @staticmethod
    def l2pos_invert(G, d, Cd, S, penalty, maxiter=1):
        """
        Computes l2-regularized least-squares with a positivity constraint.
        """
        from scipy.optimize import nnls
        W = 1.0 / Cd
        F = np.vstack((dmultl(W, G), penalty**2 * S))
        b = np.hstack((W*d, np.zeros((G.shape[1],))))
        return nnls(F, b)[0]

    @staticmethod
    def l1_invert(Gin, din, Cd, S, penalty, maxiter=5):
        """
        Computes l1-regularized least-squares.
        """

        W = 1.0 / Cd
        G = (W * Gin.T).T
        d = W * din
        l1 = sp.BaseOpt(maxiter=maxiter)
        return l1.invert(G, d, penalty)[0]

    @staticmethod
    def l2_robust(Gin, d, Cd, S, penalty, maxiter=1):
        """
        Solves the problem:
            |d - G*m|_1 + λ*|m|_2^2
        """
        from cvxopt import matrix, spmatrix, sparse, solvers
        solvers.options['show_progress'] = False
        b = matrix(d.tolist())
        A = matrix(Gin.T.tolist())
        m,n = A.size

        # Fill q
        q = matrix(0.0, (n+m,1))
        q[n:] = 1.0

        # Fill h
        h = matrix(0.0, (2*m,1))
        h[:m] = -1.0 * b
        h[m:] =  1.0 * b

        # Fill P
        P = matrix(0.0, (n+m,n+m))
        eye = spmatrix(1.0, range(n), range(n))
        P[:n,:n] = 2.0 * penalty * matrix(S)
        P = sparse(P)

        # Fill g
        G = matrix(0.0, (2*m,n+m))
        G[:m,:n] = -1.0 * A
        G[m:,:n] =  1.0 * A
        eye = spmatrix(1.0, range(m), range(m))
        G[:m,n:] = -1.0 * eye
        G[m:,n:] = -1.0 * eye
        G = sparse(G)

        # Call solver
        x = solvers.qp(P, q, G=G, h=h)['x'][:n]
        return np.array(x).squeeze()

    def gcv(self, data, sigma):
        """
        Performs ordinary cross-validation. Only for L2-solver.
        """

        lamvec = np.logspace(-1, 4, 50)
        G = self.G
        npar = G.shape[1]
        nlam = lamvec.size
        minpen = np.zeros((3,))
        for di in range(self.ncomp):
            print(' - component', di + 1)
            rss_vec = np.zeros((nlam,), dtype=float)
            for ii in range(nlam):
                # Compute GCV residual
                penalty = lamvec[ii]
                Ginv = np.linalg.inv(np.dot(G.T,G) + penalty**2*np.eye(npar))
                H = np.dot(G, np.dot(Ginv, G.T))
                dhat = np.dot(H, data[di])
                res = (data[di] - dhat) / (1.0 - np.diag(H))
                # Sum the residuals
                rss_vec[ii] = np.sum(res**2)

            plt.loglog(lamvec, rss_vec); plt.savefig('gcv_%03d.png' % di); plt.clf()

            # Find minimum
            imin = np.argmin(rss_vec)
            minpen[di] = lamvec[imin]

        # Done
        print('penalties:', minpen)
        return minpen

    def kfold_xval(self, data, sigma, kfolds=10, up=True):
        """
        Perform k-fold cross-validation to choose the optimal penalty parameter.
        """

        print('Performing cross-validation with', kfolds, 'folds')

        # Separate indices into training and testing subsets
        ndat = data[0].size
        kf = KFold(ndat, n_folds=kfolds, shuffle=True)
        lamvec = np.logspace(-1, 3, 40)
        nlam = lamvec.size

        # Loop over data
        S = self.S
        minpen = np.zeros((3,))
        for di in range(self.ncomp):
            print(' - component', di + 1)
            err = np.zeros((kfolds, lamvec.size), dtype=float)
            ii = 0
            # Loop over k-folds
            for itrain, itest in kf:

                # Grab data based on indices
                Gtrain = self.G[itrain,:]
                dtrain = data[di][itrain]
                sigma_train = sigma[di][itrain]
                Gtest  = self.G[itest,:]
                dtest  = data[di][itest]

                # Loop over regularization parameters
                for jj in range(nlam):
                    m = self.solver(Gtrain, dtrain, sigma_train, S, lamvec[jj])
                    err[ii,jj] = np.linalg.norm(dtest - np.dot(Gtest, m))**2

                ii += 1

            # Collapse error vector by summing over all k-fold experiments
            toterr = np.mean(err, axis=0)

            # Find penalty that minimizes the error
            imin = np.argmin(toterr)
            minpen[di] = lamvec[imin]

            # plot
            plt.loglog(lamvec, toterr)
            plt.loglog(lamvec[imin], toterr[imin], 'o', markersize=10)
            plt.xlabel('Penalty')
            plt.ylabel('Mean error')
            plt.savefig('xval_%03d.png' % di); plt.clf()


        # Done
        print('penalties:', minpen)
        return minpen

    #def kalman_filter(self, data, Gtemporal, cutoff=0):
    #    """
    #    Performs Kalman filtering for full spatiotemporal inversion.
    #    """

    #    M,P = Gtemporal.shape
    #    N,Q = self.G.shape
    #    npar = cutoff + (P - cutoff)*Q

    #    # Initialize the filter
    #    Rk = 10.0 * np.ones((N,N))
    #    Qk = 0.0
    #    Pk = 1.0 * np.eye(npar)
    #    A = 1.0
    #    m0 = np.zeros((npar,))
    #    filter = kf.KalmanFilter(A, Qk, Rk, Pk, m0)

    #    # Shared memory array to store coefficients
    #    ncomp = len(data)
    #    shared = Array('d', ncomp*P*Q)
    #    msh = np.reshape(np.frombuffer(shared.get_obj()), (ncomp,P*Q))

    #    # Send each component to a processor for filtering
    #    ncomp = len(data)
    #    npass = 1
    #    threads = []
    #    for id in range(ncomp):
    #        print ' - starting filter for component', id
    #        cdat = data[id]
    #        threads.append(MPKalman(cdat, filter, self.G, Gtemporal, msh, id, npass, cutoff))
    #        threads[id].start()

    #    # Wait for jobs to finish
    #    for thrd in threads:
    #        thrd.join()

    #    # Save coefficients
    #    self.m = []
    #    for i in range(ncomp):
    #        self.m.append(msh[i,:])
        

    # End of class


#class MPKalman(Process):
#    """
#    Multiprocessing class to perform Kalman filtering on each data component.
#    """
#
#    def __init__(self, dat, filter, Gwavelet, Gtemporal, m, ind, npass, cutoff):
#
#        self.dat = dat
#        self.filter = filter
#        self.Gwavelet = Gwavelet
#        self.Gtemporal = Gtemporal
#        self.m = m
#        self.ind = ind
#        self.npass = npass
#        self.cutoff = cutoff
#        Process.__init__(self)
#
#    def run(self):
#
#        # Parameters for Kalman filter
#        penalty = 200.0**2
#        pmiter = 100
#        norm = 0.0
#        alpha = 2.0
#        cutoff = self.cutoff
#
#        # Start filter
#        nstat,nobs = self.dat.shape
#        P = self.Gtemporal.shape[1]
#        for iter in range(self.npass):
#            # Perform a forward pass through all the data
#            for i in range(nobs):
#                # Form block G matrix from product of wavelet G and current epoch time G
#                # First do non-sparse components, i.e. seasonal and secular
#                G = self.Gtemporal[i,0] * np.ones((nstat,1))
#                for j in range(1,cutoff):
#                    G = np.hstack((G, self.Gtemporal[i,j]*np.ones((nstat,1))))
#                # Now do sparse components, i.e. wavelet * iB-spline
#                for j in range(cutoff,P):
#                    G = np.hstack((G, self.Gtemporal[i,j]*self.Gwavelet))
#                # Finally, perform update for current observation
#                self.filter.block_update(G, self.dat[:,i], penalty=penalty, pmiter=pmiter, 
#                                         norm=norm, alpha=alpha, cutoff=cutoff)
#                print i
#
#        # Store result
#        self.m[self.ind,:] = self.filter.x
#        # Done
#        return


# Other matrix utilities taken from GIAnT
def dmultl(dvec, mat):
    """
    Left multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res    -> dot (diag(dvec), mat)
    """

    res = (dvec*mat.T).T
    return res

def dmultr(mat, dvec):
    """
    Right multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res     -> dot(mat, diag(dvec))
    """

    res = dvec*mat
    return res
