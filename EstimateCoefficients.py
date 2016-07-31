#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sphericalwavelets as sw
import h5py
import sys

usage = """
Usage: EstimateCoefficients.py proc.cfg
"""

def main():

    if len(sys.argv[1:]) != 1:
        print(usage)
        sys.exit()
    proccfg = sys.argv[1]

    print('\nScript to estimate multiscale velocity field from GPS data')

    # Load configuration file
    print('Reading in configurations and data')
    pars = sw.cfgp.CfgParser(proccfg)

    # Create the gps class to load the data
    gdat = sw.gps.GPS()
    gdat.read_3D_data(pars.datfile, fmt=pars.dfmt)

    # Subset network by bounding box if provided
    gdat.subsetData(pars.bbox)

    # Remove rotation from data
    gdat.removeRotation()
    #gdat.plotQuiver(scale=0.1)

    # Load the grids and return a frame
    print('Loading the spherical grids and thresholding wavelets')
    frame,qthresh = sw.grids.read_qgrids(pars)

    # Threshold the wavelets
    frame,nkeep = sw.sphw.wavelet_thresh(frame, qthresh, pars.nthresh, gdat.lon, gdat.lat)

    # Construct the design matrix G, gradients Gdph/Gdth, and regularization S
    print('Constructing design matrix G')
    G,Gdph,Gdth,S = sw.sphw.construct_Gmat(frame, gdat.lon, gdat.lat, pars)

    # Perform inversions for each component of deformation
    print('Performing velocity field inversions. Cross-validation scheme is', pars.xval)
    solver = sw.solver.Solver(G, S, frame, pars)
    solver.solve_multi(gdat.data, gdat.sigma)

    # Save relevant data to h5py output file
    print('Saving data to', pars.output)
    fout = h5py.File(pars.output, 'w')
    fout['G'] = G
    fout['S'] = S
    fout['frame'] =frame
    for i in range(len(gdat.data)):
        fout.create_dataset('m%02d' % i, data=solver.m[i])
    fout.close()  


if __name__ == '__main__':
    main()
