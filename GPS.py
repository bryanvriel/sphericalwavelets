
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os


class GPS():
    """
    Generic class to hold three component deformation data. There's no restriction on
    the deformation type. May be velocity, displacements, or spline coefficients.
    """

    def __init__(self, dtype='velocity'):
        """
        Initialize the class.
        """

        return

    def read_3D_data(self, filename, fmt):
        """
        Read in 3D data according to a specific format. Mainly for reading in 3D velocity fields
        a la Tape et al., 2009.
        """

        # Parse the format string
        fields = fmt.split()

        # Read the data and store in lists
        rad = np.pi / 180.0
        ddict = {'name': [], 'lon': [], 'lat': [], 'east': [], 'north': [], 'up': [],
            'sigma_east': [], 'sigma_north': [], 'sigma_up': []}
        with open(filename, 'r') as fid:
            for line in fid:
                data = line.split()
                for i, field in enumerate(fields):
                    ddict[field].append(data[i])

        # Save arrays to self
        self.name = np.array(ddict['name'])
        self.lat = rad * np.array(ddict['lat'], dtype=float)
        self.lon = rad * np.array(ddict['lon'], dtype=float)
        self.data = [np.array(ddict[s], dtype=float) for s in ('north', 'east', 'up')]
        self.sigma = [np.array(ddict[s], dtype=float) for s in 
            ('sigma_north', 'sigma_east', 'sigma_up')]
    
        return
    
    def subsetData(self, bbox):
        """
        Subset network by a bounding box.
        """
        if len(bbox) != 4:
            return
        rad = np.pi / 180.0
        lonmin, lonmax, latmin, latmax = bbox
        ind  = (self.lon >= lonmin*rad) * (self.lon <= lonmax*rad)
        ind *= (self.lat >= latmin*rad) * (self.lat <= latmax*rad)
        self.lon, self.lat = self.lon[ind], self.lat[ind]
        self.data = [x[ind] for x in self.data]
        self.sigma = [x[ind] for x in self.sigma] 
        return

    def read_GPS_inversion(self, stackfile, resultsfile, index):
        """
        Read in time series inversion results for 3-component GPS data. The results are
        assumed to be stored in h5py format. This function only retrieves certain 
        coefficients of the time series model, indicated by the argument 'index'.
        """

        # Read the data stack file
        rad = np.pi / 180.0
        fin = h5py.File(stackfile, 'r')
        self.tdec = fin['tdec'].value
        self.lon = fin['lon'].value * rad
        self.lat = fin['lat'].value * rad
        fin.close()

        # Read the inversion results and make dummy sigma values (for now)
        fin = h5py.File(resultsfile, 'r')
        north = fin['mnorth'].value[:,index]
        east = fin['meast'].value[:,index]
        up = fin['mup'].value[:,index]
        sn = 5.0 * np.ones(north.shape)
        se = 5.0 * np.ones(east.shape)
        su = 5.0 * np.ones(up.shape)
        fin.close()

        # Store data in a list
        self.data = [north, east, up]
        self.sigma = [sn, se, su]

        return

    def read_GPS_timeseries(self, stackfile, resultsfile, ibeg, iend):
        """
        Read in time series inversion results for 3-component GPS data. The results are
        assumed to be stored in h5py format. This function retrieves the modeled/smoothed
        time series and computes the displacement between two time indices given by
        'ibeg' and 'iend'.
        """

        # Read the data stack file
        rad = np.pi / 180.0
        fin = h5py.File(stackfile, 'r')
        self.tdec = fin['tdec'].value
        self.lon = fin['lon'].value * rad
        self.lat = fin['lat'].value * rad
        fin.close()

        # Read the inversion results and get smoothed time series
        fin = h5py.File(resultsfile, 'r')
        snorth = fin['snorth']
        seast = fin['seast']
        sup = fin['sup']

        # Compute the displacement between indices ibeg and iend
        north = snorth[:,iend] - snorth[:,ibeg]
        east = seast[:,iend] - seast[:,ibeg]
        up = sup[:,iend] - sup[:,ibeg]
        sn = 5.0 * np.ones(north.shape)
        se = 5.0 * np.ones(east.shape)
        su = 5.0 * np.ones(up.shape)

        # Store data in a list
        self.data = [north, east, up]
        self.sigma = [sn, se, su]

        fin.close()
        return

    def read_network_timeseries(self, stackfile):
        """
        Loads the h5py files for all stations and stores the data such that at each epoch,
        the 3-component GPS displacements are stored for all stations.
        """

        # Read the data stack file
        rad = np.pi / 180.0
        fin = h5py.File(stackfile, 'r')
        self.tdec = fin['tdec'].value
        self.lon = fin['lon'].value * rad
        self.lat = fin['lat'].value * rad
        self.data = [fin['north'].value, fin['east'].value, fin['up'].value]
        fin.close()

        # Done
        return 

    def plotQuiver(self, scale=1):
        """
        Plot a rough quiver plot of the GPS data.
        """
        deg = 180.0 / np.pi
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(self.lon*deg, self.lat*deg, self.data[1], self.data[0], scale=scale)
        ax.set_aspect('equal')
        plt.show()

        return


    def removeRotation(self, rotStations=None):
        """
        Remove rotational component of velocity for GPS stations.
        """
        from .EulerPole import llh2xyz, gps2euler, euler2gps

        # Get the indices for the stations used to compute rotation
        if rotStations is not None:
            ind = np.in1d(self.names, np.array(rotStations))
        else:
            ind = np.ones(len(self.lon), dtype=bool)

        # Compute XYZ points of all GPS stations
        Pxyz = llh2xyz(self.lat, self.lon, np.zeros_like(self.lat))

        # Compute Euler pole for rotation stations
        elat,elon,omega = gps2euler(self.lat[ind], self.lon[ind], np.zeros_like(self.lat[ind]),
                                       self.data[1][ind], self.data[0][ind])
        evec_xyz = llh2xyz(elat, elon, 0.0)
        epole = omega * evec_xyz / np.linalg.norm(evec_xyz)

        # Compute Cartesian velocity at every GPS station and remove from data
        Vrot = euler2gps(epole, Pxyz.T)
        self.data[0] -= Vrot[:,1]
        self.data[1] -= Vrot[:,0]

        return
