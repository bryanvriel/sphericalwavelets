#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.basemap import Basemap
import sphericalwavelets as sw
import h5py
import sys, os

usage = """
Usage: ViewFields.py proc.cfg plot.cfg
"""

deg = 180.0 / np.pi
rad = np.pi / 180.0

def main():

    # Get command line options
    if len(sys.argv[1:]) != 2:
        print(usage)
        sys.exit()
    proccfg, plotcfg = sys.argv[1:]

    print('\nScript for viewing spherical wavelet results')

    # Load configuration file for processing options
    pars = sw.cfgp.CfgParser(proccfg)

    # Load configuration file for plotting options
    plot_pars = sw.cfgp.CfgParser(plotcfg, type='plot')

    # Create the gps class to load the data
    print('Loading the data')
    gdat = sw.gps.GPS()
    gdat.read_3D_data(pars.datfile, fmt=pars.dfmt)

    # Subset network by bounding box if provided
    gdat.subsetData(pars.bbox)

    # Estimate a uniform rotation field and subtract from the data
    print('Removing rotation field')
    gdat.removeRotation()
    
    # Load the frame and design matrix for points
    fin = h5py.File(plot_pars.input, 'r')
    frame = fin['frame'].value
    Gpts = fin['G'].value
    qind = frame[:,2] >= float(plot_pars.qmin)
    qind *= frame[:,2] <= float(plot_pars.qmax)

    # Get the thresholds
    qthresh = sw.grids.read_qgrids(pars)[1]

    # Estimate map bounds
    bounds = get_bounds(gdat.lon, gdat.lat)

    # Load the model coefficients estimated with EstimateSphericalWavelets.py
    m = []
    for i in range(len(gdat.data)):
        m.append(fin['m%02d' % i].value[qind])
    fin.close()

    if 'fields' in plot_pars.tasks:
        gridname = 'uniformGrids_%02dmin_%02dmax.h5' % (plot_pars.qmin, plot_pars.qmax)
        if os.path.isfile(gridname):
            print(' - Reading pre-constructed uniform grids')
            grid = h5py.File(gridname, 'r')
            Ggrd = grid['Ggrd'].value
            Gdph = grid['Gdph'].value
            Gdth = grid['Gdth'].value
            Lon = grid['Lon'].value
            Lat = grid['Lat'].value
            grid.close()
        else:
            print(' - Constructing design matrix for uniform grid')
            Ggrd, Gdph, Gdth, Lon, Lat, qind = make_Gs(frame, bounds, pars, plot_pars)
            grid = h5py.File(gridname, 'w')
            grid.create_dataset('Ggrd', data=Ggrd)
            grid.create_dataset('Gdph', data=Gdph)
            grid.create_dataset('Gdth', data=Gdth)
            grid.create_dataset('Lon', data=Lon)
            grid.create_dataset('Lat', data=Lat)
            grid.close()
    dLon = abs(Lon[0,1] - Lon[0,0])*deg
    dLat = abs(Lat[1,0] - Lat[0,0])*deg
    Lat0 = Lat[0,0] * deg
    Lon0 = Lon[0,0] * deg
    print(' - Latitude spacing:', dLat)
    print(' - Longitude spacing:', dLon)
    print(' - Starting latitude:', Lat0)
    print(' - Starting longitude:', Lon0)

    # Compute velocity field at points and on grid
    print('Constructing the velocity field')
    vp = []; vg = []
    for i in range(len(gdat.data)):
        vp.append(np.dot(Gpts[:,qind], m[i]))
        if 'fields' in plot_pars.tasks:
            vg.append(np.dot(Ggrd, m[i]).reshape(Lon.shape))
 
    # Now make the map
    print('Making maps')
    fig = plt.figure(figsize=(8,10))
    map = initialize_map(bounds)
    nx = int((map.xmax-map.xmin)/plot_pars.dx)+1
    ny = int((map.ymax-map.ymin)/plot_pars.dx)+1

    # Make valid mask and transform to map coordinates
    geoMask = make_invalid_mask(Lat, Lon, frame, qthresh, 8)
    #mask = maskFromDataCovariance(Ggrd, S, 10.0, Lon.shape)
    mask = np.round(map.transform_scalar(geoMask, Lon[0,:]*deg, 
                                         Lat[:,0]*deg, nx, ny)).astype(bool)

    if 'stations' in plot_pars.tasks:
        print('Making station plot')
        xx,yy = list(map(gdat.lon*deg, gdat.lat*deg))
        mx,my = list(map(np.array(gdat.lon)*deg, np.array(gdat.lat)*deg))
        map.plot(mx, my, '^k', markersize=6)
    
    if 'quiver' in plot_pars.tasks:
        print('Making quiver plot')
        xx,yy = list(map(gdat.lon*deg, gdat.lat*deg))
        north, east = gdat.data[:2]
        Q_dat = map.quiver(xx, yy, east, north, scale=100, color='w', edgecolor='k',
                           linewidth=0.5, headwidth=5, alpha=0.8)
        qk = plt.quiverkey(Q_dat, 0.8, 0.1, 10, '10 mm/yr', labelpos='W', 
                           fontproperties={'size': 16})

    if 'quiver_model' in plot_pars.tasks:
        print('Making quiver plot')
        xx,yy = list(map(gdat.lon*deg, gdat.lat*deg))
        north, east = vp[:2]
        Q_mod = map.quiver(xx, yy, east, north, scale=100, color='k', edgecolor='k',
                           linewidth=0.5, headwidth=5)
        qk = plt.quiverkey(Q_mod, 0.8, 0.1, 10, '10 mm/yr', labelpos='W', 
                           fontproperties={'size': 16})


    if 'horizontals' in plot_pars.tasks and 'dilatation' not in plot_pars.tasks:
        print('Making image of horizontal velocity magnitude')
        vmag = np.sqrt(vg[0]**2 + vg[1]**2)
        vmag = map.transform_scalar(vmag, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        vmag[mask] = np.nan
        im = map.imshow(vmag, cmap='spectral_r')
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Surface Vel. Mag. (mm/yr)', fontsize=16)

    if 'verticals' in plot_pars.tasks:
        print('Making image of vertical velocities')
        vmag = map.transform_scalar(vg[2], Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        im = map.imshow(vmag, cmap='jet')
        #im.set_clim([-5, 5])
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Vertical Vel. (mm/yr)', fontsize=16)

    allstrain = ['strain', 'shear', 'dilatation', 'rotation_sphere', 'rotation_tilt']
    if len(np.intersect1d(plot_pars.tasks, allstrain)) > 0:
        D,Ws,Wt,L = sw.mfields.generate_vel_tensors(Lon.ravel(), Lat.ravel(), Ggrd, 
            Gdph, Gdth, m)
        strain_dict = sw.mfields.compute_scalar_fields(D, Ws, Wt)

    if 'strain' in plot_pars.tasks:
        print('Making image of strain rate')
        strain = strain_dict['strain'].reshape(Lon.shape)
        strain = map.transform_scalar(strain, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        strain[mask] = np.nan
        scale = 1e7
        im = map.imshow(scale * strain, cmap='spectral_r')
        #im.set_clim([0.0, 10.0])
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label(r'Strain Rate (10$^{-7}$ yr$^{-1}$)', fontsize=16)

    if 'shear' in plot_pars.tasks:
        print('Making image of shear strain rate')
        shear = strain_dict['shear'].reshape(Lon.shape)
        shear = map.transform_scalar(shear, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        shear[mask] = np.nan
        scale = 1e7
        im = map.imshow(scale * shear, cmap='spectral_r')
        #im.set_clim([0.0, 10.0])
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label(r'Shear Strain Rate (10$^{-7}$ yr$^{-1}$)', fontsize=16)

    if 'dilatation' in plot_pars.tasks:
        print('Making image of dilatation rate')
        dilatation = strain_dict['dilatation'].reshape(Lon.shape)
        dilatation = map.transform_scalar(dilatation, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        dilatation[mask] = np.nan
        scale = 1e7
        im = map.imshow(scale * dilatation, cmap='seismic')
        im.set_clim([-3, 3])
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label(r'Dilatation Rate (10$^{-7}$ yr$^{-1}$)', fontsize=16)

    if 'rotation_sphere' in plot_pars.tasks:
        print('Making image of spherical rotation')
        rotation = strain_dict['rotation_sphere'].reshape(Lon.shape)
        rotation = map.transform_scalar(rotation, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        scale = 1e7
        im = map.imshow(rotation * scale, cmap='spectral_r')
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label(r'Rotation Rate (10$^{-7}$ rad/yr)', fontsize=16)

    if 'rotation_tilt' in plot_pars.tasks:
        print('Making image of tilt-like rotation')
        rotation = strain_dict['rotation_tilt'].reshape(Lon.shape)
        rotation = map.transform_scalar(rotation, Lon[0,:]*deg, Lat[:,0]*deg, nx, ny)
        scale = 1e7
        im = map.imshow(rotation * scale, cmap='spectral_r')
        cb = map.colorbar(im, 'top', size='5%', pad='2%')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=16)
        cb.set_label(r'Rotation Rate (10$^{-7}$ rad/yr)', fontsize=16)

    if 'save_binary' in plot_pars.tasks:
        save_binary(plot_pars.binary_output, gdat, vg, Lon, Lat, strain_dict)
    elif 'save_mat' in plot_pars.tasks:
        save_mat(plot_pars.binary_output, gdat, vg, Lon, Lat, strain_dict)

    plt.savefig(plot_pars.output, dpi=plot_pars.dpi)
    plt.show()


def save_mat(fname, gdat, vg, Lon, Lat, strain_dict):
    """
    Save grid related outputs to file.
    """
    from scipy.io import savemat
    deg = 180.0 / np.pi
    # Prepare output dictionary
    outdict = {'gpsLat': deg*gdat.lat, 'gpsLon': deg*gdat.lon, 'gpsEast': gdat.data[1],
        'gpsNorth': gdat.data[0], 'veast': vg[1], 'vnorth': vg[0], 
        'longitude': Lon*deg, 'latitude': Lat*deg}
    for key, value in strain_dict.items():
        outdict[key] = value.reshape(Lon.shape)

    # Save to .mat
    savemat(fname, outdict)
    return

def save_binary(fname, gdat, vg, Lon, Lat, strain_dict):
    """
    Save grid related outputs to file.
    """
    with h5py.File(fname, 'w') as fid:
        fid.create_dataset('gpsLat', data=gdat.lat)
        fid.create_dataset('gpsLon', data=gdat.lon)
        fid.create_dataset('gpsEast', data=gdat.data[1])
        fid.create_dataset('gpsNorth', data=gdat.data[0])
        fid.create_dataset('veast', data=vg[1])
        fid.create_dataset('vnorth', data=vg[0])
        fid.create_dataset('longitude', data=Lon)
        fid.create_dataset('latitude', data=Lat)
        for key, value in strain_dict.items():
            fid.create_dataset(key, data=value)

    return

def get_bounds(lon, lat):

    latmin = lat.min() - 1.0*rad
    latmax = lat.max() + 1.0*rad
    lonmin = lon.min() - 1.0*rad
    lonmax = lon.max() + 1.0*rad
    return [lonmin, lonmax, latmin, latmax]

def make_Gs(frame, bounds, pars, plot_pars):
    """
    Makes design matrices for uniform grids.
    """

    # Construct a uniform grid for latitude and longitude and form G and derivatives
    lonmin,lonmax,latmin,latmax = bounds
    lat_arr = np.linspace(latmin, latmax, plot_pars.Nlat)
    lon_arr = np.linspace(lonmin, lonmax, plot_pars.Nlon)
    Lon,Lat = np.meshgrid(lon_arr, lat_arr)
    Ggrd,Gdph,Gdth = sw.sphw.construct_Gmat(frame, Lon.ravel(), Lat.ravel(), pars)[:3]

    # Threshold the wavelets based on plotting parameters
    qind = frame[:,2] >= float(plot_pars.qmin)
    qind *= frame[:,2] <= float(plot_pars.qmax)
    Ggrd = Ggrd[:,qind]
    Gdph = Gdph[:,qind]
    Gdth = Gdth[:,qind]

    #Done
    return Ggrd, Gdph, Gdth, Lon, Lat, qind

def initialize_map(bounds):
    """
    Initialize a Basemap map.
    """
    map = Basemap(projection='merc', llcrnrlat=bounds[2]*deg, llcrnrlon=bounds[0]*deg,
                  urcrnrlat=bounds[3]*deg, urcrnrlon=bounds[1]*deg, lat_ts=-30, resolution='i')
    map.drawcoastlines(linewidth=2)
    map.drawparallels(np.arange(30.0, 50.0, 1.0), labels=[1,0,0,0], fontsize=16)
    map.drawmeridians(np.arange(-130.0, -100.0, 1.0), labels=[0,0,0,1], fontsize=16)
    map.drawmapboundary()
    return map

def view_scale_map(frame, dlon, dlat):
    """
    Plots a map of the maximum scale allowed for a given data set.
    """

    # Uniform grid
    radius = 6738.0
    deg = 180.0 / np.pi
    minlat = np.amin(dlat)
    minlon = np.amin(dlon)
    maxlat = np.amax(dlat)
    maxlon = np.amax(dlon)
    lat_arr = np.linspace(maxlat, minlat, 100)
    lon_arr = np.linspace(minlon, maxlon, 100)
    Lon,Lat = np.meshgrid(lon_arr, lat_arr)
    ny,nx = Lon.shape
    scl_map = np.zeros((ny,nx))
    for ii in range(ny):
        for jj in range(nx):
            dist = (Lon[ii,jj] - frame[:,0])**2 + (Lat[ii,jj] - frame[:,1])**2
            ind = radius*dist <= 1.0
            inframe = frame[ind,2]
            if inframe.size < 1:
                scl_map[ii,jj] = 2
            else:
                scl_map[ii,jj] = np.amax(inframe)

    plt.imshow(scl_map, aspect='equal', interpolation='nearest',
               extent=(minlon*deg,maxlon*deg,minlat*deg,maxlat*deg))
    plt.plot(dlon*deg, dlat*deg, '^')
    plt.colorbar()
    plt.show()
    assert False

def make_invalid_mask(dlat, dlon, frame, qthresh, minscale):
    """
    Masks out data that do not fall within the radius of influence of a wavelet
    of a specified scale.
    """

    # Data for minimum scale we're interested in
    ang_support = np.arccos(qthresh[minscale])
    ind = np.abs(frame[:,2] - minscale) < 0.1
    sframe = frame[ind,:]

    # Loop through data and determine if points are close to a certain grid point
    ny, nx = dlat.shape
    mask = np.zeros((ny,nx), dtype=int)
    for ii in range(ny):
        for jj in range(nx):
            distance = np.sqrt((dlon[ii,jj] - sframe[:,0])**2 + (dlat[ii,jj] - sframe[:,1])**2)
            isValid = (distance <= ang_support).nonzero()[0]
            if len(isValid) < 1:
                mask[ii,jj] = 1

    return mask


def maskFromDataCovariance(G, S, penalty, imshape):

    if os.path.exists('gridDataCovariance.dat'):
        npoints = G.shape[0]
        Cd = np.fromfile('gridDataCovariance.dat')
    else:
        Cm = np.dot(G.T, G) + penalty**2 * S
        Cd = np.diag(np.dot(G, np.dot(np.linalg.inv(Cm), G.T)))
        Cd.tofile('gridDataCovariance.dat')
    sigma = np.sqrt(Cd).reshape(imshape)
    print(sigma)

    plt.imshow(sigma)
    plt.colorbar();
    plt.show()
    assert False

            

if __name__ == '__main__':
    main()
