# -*- coding: utf-8 -*-

try:
    import ConfigParser as configparser
except ImportError:
    import configparser

class CfgParser():
    """
    Reads in the configuration file for processing or plotting options. 
    Returns a dictionary of the parameters.
    """

    def __init__(self, cfgin, type='proc'):

        if type == 'proc':

            # Create configparser object
            config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
            config.read(cfgin)

            # Get values from cfg file
            self.datfile = config.get('filenames', 'input')
            self.grddir = config.get('filenames', 'griddir')
            self.dfmt = config.get('filenames', 'format')
            if self.dfmt == 'h5py':
                self.stackfile = config.get('filenames', 'stack')
                self.mindex = config.getint('tsopts', 'mindex')

            self.wvt = config.get('wavelet', 'type')
            self.mporder = config.getint('wavelet', 'mporder')
            assert self.wvt == 'dog' or self.wvt == 'poisson', 'Wavelet: must be dog or poisson'

            self.qmax = config.getint('scales', 'max')
            self.qmin = config.getint('scales', 'min')
            self.nthresh = config.getint('scales', 'nthresh')

            # Bounding box
            self.bbox = []
            for attr in ('lonmin', 'lonmax', 'latmin', 'latmax'):
                try:
                    self.bbox.append(config.getfloat('scales', attr))
                except:
                    pass

            try:
                self.modeldata = config.get('kalman', 'modeldata')
                self.timeseries = config.get('kalman', 'timeseries')
                self.dcomponent = config.get('kalman', 'component')
                self.lalpha = config.getfloat('kalman', 'lalpha')
                self.Palpha = config.getfloat('kalman', 'Palpha')
                self.slambda = config.getfloat('kalman', 'slambda')
                self.Plambda = config.getfloat('kalman', 'Plambda')
                self.tau = config.getfloat('kalman', 'tau')
                self.rwalk = config.getfloat('kalman', 'rwalk')
                print('Processing in dynamic Kalman filter mode')
            except configparser.NoSectionError:
                print('Processing in static mode')
                self.output = config.get('filenames', 'output')

            self.xval = config.get('solver', 'xval')
            self.penalty = config.getfloat('solver', 'penalty')
            self.reg = config.get('solver', 'reg')

        elif type == 'plot':

            # Create ConfigParser object
            config = configparser.RawConfigParser(allow_no_value=True, 
                                                  inline_comment_prefixes=('#',';'))
            config.read(cfgin)

            # Get values from cfg file
            self.input = config.get('filenames', 'input')
            self.output = config.get('filenames', 'output')

            task_fields = config.items('tasks')
            self.tasks = []
            for name,val in task_fields:
                self.tasks.append(name)

            self.qmin = config.getint('scales', 'min')
            self.qmax = config.getint('scales', 'max')

            self.dpi = config.getint('im_opts', 'dpi')
            self.Nlat = config.getint('im_opts', 'Nlat')
            self.Nlon = config.getint('im_opts', 'Nlon')
            self.dx = config.getfloat('im_opts', 'dx')

            try:
                self.binary_output = config.get('save_binary', 'filename')
            except configparser.NoSectionError:
                pass

        # Done
        return
