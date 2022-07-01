# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from matplotlib import rcParams
from matplotlib import rcParamsDefault
import sys
from matplotlib import pyplot as plt
from itertools import product
import numpy as np

def set_figure_settings(Figure_Type,**kwargs):
    """
    Sets the figure settings for matplotlib to be either those suitable
    for a presentation or a paper. Updates rcParams
    
    Parameters
    ----------
    Figure_Type : str
        Either 'paper' or 'presentation' to indicate figure types.
    
    **kwargs : dict
        Takes dictionary attributes given by \
        https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file.   
    """
    rcParams.update(rcParamsDefault)
    params = {}
    if Figure_Type == 'paper':
        params = {'lines.linewidth': 2,
            'lines.markersize': 5,
            'legend.fontsize': 10,
            'legend.borderpad': 0.2,
            'legend.labelspacing': 0.2,
            'legend.handletextpad' : 0.2,
            'legend.borderaxespad' : 0.2,
            'legend.scatterpoints' :1,
            'xtick.labelsize' : 10,
            'ytick.labelsize' : 10,
            'axes.titlesize' : 10,
            'axes.labelsize' : 10,
            'figure.autolayout': True,
            'font.family': 'Arial',
            'font.size': 10}
    elif Figure_Type == 'presentation':
        params = {'lines.linewidth'   : 3,
          'legend.handlelength'  : 1.0,
          'legend.handleheight'  : 1.0,
          'legend.fontsize': 16,
          'legend.borderpad': 0.2,
          'legend.labelspacing': 0.2,
          'legend.handletextpad' : 0.2,
          'legend.borderaxespad' : 0.2,
          'legend.scatterpoints' :1,
          'xtick.labelsize' : 16,
          'ytick.labelsize' : 16,
          'axes.titlesize' : 24,
          'axes.labelsize' : 20,
          'figure.autolayout': True,
          'font.size': 16.0}
    rcParams.update(params)
    rcParams.update(kwargs)