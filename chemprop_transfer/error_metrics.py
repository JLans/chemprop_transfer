# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np

def get_r2(y_true, y_pred):
    """R2 or the error.

    Parameters
    ----------
    y_true : numpy.ndarray or list
        Ground truth (correct) values

    y_pred : numpy.ndarray or list
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        R2 value.
    """
    SStot = np.sum((y_true-y_true.mean())**2)
    SSres = np.sum((y_true-y_pred)**2)
    return 1 - SSres/SStot

def get_rmse(y_true, y_pred):
    """Compute maximum absolute error.

    Parameters
    ----------
    y_true : numpy.ndarray or list
        Ground truth (correct) values.

    y_pred : numpy.ndarray or list
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The maximum absolute error times the sign of the error.
    """
    SSres = np.mean((y_true-y_pred)**2)
    return SSres**0.5

def get_max_error(y_true, y_pred):
    """Compute maximum absolute error.

    Parameters
    ----------
    y_true : numpy.ndarray or list
        Ground truth (correct) values.

    y_pred : numpy.ndarray or list
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The maximum absolute error.
    """
    return np.array(y_pred-y_true)[np.argmax(np.abs(y_pred-y_true))]
