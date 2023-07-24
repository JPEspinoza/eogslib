"""
Description:
eOGS is an rule-based evolving granular prediction system for nonlinear numerical systems

Main paper: Optimal Rule-based Granular Systems from Data Streams

This file compiles to a flexible eOGS library

"""

import pandas
import cython

def integrate_f(double a, double b):
    cdef int i

    return a * b