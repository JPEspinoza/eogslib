"""
Description:
eOGS is a rule-based evolving granular prediction system for nonlinear numerical systems

Main paper: Optimal Rule-based Granular Systems from Data Streams

This file compiles to a flexible eOGS library

"""

import pandas
import cython
import numpy

class EOGS:
    def __init__(self, alpha = 0):
        self.alpha = alpha
        pass

    def __repr__(self):
        # shows some stats about eogs
        pass

    def create_granule(self, int a):
        pass

    def garbage_collect_granules(self):
        pass

    def merge_granules(self):
        pass