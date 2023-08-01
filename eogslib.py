"""
Description:
eOGS is a rule-based evolving granular prediction system for nonlinear numerical systems

Main paper: Optimal Rule-based Granular Systems from Data Streams

This file compiles to a flexible eOGS library

high level overview of eogs:

- granules are created from alpha-level cuts of the membership function

"""

import numpy

# MODES
MODE_AUTOMATIC = 0           # the parameters self-tune in runtime
MODE_INTERACTIVE = 1         # parameters are set on instancing

# INITIAL GRANULE VALUES
INITIAL_STIEGLER = 0    # with stiegler initial value, granules tend to be oversized and shrink over time
INITIAL_MINIMAL = 1     # with minimal initial value, granules are undersized and expand over time
INITIAL_AVERAGE = 2     # experimental method that initializes granules with same size as current granules


class Granule:
    """
    n-dimensional hyperbox representing a granule
    """
    def __init__(self):
        pass


class EOGS:
    """
    eOGS is an evolving prediction system for nonlinear numerical systems
    """

    def __init__(
            self,
            smoothness:float = 0.0,
            mode:int=MODE_AUTOMATIC,
            alpha:float = 0.0, 
            psi:float = 2,
            minimum_distance:float = 1.0,
            window:float = numpy.inf,
            initial_values:int=INITIAL_STIEGLER
    ):
        """
        Initializes the eogs system
        :param smoothness: smoothness of the output
        :param mode: mode of operation
        :param alpha:  
        :param psi: 
        :param minimum_distance: minimum distance between granules before they are merged
        :param window: number of samples to keep in memory
        :param initial_values: algorithm for initializing values when new granule is created
        """
        # keeps track of the data dimension
        self.data_width = -1

    def __repr__(self):
        # shows some stats about eogs
        pass

    def train(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Unlike normal scikit-learn libraries, eogslib train accepts a single n-dimensional sample of data
        uses it to update its internal structures and provide a prediction for the next value

        train is supposed to be used in a loop, where the next value is fed back into the system
        """

        # if this is the first time training, set the data width
        # if data width is incorrect, raise an error
        if self.data_width == -1:
            self.data_width = len(data)
        elif self.data_width != len(data):
            raise TypeError("Data width does not match previous data width")

        # run process

        return numpy.array([])

    def train_many(self, data: numpy.ndarray) -> None:
        """
        Accepts a grid of n
        """
        pass

    """
    A few 'private' functions to split the code better 
    """

    def create_granule(self, granule_a: Granule):
        pass

    def garbage_collect_granules(self):
        pass

    def merge_granules(self, granule_a: Granule, granule_b: Granule):
        """
        Merges two granules into a single granule
        """
        pass

    def specificity(self):
        """
        Calculates the specificity of the current granules
        """
        pass
