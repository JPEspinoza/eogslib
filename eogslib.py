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
    n-dimensional hyperbox representing a granule AND a rule
    """
    def __init__(self):
        # keeps track of the time the granule was created
        self.height = 0


class EOGS:
    """
    eOGS is an evolving prediction system for nonlinear numerical systems
    """

    def __init__(
            self,
            smoothness:float = 0.0,
            mode:int=MODE_AUTOMATIC,
            initial_values:int=INITIAL_STIEGLER,
            alpha:float = 0.1, 
            psi:float = 2,
            minimum_distance:float = 1.0,
            window:float = numpy.inf,
    ):
        """
        Initializes the eogs system
        :param smoothness: smoothness of the output
        :param mode: mode of operation
        :param alpha: cut point for alpha-level cuts
        :param psi: used to force gaussian dispersions to shrink or expand faster
        :param minimum_distance: minimum distance between granules before they are merged, higher values mean less granules and reduce computational complexity
        :param window: number of samples to keep in memory, lower values reduce space complexity
        :param initial_values: algorithm for initializing values when new granule is created
        """

        self.smoothness = smoothness
        self.mode = mode
        self.alpha = alpha
        self.psi = psi
        self.minimum_distance = minimum_distance
        self.window = window
        self.initial_values = initial_values

        # keeps track of the data dimension
        self.data_width = -1

        # keeps track of the number of samples seen
        self.height = 0

    def __repr__(self):
        # shows some stats about eogs
        pass

    def train(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Unlike normal scikit-learn libraries, eogslib train accepts a single n-dimensional sample of data
        uses it to update its internal structures and simultaneously provide a prediction for the next value

        train is supposed to be used in a loop, where the next value is fed into the system constantly
        """

        # if this is the first time training, set the data width
        # if data width is incorrect, raise an error
        if self.data_width == -1:
            self.data_width = len(data)
        elif self.data_width != len(data):
            raise TypeError("Data width does not match previous data width")

        # run process
        if self.height == 0:
            # create granule and rule
            # provide prediction
            pass
        else: 
            # provide prediction
            # if granule doesnt fit, create new granule and rule
            # else: adapt active granules
            pass

        # delete inactive granules and rules
        # merge granules and rules

        if(self.mode == MODE_AUTOMATIC):
            # update parameters
            pass

        return numpy.array([])

    def train_many(self, data: numpy.ndarray) -> None:
        """
        Accepts a grid of m samples of n dimensional data
        returns a single prediction for the LAST sample
        """
        for sample in data:
            self.train(sample)