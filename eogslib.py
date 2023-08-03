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
INITIAL_STIEGLER = 1/(2*numpy.pi) # with stiegler initial value, granules tend to be oversized and shrink over time
INITIAL_MINIMAL = 0.01     # with minimal initial value, granules are undersized and expand over time

class Granule:
    """
    n-dimensional hyperbox representing a granule AND a rule
    """
    def __init__(
            self,
            initial_sample: int,
            input_lower_bounds: numpy.ndarray,
            input_upper_bounds: numpy.ndarray,
            output_lower_bound: numpy.ndarray,
            output_upper_bound: numpy.ndarray,
        ):
        # keeps track of the time the granule was created
        self.initial_sample = initial_sample
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds

        self.output_lower_bound = output_lower_bound
        self.output_upper_bound = output_upper_bound

        self.central_point = 0  # average
        self.dispersion = 0     # standard deviation

    def __repr__(self):
        input_dimensions = len(self.input_lower_bounds)
        output_dimensions = len(self.output_lower_bound)
        return f"Granule with {input_dimensions} input dimensions and {output_dimensions} output dimensions"

class EOGS:
    """
    eOGS is an evolving prediction system for nonlinear numerical systems
    """

    def __init__(
            self,
            smoothness:float = 0.0,
            mode:int=MODE_AUTOMATIC,
            initial_dispersion:float=INITIAL_STIEGLER,
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
        self.initial_dispersion = initial_dispersion

        # keeps track of the data dimension
        self.data_width = -1

        # keeps track of the number of samples seen
        self.height = 0

        self.granules = []

    def __repr__(self):
        # shows some stats about eogs
        pass
    

    def train(self, x: numpy.ndarray, y: float) -> tuple[float, float, float]:
        """
        Unlike normal scikit-learn libraries, eogslib train accepts a single n-dimensional sample of data
        uses it to update its internal structures and simultaneously provide a prediction for the next value

        train is supposed to be used in a loop, where the next value is fed into the system constantly

        x is a single sample of input data (n dimensions)
        y is the result of the sample (single value)

        returns a tuple with three values, a numerical prediction and a granular prediction with a lower
        and upper bounds
        """

        # if this is the first time training, set the data width
        # if data width is incorrect, raise an error
        if self.data_width == -1:
            self.data_width = len(x)
        elif self.data_width != len(x):
            raise TypeError("Data width does not match previous data width")

        # run process
        if self.height == 0:
            # create granule
            dispersion = numpy.sqrt(-2 * numpy.square(self.initial_dispersion) * numpy.log(self.alpha))

            input_lower_bounds = x - dispersion
            input_upper_bounds = x + dispersion

            output_lower_bound = y - dispersion
            output_upper_bound = y + dispersion

            granule = Granule(
                self.height, 
                input_lower_bounds, 
                input_upper_bounds,
                output_lower_bound,
                output_upper_bound)

            self.granules.append(granule)
        else: 
            # provide prediction


            # get the sample space
            dispersion = numpy.sqrt(-2 * numpy.square(self.initial_dispersion) * numpy.log(self.alpha))
            input_lower_bounds = x - dispersion
            input_upper_bounds = x + dispersion 

            fitting_granules = []

            # check if the sample space is already covered by a granule
            for granule in self.granules:
                if numpy.all(input_lower_bounds >= granule.input_lower_bounds) and numpy.all(input_upper_bounds <= granule.input_upper_bounds):
                    fitting_granules.append(granule)

            # if the sample space is not covered by a granule, create a new granule
            # if the sample space is covered by a single granule, add the sample to the granule
            # if the sample space is covered by multiple granules

            # else: adapt active granules
            pass

        # delete inactive granules and rules
        # merge granules and rules

        if(self.mode == MODE_AUTOMATIC):
            # update parameters
            pass


    def train_many(self, x: numpy.ndarray, y:numpy.ndarray) -> None:
        """
        Accepts a grid of m samples of n dimensional data
        returns a single prediction for the LAST sample
        """
        for sample, result in zip(x, y):
            self.train(sample, result)

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Accepts a single n-dimensional sample of data
        returns a single prediction for the sample
        """
        pass

