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
INITIAL_STIEGLER = 1/(2*numpy.pi)   # with stiegler initial value, granules tend to be oversized and shrink over time
INITIAL_MINIMAL = 0.01              # with minimal initial value, granules are undersized and expand over time


class Granule:
    """
    n-dimensional hyperbox representing a granule
    """

    def __init__(
            self,
            initial_sample: int,
            input_lower_bounds: numpy.ndarray,
            input_upper_bounds: numpy.ndarray,
            input_central_point: numpy.ndarray,
            input_dispersion: numpy.ndarray,
            output_lower_bound: numpy.ndarray,
            output_upper_bound: numpy.ndarray,
            output_central_point: numpy.ndarray,
            output_dispersion: numpy.ndarray,
        ):
        """
        Initializes a granule
        :param int initial_sample: the height of the first sample
        :param numpy.ndarray input_lower_bounds: the lower bounds of the input space (n dimensional)
        :param numpy.ndarray input_upper_bounds: the upper bounds of the input space (n dimensional)
        :param numpy.ndarray input_central_point: the central point (avg) of the input space (n dimensional)
        :param numpy.ndarray input_dispersion: the dispersion (std. deviation) of the input space (n dimensional)
        :param numpy.ndarray output_lower_bound: the lower bound of the output space (m dimensional)
        :param numpy.ndarray output_upper_bound: the upper bound of the output space (m dimensional)
        :param numpy.ndarray output_central_point: the central point of the output space (m dimensional)
        :param numpy.ndarray output_dispersion: the dispersion of the output space (m dimensional)
        """

        # check dimensionality
        if len(input_lower_bounds) != len(input_upper_bounds) != len(input_central_point) != len(input_dispersion):
            raise TypeError("Input dimensionality does not match")
        if len(output_lower_bound) != len(output_upper_bound) != len(output_central_point) != len(output_dispersion):
            raise TypeError("Output dimensionality does not match")

        # keeps track of the samples the granule contains
        self.samples = [initial_sample]

        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.input_central_point = input_central_point
        self.input_dispersion = input_dispersion

        self.output_lower_bound = output_lower_bound
        self.output_upper_bound = output_upper_bound
        self.output_central_point = output_central_point
        self.output_dispersion = output_dispersion

    def __repr__(self):
        input_dimensions = len(self.input_lower_bounds)
        output_dimensions = len(self.output_lower_bound)

        central_point_id = numpy.average(self.input_central_point)

        return (f"""
        Granule cid: {central_point_id} 
        Input dimensions: {input_dimensions}
        Output dimensions: {output_dimensions}
        Number of samples: {len(self.samples)}
        """)


class EOGS:
    """
    eOGS is an evolving prediction system for nonlinear numerical systems
    """

    def __init__(
            self,
            smoothness: float = 0.0,
            mode: int = MODE_AUTOMATIC,
            initial_dispersion: float = INITIAL_STIEGLER,
            alpha: float = 0.1,
            psi: float = 2,
            minimum_distance: float = 1.0,
            window: float = numpy.inf,
    ):
        """
        Initializes the eogs system
        :param smoothness: smoothness of the output
        :param mode: mode of operation
        :param alpha: cut point for alpha-level cuts
        :param psi: used to force gaussian dispersions to shrink or expand faster
        :param minimum_distance: minimum distance between granules before they are merged,
        higher values mean less granules and reduce computational complexity
        :param window: number of samples to keep in memory, lower values reduce space complexity
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

        # the database of current granules
        self.granules = list[Granule]()

    def __repr__(self):
        # shows some stats about eogs
        pass

    def central_point(self, sample_count: int, sample: numpy.ndarray, central_point: numpy.ndarray) -> numpy.ndarray:
        return (
                (central_point * sample_count + sample) /
                (sample_count + 1)
        )

    def dispersion(self, sample_count: int, sample: numpy.ndarray, dispersion: numpy.ndarray, lower_bounds: numpy.ndarray, central_point: numpy.ndarray) -> numpy.ndarray:
        return numpy.sqrt(
            dispersion ** 2 * (
                (   # beta
                    sample_count *
                    (central_point - lower_bounds) +
                    self.psi * numpy.abs(central_point - sample)
                ) /
                (sample_count + 1) * (central_point - lower_bounds)
            )
        )

    def distance(self, x1: Granule, x2: Granule) -> float:
        """
        returns the distance between two granules

        :param Granule x1: first granule
        :param Granule x2: second granule
        :return: the distance between x1 and x2
        :rtype: float

        TODO
        """
        pass

    def get_closest_granule(self, x1: Granule) -> Granule:
        """
        :param Granule x1:
        :return: the closest granule to x1
        :rtype: Granule

        TODO
        """
        pass

    def train(self, x, y) -> None:
        """
        Unlike normal scikit-learn libraries, eogslib train accepts a single n-dimensional sample of data
        uses it to update its internal structures and simultaneously provide a prediction for the next value

        train is supposed to be used in a loop, where the next value is fed into the system constantly

        returns a tuple with three values, a numerical prediction and a granular prediction with a lower
        and upper bounds

        :param arraylike x: single sample (n dimensions)
        :param arraylike y: single output (m dimensions)
        """

        # enforce shape (1,) and prevent unnecesary dimensions
        x = numpy.atleast_1d(numpy.squeeze(x))
        y = numpy.atleast_1d(numpy.squeeze(y))

        # if this is the first time training, set the data width
        # if data width is incorrect, raise an error
        if self.data_width == -1:
            self.data_width = len(x)
        elif self.data_width != len(x):
            raise TypeError("Data width does not match previous data width")

        # check if the sample space is already covered by a granule
        fitting_granules = []
        for granule in self.granules:
            if(
                numpy.all(granule.input_lower_bounds <= x) and
                numpy.all(granule.input_upper_bounds >= x) and
                numpy.all(granule.output_lower_bound <= y) and
                numpy.all(granule.output_upper_bound >= y)
            ):
                fitting_granules.append(granule)

        # if the sample space is not covered by a granule, "create" the new granule
        # by adding it to the granule list and preventing it from falling out of scope
        if len(fitting_granules) == 0:
            input_dispersion = numpy.full(len(x), self.initial_dispersion)
            output_dispersion = numpy.full(len(y), self.initial_dispersion)

            interval: float = numpy.sqrt(-2 * numpy.square(self.initial_dispersion) * numpy.log(self.alpha))

            input_lower_bounds = x - interval
            input_upper_bounds = x + interval

            output_lower_bound = y - interval
            output_upper_bound = y + interval

            granule = Granule(
                initial_sample=self.height,
                input_lower_bounds=input_lower_bounds, 
                input_upper_bounds=input_upper_bounds,
                input_central_point=x,
                input_dispersion=input_dispersion,
                output_lower_bound=output_lower_bound,
                output_upper_bound=output_upper_bound,
                output_central_point=y,
                output_dispersion=output_dispersion)

            self.granules.append(granule)

        # if the sample space is covered by a single granule, add the sample to the granule and update it
        elif len(fitting_granules) == 1:
            g: Granule = fitting_granules.pop()

            # forget old samples (samples that are outside the window)
            g.samples = [sample for sample in g.samples if self.height - sample > self.window]

            sample_count = len(g.samples)

            # update granule
            # update central point and dispersion
            g.input_central_point = self.central_point(sample_count, x, g.input_central_point)
            g.input_dispersion = self.dispersion(sample_count, x, g.input_dispersion, g.input_lower_bounds, g.input_central_point)

            g.output_central_point = self.central_point(sample_count, y, g.output_central_point)
            g.output_dispersion = self.dispersion(sample_count, y, g.output_dispersion, g.output_lower_bound, g.output_central_point)

            print(g)

            # update bounds
            

            # add current sample
            g.samples.append(self.height)

        # if the sample space is covered by multiple granules, merge the granules and add the sample to the new granule
        elif len(fitting_granules) > 1:
            pass

        # delete inactive granules

        if self.mode == MODE_AUTOMATIC:
            # update parameters
            pass

        self.height += 1

    def train_many(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        """
        Accepts an array of t samples of n dimensional data
        with a second array of t results of m dimensional data
        """
        for sample, result in zip(x, y):
            self.train(sample, result)

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Accepts a single n-dimensional sample of data
        returns a single m-dimensional prediction for the sample

        TODO
        """
        pass

    def predict_granular(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Accepts a single n-dimensional sample of data
        returns an m*2-dimensional interval prediction for the sample

        TODO
        """
        pass
