"""
Description:
eOGS is a rule-based evolving granular prediction system for nonlinear numerical systems

Main paper: Optimal Rule-based Granular Systems from Data Streams

This file compiles to a flexible eOGS library

high level overview of eogs:

- granules are created from alpha-level cuts of the membership function

"""

import numpy
import itertools

# MODES
MODE_AUTOMATIC = 0           # the parameters self-tune in runtime
MODE_INTERACTIVE = 1         # parameters are set on instancing

# INITIAL GRANULE VALUES
INITIAL_STIEGLER = 1/(2*numpy.pi)   # with stiegler initial value, granules tend to be oversized and shrink over time
INITIAL_MINIMAL = 0.01              # with minimal initial value, granules are undersized and expand over time

class Sample:
    def __init__(
            self,
            height: int,
            input: numpy.ndarray,
            output: numpy.ndarray):
        self.height = height
        self.input = input
        self.output = output

    def __repr__(self):
        return f"Sample: {self.height} {self.input} {self.output}"

class Granule:
    """
    n-dimensional hyperbox representing a granule
    """

    def __init__(
            self,
            initial_samples: list[Sample],
            input_lower_bounds: numpy.ndarray,
            input_upper_bounds: numpy.ndarray,
            input_central_point: numpy.ndarray,
            input_dispersion: numpy.ndarray,
            output_lower_bounds: numpy.ndarray,
            output_upper_bounds: numpy.ndarray,
            output_central_point: numpy.ndarray,
            output_dispersion: numpy.ndarray,
            coefficients: numpy.ndarray,
        ):
        """
        Initializes a granule
        :param int initial_sample: the initial sample of the granule
        :param numpy.ndarray input_lower_bounds: the lower bounds of the input space (1,n)
        :param numpy.ndarray input_upper_bounds: the upper bounds of the input space (1,n)
        :param numpy.ndarray input_central_point: the central point (avg) of the input space (1,n)
        :param numpy.ndarray input_dispersion: the dispersion (std. deviation) of the input space (1,n)
        :param numpy.ndarray output_lower_bound: the lower bound of the output space (1,m)
        :param numpy.ndarray output_upper_bound: the upper bound of the output space (1,m)
        :param numpy.ndarray output_central_point: the central point of the output space (1,m)
        :param numpy.ndarray output_dispersion: the dispersion of the output space (1,m)
        :param numpy.ndarray coefficients: the output coefficients of the granule (n+1,m)
        """

        # check dimensionality
        if len(input_lower_bounds) != len(input_upper_bounds) != len(input_central_point) != len(input_dispersion):
            raise TypeError("Input dimensionality does not match")
        if len(output_lower_bounds) != len(output_upper_bounds) != len(output_central_point) != len(output_dispersion):
            raise TypeError("Output dimensionality does not match")

        # keeps track of the samples the granule contains
        self.samples = initial_samples

        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.input_central_point = input_central_point
        self.input_dispersion = input_dispersion

        self.output_lower_bounds = output_lower_bounds
        self.output_upper_bounds = output_upper_bounds
        self.output_central_point = output_central_point
        self.output_dispersion = output_dispersion

        self.coefficients = coefficients

    def __repr__(self):
        return f"""
Central ID: {numpy.average(self.input_central_point)} 
Input dimension: {len(self.input_lower_bounds)}
Output dimension: {len(self.output_lower_bounds)}
Number of samples: {len(self.samples)}
Samples: {self.samples}
Input lower id: {numpy.average(self.input_lower_bounds)}
Input upper id: {numpy.average(self.input_upper_bounds)}
Output lower id: {numpy.average(self.output_lower_bounds)}
Output upper id: {numpy.average(self.output_upper_bounds)}"""


class EOGS:
    """
    eOGS is an evolving prediction system for nonlinear numerical systems

    it accepts samples of n dimensions and returns outputs of m dimensions

    it is designed to handle data streams where the results become available delayed
    ex: meteorological data, stock market data, etc.

    To use with delayed results:
    1. train with your current samples 
    2. predict the current sample without a result
    3. when the result becomes available, train with the sample and result
    """

    def __init__(
            self,
            smoothness: float = 0.0,
            mode: int = MODE_AUTOMATIC,
            initial_dispersion: float = INITIAL_STIEGLER,
            alpha: float = 0.1,
            psi: float = 2,
            minimum_distance: float = 0.0,
            window: float = 1000,
    ):
        """
        Initializes the eogs system
        :param smoothness: smoothness of the output
        :param mode: mode of operation
        :param alpha: cut point for alpha-level cuts. Lower means larger intervals
        :param psi: used to force gaussian dispersions to shrink or expand faster
        :param minimum_distance: minimum distance between granules before they are merged, higher values reduce complexity. If 0 or below, merge is disabled
        :param window: maximum age of samples. Default infinite. Info, samples are never kept in memory, this simply kills granules that are too old
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
        return f""

    def central_point(self, sample_count: int, sample: numpy.ndarray, central_point: numpy.ndarray) -> numpy.ndarray:
        """
        Returns the equation to update the central point of a granule

        new_u = v * old_u + x / n + 1

        where:
        old_u is the old central_point
        v is the number of samples
        x is the new sample

        :param int sample_count: the number of samples
        :sample numpy.ndarray sample: the new sample (n dimensional)
        :central_point numpy.ndarray central_point: the old central point (n dimensional)
        :return: the new central point (n dimensional)
        :rtype numpy.ndarray:
        """
        return (
            (central_point * sample_count + sample) /
            (sample_count + 1)
        )

    def dispersion(
            self, 
            sample_count: int, 
            sample: numpy.ndarray, 
            dispersion: numpy.ndarray, 
            lower_bounds: numpy.ndarray, 
            central_point: numpy.ndarray) -> numpy.ndarray:
        """
        Does the equation calculate the dispersion of a granule, used for updating granules

        new_o**2 = b * old_o**2
        where: 
        old_o is the old dispersion
        b is the beta value

        b = (v * (u - l) + psi * |u - x|) / (v + 1) * (u - l)
        where:
        v is the number of samples

        """
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

    def interval(self, dispersion: numpy.ndarray) -> numpy.ndarray:
        """
        :param dispersion: the dispersion of the granule (n dimensional) (1,n)
        :return: the interval of the granule (n dimensional) (1,n)
        :rtype numpy.ndarray:
        """
        return numpy.sqrt(-2 * numpy.log(self.alpha) * numpy.square(dispersion) )

    def create_granule(self, 
            initial_samples: list[numpy.ndarray],
            input_central_point: numpy.ndarray, 
            input_dispersion: numpy.ndarray, 
            output_central_point: numpy.ndarray, 
            output_dispersion: numpy.ndarray) -> Granule:

        input_lower_bounds = input_central_point - self.interval(input_dispersion)
        input_upper_bounds = input_central_point + self.interval(input_dispersion)

        output_lower_bounds = output_central_point - self.interval(output_dispersion)
        output_upper_bounds = output_central_point + self.interval(output_dispersion)

        """
        # calculate coefficients to use for predictions
        [ [ y1, y2, ... ,ym],
          [ 0,  0,  ..., 0 ],
          ...
          [ 0,  0,  ..., 0 ]]

        n rows, m columns
        """
        # top row
        coefficients_0 = numpy.array(output_central_point)
        # zero rows
        coefficients_n = numpy.zeros((len(input_central_point),len(output_central_point)))
        # assemble coefficients
        coefficients = numpy.vstack([coefficients_0, coefficients_n])

        return Granule(
            initial_samples=initial_samples,
            input_lower_bounds=input_lower_bounds,
            input_upper_bounds=input_upper_bounds,
            input_central_point=input_central_point,
            input_dispersion=input_dispersion,
            output_lower_bounds=output_lower_bounds,
            output_upper_bounds=output_upper_bounds,
            output_central_point=output_central_point,
            output_dispersion=output_dispersion,
            coefficients=coefficients)

    def update_granule(self, g: Granule, x: numpy.ndarray, y: numpy.ndarray) -> None:
        """
        Takes a granule and a sample and updates the granule
        """
        sample_count = len(g.samples)

        # update granule
        # update central point and dispersion
        g.input_central_point = self.central_point(sample_count, x, g.input_central_point)
        g.input_dispersion = self.dispersion(sample_count, x, g.input_dispersion, g.input_lower_bounds, g.input_central_point)

        g.output_central_point = self.central_point(sample_count, y, g.output_central_point)
        g.output_dispersion = self.dispersion(sample_count, y, g.output_dispersion, g.output_lower_bounds, g.output_central_point)

        # update bounds
        g.input_lower_bounds = g.input_central_point - self.interval(g.input_dispersion)
        g.input_upper_bounds = g.input_central_point + self.interval(g.input_dispersion)

        g.output_lower_bounds = g.output_central_point - self.interval(g.output_dispersion)
        g.output_upper_bounds = g.output_central_point + self.interval(g.output_dispersion)

        # add sample to sample list
        sample = Sample(self.height, x, y)
        g.samples.append(sample)

        # update coefficients
        # I have no idea if this is even remotely correct
        # at the very least it doesn't crash
        XX = numpy.array([sample.input for sample in g.samples])
        YY = numpy.array([sample.output for sample in g.samples])

        g.coefficients = numpy.linalg.lstsq(XX, YY, rcond=None)[0]

    def get_closest_granule(self, x: numpy.ndarray) -> Granule | None:
        """
        Returns the closest FITTING granule to the sample
        """
        # check fitting granules
        fitting_granules = []

        for granule in self.granules:
            if(numpy.all(granule.input_lower_bounds <= x) and numpy.all(granule.input_upper_bounds >= x)):
                fitting_granules.append(granule)

        # get closest fitting granule
        minimum_distance = numpy.inf
        closest_granule = None

        for granule in fitting_granules:
            distance = numpy.linalg.norm(granule.input_central_point - x)
            if(distance < minimum_distance):
                minimum_distance = distance
                closest_granule = granule

        if(closest_granule is None):
            return None
        else:
            return closest_granule

    def train(self, x, y) -> None:
        """
        Unlike normal scikit-learn libraries, eogslib train accepts a single n-dimensional sample of data
        uses it to update its internal structures and simultaneously provide a prediction for the next value

        train is supposed to be used in a loop, where the next value is fed into the system constantly

        returns a tuple with three values, a numerical prediction and a granular prediction with a lower
        and upper bounds

        :param arraylike x: single sample (n dimensions) (1,n)
        :param arraylike y: single output (m dimensions) (1,m)
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

        granule = self.get_closest_granule(x)

        if granule == None:
            # if no granule fits, create new one
            input_dispersion = numpy.full(len(x), self.initial_dispersion)
            output_dispersion = numpy.full(len(y), self.initial_dispersion)

            sample = [Sample(self.height, x, y)]

            granule = self.create_granule(sample, x, input_dispersion, y, output_dispersion)
            
            self.granules.append(granule)
        else:
            # if granule fits, update it
            self.update_granule(granule, x, y)

        # delete inactive granules
        for granule in self.granules:
            # forget old samples (samples that are older than the window)
            granule.samples = [sample for sample in granule.samples if self.height - sample.height < self.window]

            # delete inactive granules
            if len(granule.samples) == 0:
                self.granules.remove(granule)

        # merge granules
        if(self.minimum_distance > 0.0):
            # this code is enormously slow and above ~200 granules it becomes too slow to be practical
            # TODO: put granules in k-d tree to speed up distance calculation

            granules_to_merge = set()
            new_granules = set()
            for g1, g2 in itertools.combinations(self.granules, 2):
                # if either of granules have already been merged, skip
                if g1 in granules_to_merge or g2 in granules_to_merge:
                    continue

                # calculate disntance between granules
                distance = numpy.linalg.norm(g1.input_central_point - g2.input_central_point)

                if distance < self.minimum_distance:
                    # combine the granules
                    input_central_point = g1.input_central_point * len(g1.samples) + g2.input_central_point * len(g2.samples) / (len(g1.samples) + len(g2.samples))
                    input_dispersion = numpy.max([g1.input_dispersion, g2.input_dispersion], axis=0)

                    output_central_point = g1.output_central_point * len(g1.samples) + g2.output_central_point * len(g2.samples) / (len(g1.samples) + len(g2.samples))
                    output_dispersion = numpy.max([g1.output_dispersion, g2.output_dispersion], axis=0)

                    samples = g1.samples + g2.samples

                    new_granule = self.create_granule(samples, input_central_point, input_dispersion, output_central_point, output_dispersion)

                    granules_to_merge.add(g1)
                    granules_to_merge.add(g2)
                    new_granules.add(new_granule)

            # remove merged granules
            for granule in granules_to_merge:
                self.granules.remove(granule)

            # add new granules
            for granule in new_granules:
                self.granules.append(granule)

        if self.mode == MODE_AUTOMATIC:
            # update parameters
            # TODO
            pass

        # update the height
        self.height += 1

    def train_many(self, x, y) -> None:
        """
        Accepts an array of t samples of n dimensional data
        with a second array of t results of m dimensional data

        :param arraylike x: samples (t,n)
        :param arraylike y: results (t,m)
        """
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)

        for sample, result in zip(x, y):
            self.train(sample, result)

    def fit(self, x, y) -> None:
        """
        Alias for train
        """
        self.train_many(x, y)

    def predict_scalar(self, x) -> numpy.ndarray:
        """
        Accepts a single (n,1) sample of data
        Returns a single (m,1) prediction for the sample

        check which granule fits
        if no granule fits fail
        if one granule fits, return the prediction
        if multiple granules fit, use closest one

        formulae
        y = a0 + (a1 * x1 + a2 * x2 + ... + an * xn)
        """
        x = numpy.atleast_1d(numpy.squeeze(x))

        if(self.data_width != len(x)):
            raise TypeError("Data width does not match previous data width")

        granule = self.get_closest_granule(x)

        if granule == None:
            raise Exception("No granule fits the sample")

        return granule.coefficients[0] + numpy.dot(x, granule.coefficients[1:])        
    
    def predict(self, x) -> numpy.ndarray:
        """
        Same as predict_scalar
        """
        return self.predict_scalar(x)

    def predict_interval(self, x) -> numpy.ndarray:
        """
        Accepts a single n-dimensional sample of data
        returns an (2,m) interval prediction for the sample
        first row is lower bound, second row is upper bound
        each column is a dimension of the output
        """

        x = numpy.atleast_1d(numpy.squeeze(x))

        if(self.data_width != len(x)):
            raise TypeError("Data width does not match previous data width")

        granule = self.get_closest_granule(x)

        if granule == None:
            raise Exception("No granule fits the sample")

        return numpy.array([granule.output_lower_bounds, granule.output_upper_bounds])
