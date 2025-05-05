from dataclasses import dataclass
import numpy as np

from moana.lens import ResonantCaustic as CausticCalculator
from moana.lens import close_limit_2l, wide_limit_2l


@dataclass
class Caustic:
    """
    This class is a wrapper for the caustic_calculator function from the moana package. It takes in
    the separation_s and mass_ratio_q and calculate the upper and lower half of the caustic,
    or as an alternative the horizontal and vertical components of the caustic.
    """
    type_of_caustics: str
    separation_s: float
    mass_ratio_q: float
    number_of_data_points: int
    half_caustic: np.ndarray
    upper_half: tuple
    lower_half: tuple
    horizontal: np.ndarray
    vertical: np.ndarray

    def __init__(self, separation_s, mass_ratio_q, number_of_data_points=10000):
        self.type_of_caustics = None
        self.separation_s = separation_s
        self.mass_ratio_q = mass_ratio_q
        caustic_data_points = CausticCalculator(sep=separation_s, q=mass_ratio_q)
        self.number_of_data_points = number_of_data_points
        caustic_data_points._sample(self.number_of_data_points)
        self.half_caustic = caustic_data_points.full['zeta'].values
        self.upper_half = np.real(self.half_caustic), np.imag(self.half_caustic)
        self.lower_half = np.real(self.half_caustic), -np.imag(self.half_caustic)

        self.horizontal = np.concatenate((np.real(self.half_caustic), np.real(self.half_caustic)))
        self.vertical = np.concatenate((np.imag(self.half_caustic), -np.imag(self.half_caustic)))

    def define_topology(self):
        """
        Check the topology type of the binary lens system based on moana
        :return:
        """
        if self.separation_s < close_limit_2l(self.mass_ratio_q):
            self.type_of_caustics = 'close'
        elif self.separation_s <= wide_limit_2l(self.mass_ratio_q):
            self.type_of_caustics = 'resonant'
        elif self.separation_s > wide_limit_2l(self.mass_ratio_q):
            self.type_of_caustics = 'wide'
        else:
            self.type_of_caustics = 'Unknown'
        return self.type_of_caustics


if __name__ == '__main__':
    caustic_object = Caustic(separation_s=0.8, mass_ratio_q=0.3, number_of_data_points=100)
    import matplotlib.pyplot as plt

    # plt.scatter(caustic_object.upper_half[0], caustic_object.upper_half[1], s=0.1)
    # plt.scatter(caustic_object.lower_half[0], caustic_object.lower_half[1], s=0.1)
    plt.scatter(caustic_object.horizontal, caustic_object.vertical)
    plt.show()
    type_of_caustics = caustic_object.define_topology()
    print(type_of_caustics)
