import numpy as np
from moana.lens import ResonantCaustic as caustic_calculator


class Caustic:
    """
    This class is a wrapper for the caustic_calculator function from the moana package. It takes in
    the separation_s and mass_ratio_q and calculate the upper and lower half of the caustic,
    or as an alternative the horizontal and vertical components of the caustic.
    """
    def __init__(self, separation_s, mass_ratio_q, number_of_data_points=10000):
        self.separation_s = separation_s
        self.mass_ratio_q = mass_ratio_q
        caustic_data_points = caustic_calculator(sep=separation_s, q=mass_ratio_q)
        self.number_of_data_points = number_of_data_points
        caustic_data_points._sample(self.number_of_data_points)
        self.half_caustic = caustic_data_points.full['zeta'].values
        self.upper_half = np.real(self.half_caustic), np.imag(self.half_caustic)
        self.lower_half = np.real(self.half_caustic), -np.imag(self.half_caustic)

        self.horizontal = np.concatenate((np.real(self.half_caustic), np.real(self.half_caustic)))
        self.vertical = np.concatenate((np.imag(self.half_caustic), -np.imag(self.half_caustic)))


if __name__ == '__main__':
    caustic_object = Caustic(separation_s=0.8, mass_ratio_q=0.03, number_of_data_points=100)
    import matplotlib.pyplot as plt
    # plt.scatter(caustic_object.upper_half[0], caustic_object.upper_half[1], s=0.1)
    # plt.scatter(caustic_object.lower_half[0], caustic_object.lower_half[1], s=0.1)
    plt.scatter(caustic_object.horizontal, caustic_object.vertical)
    plt.show()
