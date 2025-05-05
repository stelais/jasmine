import numpy as np
from src.jasmine.caustics.caustic_creator_cls import Caustic


class SourceTrajectoryCausticCrossing:
    """
    This class creates the source trajectory of the lensed source star if crossing a caustic.
    y = ax + b

    """

    def __init__(self, separation_s, mass_ratio_q,
                 inclination_of_source_trajectory,
                 index_for_caustic_data_points,
                 caustic_number_of_data_points=100,
                 source_trajectory_number_of_data_points=100):
        self.separation_s = separation_s
        self.mass_ratio_q = mass_ratio_q
        self.inclination_of_source_trajectory = inclination_of_source_trajectory
        self.index_for_caustic_data_points = index_for_caustic_data_points
        self.caustic_number_of_data_points = caustic_number_of_data_points
        self.source_trajectory_number_of_data_points = source_trajectory_number_of_data_points
        self.caustic = Caustic(separation_s=separation_s,
                               mass_ratio_q=mass_ratio_q,
                               number_of_data_points=caustic_number_of_data_points)
        self.starting_point_x = self.caustic.horizontal[self.index_for_caustic_data_points]
        self.starting_point_y = self.caustic.vertical[self.index_for_caustic_data_points]
        self.offset_of_source_trajectory = offset_for_line_through_point(self.starting_point_x,
                                                                         self.starting_point_y,
                                                                         self.inclination_of_source_trajectory)
        self.source_x = np.linspace(-3, 3, self.source_trajectory_number_of_data_points)
        self.source_y = self.inclination_of_source_trajectory * self.source_x + self.offset_of_source_trajectory

    # caustic.define_topology()


def offset_for_line_through_point(x0, y0, a):
    """
    Returns a function of a line y = ax + b that passes through the point (x1, y1).

    Parameters:
    x0 (float): The x-coordinate of the starting/crossing point.
    y0 (float): The y-coordinate of the starting/crossing point.
    a (float): The slope of the line.
    Returns:
    b (float): offset b
    """
    # Calculate the y-intercept b using the point (x1, y1)
    b = y0 - a * x0
    return b


if __name__ == '__main__':
    source_trajectory = SourceTrajectoryCausticCrossing(separation_s=0.8,
                                                        mass_ratio_q=0.3,
                                                        inclination_of_source_trajectory=0.5,
                                                        index_for_caustic_data_points=50,
                                                        caustic_number_of_data_points=100,
                                                        source_trajectory_number_of_data_points=100)
    import matplotlib.pyplot as plt
    plt.scatter(source_trajectory.caustic.horizontal, source_trajectory.caustic.vertical, s=0.1)
    plt.scatter(source_trajectory.starting_point_x, source_trajectory.starting_point_y, color='red')
    plt.plot(source_trajectory.source_x, source_trajectory.source_y, c='black')
    plt.show()