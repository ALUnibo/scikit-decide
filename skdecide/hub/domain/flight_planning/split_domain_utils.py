import numpy as np
from math import floor, ceil
from pygeodesy.ellipsoidalVincenty import LatLon


def divide_graph(env):
    """
    Divide the graph in 4 parts
    """
    # TODO:  REPLACE TEMPORARY IMPLEMENTATION!!
    network = env.get_network()
    middle_point = len(network[0]) // 2
    intermediate_points = [[0, middle_point]] + [[10, middle_point], [20, middle_point],
                                                 [30, middle_point], [40, middle_point]]
    fwd_points = 11
    lat_points = 5

    network = [network[p][middle_point - lat_points//2 : middle_point + lat_points//2 + 1] for p in range(len(network))]
    for midpoint in intermediate_points:
        for c in range(middle_point - fwd_points//2, middle_point + fwd_points//2 + 1):
            network[midpoint[0]][c] = network[midpoint[0]][midpoint[1]]

    return network
