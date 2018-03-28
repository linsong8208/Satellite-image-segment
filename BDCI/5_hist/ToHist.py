"""
dis data
"""
from time import clock
import numpy as np
import matplotlib.pyplot as plt
from PreData import PreData
from KDTree import CalDistanceByKDTree


class Hist(object):

    @staticmethod
    def to_hist(dis_data):
        plt.figure("hist")
        plt.hist(dis_data, bins=256, normed=1, edgecolor='None', facecolor='red')
        plt.show()
        pass

    @staticmethod
    def draw_hist(data_list, title, bins, x_label, y_label, x_min, x_max, y_min, y_max):
        plt.hist(data_list, bins)
        plt.xlabel(x_label)
        plt.xlim(x_min, x_max)
        plt.ylabel(y_label)
        plt.ylim(y_min, y_max)
        plt.title(title)
        plt.show()

    pass

if __name__ == '__main__':
    time_1 = clock()
    pre_data = PreData()
    data_curve = pre_data.read_curve("./data/1_curve.txt")
    data = [data_one["mid"] for data_one in data_curve]
    time_2 = clock()
    print(time_2 - time_1)

    time_1 = clock()
    org_point, k_dist, k_point = CalDistanceByKDTree(data=data, k=5)()
    time_2 = clock()
    print(time_2 - time_1)

    time_1 = clock()
    # Hist.to_hist(k_dist)
    Hist.draw_hist(k_dist, r'Hist', int(np.max(k_dist)), r'agreetation', r'count', 0.0, np.max(k_dist), 0.0, 1700)
    time_2 = clock()
    print(time_2 - time_1)
