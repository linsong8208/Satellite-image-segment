"""
http://blog.csdn.net/pipisorry/article/details/53156836
http://scikit-learn.org/stable/modules/neighbors.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree
"""
import numpy as np
from time import clock
from random import randint
from sklearn.neighbors import KDTree, BallTree


class RandomData(object):

    # 产生一个k维随机向量，每维分量值在0~1之间
    @staticmethod
    def random_point(dim):
        return [randint(0, 50) for _ in range(dim)]

    # 产生n个k维随机向量
    def random_points(self, dim, n):
        return [self.random_point(dim) for _ in range(n)]

    pass


# 使用kd-tree
class CalDistanceByKDTree(object):

    def __init__(self, data, k):
        self.k = k + 1
        self.data = data
        self.data_length = len(self.data)
        self.kd_tree = KDTree(self.data, leaf_size=self.data_length//2)  # 构建kd树
        pass

    def __call__(self):
        # 计算距离
        kd_dist, kd_ind = self._query_all()

        # 返回的数据， 聚集度
        org_point = self.data
        k_dist = np.sum(kd_dist, axis=1) / (self.k - 1)
        k_point = [[self.data[ind] for ind in inds]for inds in kd_ind]
        return org_point, k_dist, k_point

    def _query_all(self):
        kd_dist, kd_ind = self.kd_tree.query(self.data, k=self.k)
        kd_dist, kd_ind = kd_dist[:, 1:], kd_ind[:, 1:]
        return kd_dist, kd_ind

    def query_one(self, point):
        kd_dist, kd_ind = self.kd_tree.query([point], k=self.k)
        return kd_dist[0], kd_ind[0]

    def test_query_one(self):
        _time_1 = clock()
        for index in range(10000):
            point_now = RandomData.random_point(dim=2)
            kd_dist, kd_ind = self.query_one(point_now)
            pass
        _time_2 = clock()
        print(_time_1 - _time_2)
        pass

    def test_query_all(self):
        _time_1 = clock()
        self._query_all()
        _time_2 = clock()
        print(_time_1 - _time_2)

    @staticmethod
    def main_test():
        data_n = RandomData().random_points(dim=2, n=10000)
        time_1 = clock()
        cal_dis = CalDistanceByKDTree(data=data_n, k=5)
        time_2 = clock()
        print(time_2 - time_1)

        cal_dis.test_query_one()

        cal_dis.test_query_all()
        pass

    pass

if __name__ == '__main__':
    data_n = RandomData().random_points(dim=2, n=100)
    org_point, k_dist, k_point = CalDistanceByKDTree(data=data_n, k=5)()
    pass
