import shutil
import os
from glob import glob
import math
import numpy as np


# copy the initial_labelPixel.bmp  to together
def copy_hist_together(src_path, dest_path):
    image_names = glob(os.path.join(src_path, "*.bmp"))
    for image_name in image_names:
        now_path = os.path.join(os.path.splitext(image_name)[0], "k5_delta3_D10_0120_edge", "1_segmentLabel")
        now_bmp = os.path.join(now_path, "segmen_7meansDistance_Hist_smooth.bmp")
        new_bmp_name = os.path.splitext(os.path.split(image_name)[1])[0] + "_segmen_7meansDistance_Hist_smooth.bmp"
        try:
            shutil.copy(now_bmp, os.path.join(dest_path, new_bmp_name))
        except FileNotFoundError:
            print("----------------------------------")
            print("No such file or directory: {}".format(now_bmp))
            print("----------------------------------")
            pass
    pass


def agreetation(arr, k):
    sum = 0
    for i in range(k):
        sum = arr[i]
    return sum // k


def statistic_hist(lines):

    # 1. get the [mid_point, len, director]
    new_lines = []
    for index, line in enumerate(lines):
        new_lines[index] = {}
        len = len(line)
        new_lines[index]["point"] = line[len // 2]
        new_lines[index]["len"] = len
        new_lines[index]["director"] = ????

    # 2. get 2 means mat for every line
    two_mean = []
    line_number = len(new_lines)
    for row in range(line_number):
        src_x, src_y = line[row]["point"]
        for col in range(line_number):
            dest_x, dest_y = line[col]['point']
            distance = math.sqrt(math.pow(src_x - dest_x, 2) + math.pow(src_y - dest_y, 2))
            two_mean[row][col] = distance

    # 3. get  k-agreetation for every line
    agreetations = []
    for row in range(line_number):
        agreetation = agreetation(sort(two_mean[row]), k)
        agreetations.append(agreetation)

    # 4. get hist
    hists = {}
    for row in range(line_number):
        hists[agreetations[row]] += 1

    # 5 判断 ： 聚集性直方图中存在主统计波
    #       5.1  标记聚集性统计数大于等于峰值1/4的聚集性对应的所有线段为候选地物线段基元
    #       5.2  get agreetation-best [agreetation-peak - delt, agreetation-peak + delt]
    candidate_base_line = []
    peak = np.max(hists)
    agreetation_best = (peak - delt, peak + delt)
    if peak > 0:
        for index, agreetation in enumerate(agreetations):
            if agreetation in range(peak // 4, peak + 1):
                candidate_base_line.append(new_lines[index])


    # 6.依据距离在最聚集性区间 A best greetation 内的邻域线段的分布，判定线段的分布结构
        #   所有与线段 i 的距离在最优聚集性区间内的线段 j 称为线段 i 的邻域线段。
        #   依据邻域线段在线段附近的分布情况，确定线段的分布结构类型。
    for row in range(line_number):


    # 7. 对于每一条候选地物基元， 双侧聚集标记为 地物线段基元
    base_line = []
    for line in candidate_base_line:
        if bi_agreetation(line):
            base_line.append(line)

    # 8. 将其它线段标记为 非地物线段基元
    not_base_line =[]
    for line in new_lines:
        if line not in base_line:
            not_base_line.append(line)

    pass

if __name__ == '__main__':

    copy_hist_together("../dist/1.0_1.7/test_3/2_area", "./together")
