"""
    existed: labels,positions

    1.padding
    2.flatten

    result: csv
"""
import csv
from Param import Param
import pickle
from PIL import Image
import numpy as np


class FinalResult:

    @staticmethod
    def to_csv(datas, which, name):
        # save
        with open("../dist/{}/result/{}.csv".format(name, which), 'w') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(["ID", "ret"])
            for col in range(len(datas[0])):
                for row in range(len(datas)):
                    spamwriter.writerow([which, int(datas[row, col] + 1)])
            pass
        pass

    @staticmethod
    def load_data(data_file_name):
        # data
        with open(data_file_name, 'rb') as f:
            datas = pickle.load(f)
            labels = datas["labels"]
            positions = datas['positions']

        return labels, positions
        pass

    @staticmethod
    def padding_data(labels, positions, stripe_size, image_x, image_y):
        new_labels = np.zeros((image_x, image_y))
        # fill
        for index, label in enumerate(labels):
            now_x, now_y = positions[index]
            new_labels[now_x - stripe_size // 2: now_x + stripe_size // 2 + 1,
            now_y - stripe_size // 2: now_y + stripe_size // 2 + 1] = label
        # padding
        right_padding_size = image_y % stripe_size
        bottom_padding_size = image_x % stripe_size
        if right_padding_size != 0:
            new_labels[:, -right_padding_size:] = new_labels[:, - 2 * right_padding_size: -right_padding_size]
        if bottom_padding_size != 0:
            new_labels[-bottom_padding_size:, :] = new_labels[- 2 * bottom_padding_size: - bottom_padding_size, :]
        return new_labels

    @staticmethod
    def write_color(labels, image_file_name):
        colors = labels * 85
        Image.fromarray(colors).convert('L').save(image_file_name)
    pass


if __name__ == '__main__':

    name = Param.name
    stripe = Param.test_stripe

    for index in range(3):
        labels, positions = FinalResult.load_data("../dist/{}/result/result_{}.pkl".format(name, index))
        new_labels = FinalResult.padding_data(labels, positions, stripe, 5190, 5204)
        FinalResult.write_color(new_labels, "../dist/{}/result/result_{}.bmp".format(name, index + 1))
        FinalResult.to_csv(new_labels, index + 1, name)

    pass
