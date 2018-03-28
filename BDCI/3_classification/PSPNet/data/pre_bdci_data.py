import os
import time
import numpy as np
from PIL import Image


class Tools(object):

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_info(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    pass


class OneImage(object):

    def __init__(self, image_file, label_file, result_image_path, result_label_path):
        self.image_file = image_file
        self.label_file = label_file
        self.result_image_path = result_image_path
        self.result_label_path = result_label_path
        self.image_data = np.copy(np.asarray(Image.open(self.image_file)))
        self.label_data = np.copy(np.asarray(Image.open(self.label_file)))
        pass

    def fenge(self, index, stripe, image_size):
        x = len(self.image_data)
        y = len(self.image_data[0])
        image_list = []
        for i in range(0, x - image_size, stripe):
            for j in range(0, y - image_size, stripe):
                now_image = self.image_data[i: i + image_size, j: j + image_size, :]
                now_label = self.label_data[i: i + image_size, j: j + image_size]
                result_file_name = "{}_{}_{}.png".format(index, i, j, stripe)
                Image.fromarray(now_image).convert("RGB").save(os.path.join(self.result_image_path, result_file_name))
                Image.fromarray(now_label).convert("L").save(os.path.join(self.result_label_path, result_file_name))
                image_list.append(result_file_name)
            pass
        return image_list

    pass


class DivideImage(object):

    def __init__(self, image_file, result_image_path):
        self.image_file = image_file
        self.result_image_path = result_image_path
        self.image_data = np.copy(np.asarray(Image.open(self.image_file)))
        pass

    def fenge_by_number(self, index, divide_number):
        x = len(self.image_data)
        y = len(self.image_data[0])
        x_stripe = x // divide_number
        y_stripe = y // divide_number
        for i in range(0, x - x_stripe + 1, x_stripe):
            for j in range(0, y - y_stripe + 1, y_stripe):
                now_image = self.image_data[i: i + x_stripe, j: j + y_stripe, :]
                result_file_name = "{}_{}_{}_{}_{}.png".format(index, i, j, x_stripe, y_stripe)
                Image.fromarray(now_image).convert("RGB").save(os.path.join(self.result_image_path, result_file_name))
            pass
        pass

    def fenge_by_size(self, index, divide_size):
        x = len(self.image_data)
        y = len(self.image_data[0])
        for i in range(0, x - divide_size, divide_size):
            for j in range(0, y - divide_size, divide_size):
                now_image = self.image_data[i: i + divide_size, j: j + divide_size, :]
                result_file_name = "{}_{}_{}_{}.png".format(index, i, j, divide_size)
                Image.fromarray(now_image).convert("RGB").save(os.path.join(self.result_image_path, result_file_name))
            pass
        pass

    pass


class BDCIData(object):

    def __init__(self, image_files, label_files, result_image_path, result_label_path, result_list_txt_file):
        self.image_files = image_files
        self.label_files = label_files
        self.result_image_path = result_image_path
        self.result_label_path = result_label_path
        self.result_list_txt_file = result_list_txt_file

        self.image_list = []
        pass

    def run(self, stripe, image_size):
        self.to_fenge(stripe, image_size)
        self.to_txt()
        pass

    def to_txt(self):
        with open(self.result_list_txt_file, "w") as f:
            for image_file in self.image_list:
                f.write("{} {}\n".format(os.path.join(self.result_image_path, image_file),
                                         os.path.join(self.result_label_path, image_file)))
            pass
        pass

    def to_fenge(self, stripe, image_size):

        for index, image_file in enumerate(self.image_files):
            one_image = OneImage(image_file=image_file, label_file=self.label_files[index],
                                 result_image_path=self.result_image_path, result_label_path=self.result_label_path)
            self.image_list.extend(one_image.fenge(index, stripe, image_size))
            Tools.print_info("image {} ok".format(index))
        pass

    pass


def main_bdci():
    all_image_file = ["bdci_semi/training/train1.png",
                      "bdci_semi/training/train2.png",
                      "bdci_semi/training/train3.png"]
    all_label_file = ["bdci_semi/training/train1_labels_8bits.png",
                      "bdci_semi/training/train2_labels_8bits.png",
                      "bdci_semi/training/train3_labels_8bits.png"]
    BDCIData(image_files=all_image_file, label_files=all_label_file,
             result_image_path=Tools.new_dir("bdci/train/image"),
             result_label_path=Tools.new_dir("bdci/train/label"),
             result_list_txt_file="bdci/train_list.txt").run(stripe=100, image_size=713)

    pass


def divide_image_by_number():
    DivideImage(image_file="bdci_semi/testing/testing1.png",
                result_image_path=Tools.new_dir("bdci/vali")).fenge_by_number("testing1", 2)
    pass


def divide_image_by_size():
    DivideImage(image_file="bdci_semi/testing/testing1.png",
                result_image_path=Tools.new_dir("bdci/vali")).fenge_by_size("testing1", 713)
    pass


if __name__ == '__main__':
    divide_image_by_size()
