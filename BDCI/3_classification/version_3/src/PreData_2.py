# encoding: utf-8
import os
import time
import glob
import pickle
import numpy as np
from Param import Param
from PIL import Image
from collections import Counter


class OneImage:

    def __init__(self, img_name, label_img, crop_size, stripe, save_path, ratio=0):
        self.img_name = img_name
        self.label_img = label_img
        self.crop_size = crop_size
        self.stripe = stripe
        self.ratio = ratio

        self.images = Image.open(self.img_name)
        self.labels = np.asarray(Image.open(self.label_img).convert("L"))

        self.w, self.h = self.images.size

        self.save_path = self.new_dir(save_path)
        pass

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # judge
    def stat_label(self, y, x, crop_size, ratio):
        now_label = self.labels[x][y]
        # not stat
        if ratio == 0:
            return now_label

        count = 0
        for i in range(x - crop_size // 2, x + crop_size // 2):
            for j in range(y - crop_size // 2, y + crop_size // 2):
                if self.labels[i][j] == now_label:
                    count += 1
            pass

        if count / (crop_size * crop_size * 1.0) > float(ratio):
            return now_label
        else:
            return -1
        pass

    def cut(self, need_label):
        counter = Counter()
        for x in range(self.crop_size // 2, self.w - self.crop_size // 2, self.stripe):
            for y in range(self.crop_size // 2, self.h - self.crop_size // 2, self.stripe):
                now_label = self.labels[y][x]
                if now_label in need_label:
                    region = (x - self.crop_size // 2, y - self.crop_size // 2,
                              x + self.crop_size // 2 + 1, y + self.crop_size // 2 + 1)
                    crop_img = self.images.crop(region)

                    label = self.stat_label(x, y, self.crop_size, self.ratio)

                    if label != -1 and label != 0:
                        counter[label] += 1
                        crop_img_name = "{}-{}.bmp".format(np.random.randint(0, 1000000), label)
                        crop_img.save(os.path.join(self.save_path, crop_img_name))
                        pass
                    pass
            pass

        for key in counter.keys():
            self.print_info("{}:{}".format(key, counter[key]))

        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    pass


class PreData:

    def __init__(self, images, labels, stripe, crop_size, ratio, number, result_pkl, need_label):
        self.need_label = need_label

        self.print_info("begin")
        if os.path.exists(result_pkl):
            self.print_info("{} existed...".format(result_pkl))
        else:
            self.result_image_path = self.new_dir(result_pkl.split(".pkl")[0])
            # 切图
            for index, image in enumerate(images):
                self.print_info("begin to cut {}".format(image))
                one_image = OneImage(image, labels[index], crop_size=crop_size,
                                     stripe=stripe, save_path=self.result_image_path, ratio=ratio)
                one_image.cut(need_label=self.need_label)
                pass
            # 写入pkl
            self.print_info("begin to pkl")
            self.to_pkl(result_pkl, number=number)
            self.print_info("delete images")
            self.del_images(self.result_image_path)
            pass

        self.print_info("end")
        pass

    # 写到pkl里
    def to_pkl(self, result_pkl, number):
        data_images = os.listdir(self.result_image_path)

        datas = []
        labels = []
        counter = Counter()
        for data_img in data_images:
            label = int(data_img.split("-")[1].split(".")[0])
            if counter[label] >= number:
                continue
            counter[label] += 1
            label = self.one_hot(label)
            now_data_img_path = os.path.join(self.result_image_path, data_img)
            now_data = np.copy(np.asarray(Image.open(now_data_img_path)))

            # 中心化
            # now_data = now_data - np.mean(now_data)
            # 正则化
            # now_data = now_data / np.std(now_data)
            # 归一化
            now_data = now_data / 255.0

            datas.append(now_data)
            labels.append(label)
            pass

        for key in counter.keys():
            self.print_info("{}:{}".format(key, counter[key]))

        with open(result_pkl, "wb") as f:
            pickle.dump({"X": datas, "Y": labels}, f)
        pass

    def one_hot(self, label):
        if len(self.need_label) == 4:
            if label == 1:
                return [1, 0, 0, 0]
            elif label == 2:
                return [0, 1, 0, 0]
            elif label == 3:
                return [0, 0, 1, 0]
            elif label == 4:
                return [0, 0, 0, 1]
        elif len(self.need_label) == 2:
            if label == self.need_label[0]:
                return [1, 0]
            elif label == self.need_label[1]:
                return [0, 1]
        pass

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    @staticmethod
    def del_images(images_path):
        images = glob.glob(os.path.join(images_path, "*.bmp"))
        for image in images:
            os.remove(image)
        os.removedirs(images_path)
        pass
    pass

if __name__ == "__main__":

    name = Param.name
    image_size = Param.image_size
    PreData(images=["../data/CCF-training/1.png", "../data/CCF-training/2.png"],
            labels=["../data/CCF-training/1_class.png", "../data/CCF-training/2_class.png"],
            stripe=300, crop_size=image_size, ratio=0, number=5000, result_pkl="../dist/{}/data/train.pkl".format(name),
            need_label=[1, 2, 3, 4])
