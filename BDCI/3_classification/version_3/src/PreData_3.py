# encoding: utf-8
"""
    add channel
"""
import os
import time
import glob
import pickle
import numpy as np
from Param import Param
from PIL import Image
from collections import Counter


class OneImage:

    def __init__(self, img_name, label_img, sketch, area, crop_size, stripe, save_path, ratio=0):
        self.img_name = img_name
        self.label_img = label_img
        self.area = area
        self.sketch = sketch
        self.crop_size = crop_size
        self.stripe = stripe
        self.ratio = ratio

        self.image_data = np.copy(np.asarray(Image.open(self.img_name)))
        self.label_data = np.asarray(Image.open(self.label_img).convert("L"))
        self.sketch_data = np.copy(np.asarray(Image.open(self.sketch).convert("L")))
        self.area_data = np.copy(np.asarray(Image.open(self.area).convert("L")))

        self.image_x, self.image_y = len(self.image_data), len(self.image_data[0])

        self.image_data_after_add = np.zeros(shape=[self.image_x, self.image_y, 5], dtype=np.uint8)
        self.image_data_after_add[:, :, 0: 3] = self.image_data
        self.image_data_after_add[:, :, -2] = self.area_data
        self.image_data_after_add[:, :, -1] = self.sketch_data

        self.save_path = self.new_dir(save_path)
        pass

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # judge
    def stat_label(self, x, y, crop_size, ratio):
        now_label = self.label_data[x][y]
        # not stat
        if ratio == 0:
            return now_label

        count = 0
        for i in range(x - crop_size // 2, x + crop_size // 2):
            for j in range(y - crop_size // 2, y + crop_size // 2):
                if self.label_data[i][j] == now_label:
                    count += 1
            pass

        if count / (crop_size * crop_size * 1.0) > float(ratio):
            return now_label
        else:
            return -1
        pass

    def cut(self, need_label):
        counter = Counter()
        half_crop_size = self.crop_size // 2
        for x in range(half_crop_size, self.image_x - half_crop_size, self.stripe):
            for y in range(half_crop_size, self.image_y - half_crop_size, self.stripe):
                now_label = self.label_data[x][y]
                if now_label in need_label:
                    crop_img = self.image_data_after_add[x - half_crop_size: x + half_crop_size + 1,
                               y - half_crop_size: y + half_crop_size + 1, :]

                    label = self.stat_label(x, y, self.crop_size, self.ratio)

                    if label != -1 and label != 0:
                        counter[label] += 1
                        crop_img_name = "{}-{}.pkl".format(np.random.randint(0, 1000000), label)
                        self.save_pkl(os.path.join(self.save_path, crop_img_name), crop_img)
                        pass
                    pass
            pass

        for key in counter.keys():
            self.print_info("{}:{}".format(key, counter[key]))

        pass

    @staticmethod
    def save_pkl(image_path, image_data):
        with open(image_path, "wb") as f:
            pickle.dump({"data": image_data}, f)
            pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    pass


class PreData:

    def __init__(self, images, labels, sketchs, areas, stripe, crop_size, ratio, number, result_pkl, need_label):
        self.need_label = need_label

        self.print_info("begin")
        if os.path.exists(result_pkl):
            self.print_info("{} existed...".format(result_pkl))
        else:
            self.result_image_path = self.new_dir(result_pkl.split(".pkl")[0])
            # 切图
            for index, image in enumerate(images):
                self.print_info("begin to cut {}".format(image))
                one_image = OneImage(image, labels[index], sketchs[index], areas[index], crop_size=crop_size,
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
            now_data = self.read_pkl(now_data_img_path)

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
    def read_pkl(image_path):
        with open(image_path, "rb") as f:
            return pickle.load(f)["data"]

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
        images = glob.glob(os.path.join(images_path, "*.pkl"))
        for image in images:
            os.remove(image)
        os.removedirs(images_path)
        pass
    pass

if __name__ == "__main__":

    name = Param.name
    image_size = Param.image_size
    PreData(images=["../data/CCF-training/1.png",
                    "../data/CCF-training/2.png"],
            labels=["../data/CCF-training/1_class.png",
                    "../data/CCF-training/2_class.png"],
            sketchs=["../data/sketch_area/1.0_1.7/train_1_sketch.bmp",
                     "../data/sketch_area/1.0_1.7/train_2_sketch.bmp"],
            areas=["../data/sketch_area/1.0_1.7/train_1_sketch_area.bmp",
                   "../data/sketch_area/1.0_1.7/train_2_sketch_area.bmp"],
            stripe=20, crop_size=image_size, ratio=0, number=5000,
            result_pkl="../dist/{}/data/train.pkl".format(name),
            need_label=[1, 2, 3, 4])
