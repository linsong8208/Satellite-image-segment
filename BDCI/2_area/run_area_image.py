import numpy as np
from PIL import Image
from glob import glob
import os
import multiprocessing
import shutil


class SketchToArea:
    def __init__(self, sketch_path='image_sar/temp', image_x=0, image_y=0, is_mul_processing=True,
                 area_exe='SARsegment_PSM_semanticClusster_YJL_1107_7_5_3.exe'):
        # area 程序
        self.area_exe = area_exe
        # 当前根路径
        self.root_path = self._new_dir(sketch_path)
        # 图片的名称
        self.image_names = [image.replace("\\", "/") for image in glob("{}/{}".format(sketch_path, "*.bmp"))]
        # 是否使用多进程
        self.is_mul_processing = is_mul_processing
        # 原始图片的大小
        self.image_x = image_x
        self.image_y = image_y
        # 结果
        self.result_file_name = self.root_path + "_area.bmp"
        pass

    def run(self):
        if self.is_mul_processing:
            self.area_mul_processing_with_pool()
        else:
            self.area_no_process()
        # 复制生成的结果到一起
        self.prepare_network_image()
        # merge
        self.merge_area()
        pass

    # area one
    def _area_single(self, image_name, info=""):
        print("{} {}".format(info, image_name))
        os.system(self.area_exe + ' ' + image_name)
        print("{} {}".format(info, image_name))
        pass

    # not use process
    def area_no_process(self):
        for image_name in self.image_names:
            self._area_single(image_name)
        pass

    # use pool
    def _area_batch_with_pool(self, which):
        self._area_single(self.image_names[which], info=which)
        pass

    # pool
    def area_mul_processing_with_pool(self):
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpu_count)
        for index in range(len(self.image_names)):
            pool.apply_async(func=self._area_batch_with_pool, args=(index,))
        pool.close()
        pool.join()
        pass

    # copy the initial_labelPixel.bmp  to together
    def prepare_network_image(self):
        area_path = self._new_dir(self.root_path + "_area")
        for image_name in self.image_names:
            now_path = os.path.join(os.path.splitext(image_name)[0], "k5_delta3_D10_0120_edge", "3_smoothPixelRegion")
            now_bmp = os.path.join(now_path, "initial_labelPixel.bmp")
            new_bmp_name = os.path.splitext(os.path.split(image_name)[1])[0] + "_initial_labelPixel.bmp"
            try:
                shutil.copy(now_bmp, os.path.join(area_path, new_bmp_name))
            except FileNotFoundError:
                print("----------------------------------")
                print("No such file or directory: {}".format(now_bmp))
                print("----------------------------------")
                pass
        pass

    # 合并所有的区域图为一张大图
    def merge_area(self):
        area_data = np.zeros(shape=[self.image_x, self.image_y], dtype=np.uint8)
        # area_data = np.zeros(shape=[1506, 3543], dtype=np.uint8)
        area_path = self._new_dir(self.root_path + "_area")
        self.image_names = glob(os.path.join(area_path, "*_initial_labelPixel.bmp"))
        for image_name in self.image_names:
            print(image_name)
            now_data = np.asarray(Image.open(image_name), dtype=np.uint8)
            name = os.path.splitext(os.path.split(image_name)[1])[0]
            poss = [position for position in name.split("_")][0:4]
            poss = [int(position) for position in poss]
            area_data[poss[0]: poss[2], poss[1]: poss[3]] = now_data
            pass
        self._save_data_to_image(area_data, self.result_file_name, "L")
        pass

    @staticmethod
    def _new_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @staticmethod
    def _save_data_to_image(data, file_name, convert="RGB"):
        Image.fromarray(data).convert(convert).save(file_name)
        pass

    pass


if __name__ == '__main__':

    images = ['image_sar/1_sketch']

    for image in images:
        sketch_to_area = SketchToArea(sketch_path=image, image_x=5190, image_y=5204)
        sketch_to_area.run()

    pass
