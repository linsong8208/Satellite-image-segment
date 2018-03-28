import os
import ctypes
import shutil
import numpy as np
from PIL import Image
import multiprocessing
from glob import glob


class SketchLargeImageToSar:

    def __init__(self, image_file, lf, hf, max_size=800, is_mul_processing=True,
                 sketch_dll="Sketch_SAR_WJ_DLL_ALISURE_5.dll"):
        self.lf = lf
        self.hf = hf
        self.image_file = image_file
        # sketch程序
        self.sketch_dll = sketch_dll
        # 切分的最大大小
        self.max_size = max_size
        # 是否使用多进程
        self.is_mul_processing = is_mul_processing
        # 原始数据
        self.image_data = np.copy(np.asarray(Image.open(image_file).convert("L")))
        # 原始图片的大小
        self.image_x = len(self.image_data)
        self.image_y = len(self.image_data[0])
        # 切分的个数
        self.number_x = int(np.ceil(self.image_x / self.max_size))
        self.number_y = int(np.ceil(self.image_y / self.max_size))
        # 切分后每个的大小
        self.pre_size_x = self.image_x // self.number_x
        self.pre_size_y = self.image_y // self.number_y
        # 当前根路径
        self.root_path = self._new_dir(os.path.splitext(image_file)[0])
        # 切分后的名称
        self.image_names = []
        # 结果
        self.result_file_name = self.root_path + "_sketch.bmp"
        pass

    def run(self):
        # 切分
        self._divide_image()
        # 跑素描图
        if self.is_mul_processing:
            self.sketch_mul_processing_with_pool()
        else:
            self.sketch_no_process()
        # 准备跑区域图的数据
        self.prepare_area_image()
        # 将切块后生成的各个素描图拷贝到一个文件夹下
        self.copy_sketch_to_together()
        # 合并切块跑出来的素描图为一张大图
        self.merge_sketch()
        pass

    # divide
    def _divide_image(self):
        for x in range(self.number_x):
            for y in range(self.number_y):
                x_start = self.pre_size_x * x
                y_start = self.pre_size_y * y
                x_end = self.image_x if x == self.number_x - 1 else self.pre_size_x * (x + 1)
                y_end = self.image_y if y == self.number_y else self.pre_size_y * (y + 1)
                now_data = self.image_data[x_start: x_end, y_start: y_end]
                file_name = os.path.join(self.root_path, "{}_{}_{}_{}.bmp".format(x_start, y_start, x_end, y_end))
                self.image_names.append(file_name)
                self._save_data_to_image(now_data, file_name, convert="L")
                pass
            pass
        pass

    # sketch one
    def _sketch_single(self, image_name, info=""):
        try:
            library = ctypes.CDLL(self.sketch_dll)
            char_point_para = ctypes.c_char_p(bytes(image_name, encoding="utf8"))
            float_para_1 = ctypes.c_float(4.0)
            float_para_2 = ctypes.c_float(5.0)

            float_para_3 = ctypes.c_float(self.hf)
            float_para_4 = ctypes.c_float(self.lf)
            library.sketch(char_point_para, float_para_1, float_para_2, float_para_3, float_para_4)
        except Exception:
            print("请尝试换成为win32版本")
            pass
        print("{} {}".format(info, image_name))
        pass

    # not use process
    def sketch_no_process(self):
        for image_name in self.image_names:
            self._sketch_single(image_name)
        pass

    # use process
    def _sketch_batch_with_process(self, start, end, which):
        for index in range(start, end):
            self._sketch_single(self.image_names[index], info=which)
        pass

    def sketch_mul_processing_with_process(self):
        cpu_count = multiprocessing.cpu_count() - 1 + 1
        pre_cpu_number = int(np.ceil(len(self.image_names) // cpu_count)) == 0
        processes = []
        for index in range(cpu_count):
            start = index * pre_cpu_number
            end = (index + 1) * pre_cpu_number if index < cpu_count - 1 else len(self.image_names)
            process = multiprocessing.Process(target=self._sketch_batch_with_process, args=(start, end, index))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        pass

    # use pool
    def _sketch_batch_with_pool(self, which):
        self._sketch_single(self.image_names[which], info=which)
        pass

    def sketch_mul_processing_with_pool(self):
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpu_count)
        for index in range(len(self.image_names)):
            pool.apply_async(func=self._sketch_batch_with_pool, args=(index,))
        pool.close()
        pool.join()
        pass

    # 准备跑区域图的数据
    def prepare_area_image(self):
        sketch_path = self._new_dir(self.root_path + "_sketch")
        for image_name in self.image_names:
            now_path = os.path.join(os.path.splitext(image_name)[0] + "_Sketch", "pp_5.00")
            now_txt = os.path.join(now_path, "Branch.txt.txt")
            new_txt_name = os.path.splitext(os.path.split(image_name)[1])[0] + "_curve.txt"
            try:
                shutil.copy(now_txt, os.path.join(sketch_path, new_txt_name))
            except FileNotFoundError:
                print("----------------------------------")
                print("No such file or directory: {}".format(now_txt))
                print("----------------------------------")
                pass
            # 复制原图
            shutil.copy(image_name, os.path.join(sketch_path, os.path.split(image_name)[1]))
        pass

    @staticmethod
    def _save_data_to_image(data, file_name, convert="RGB"):
        Image.fromarray(data).convert(convert).save(file_name)
        pass

    @staticmethod
    def _new_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # 将切块后生成的各个素描图拷贝到一个文件夹下
    def copy_sketch_to_together(self):
        together_path = self._new_dir(self.root_path + "_sketch_together")
        for image_name in self.image_names:
            now_path = os.path.join(os.path.splitext(image_name)[0] + "_Sketch", "pp_5.00")
            now_bmp = os.path.join(now_path, "SketchMap_6.bmp")
            new_bmp_name = os.path.split(os.path.splitext(image_name)[0])[1] + "_SketchMap.bmp"
            try:
                shutil.copy(now_bmp, os.path.join(together_path, new_bmp_name))
            except FileNotFoundError:
                print("----------------------------------")
                print("No such file or directory: {}".format(now_bmp))
                print("----------------------------------")
                pass
        pass

    # 合并切块跑出来的素描图为一张大图
    def merge_sketch(self):
        sketch_data = np.zeros(shape=[self.image_x, self.image_y], dtype=np.uint8)
        together_path = self._new_dir(self.root_path + "_sketch_together")
        self.image_names = glob(os.path.join(together_path, "*.bmp"))
        for image_name in self.image_names:
            print(image_name)
            now_data = np.asarray(Image.open(image_name), dtype=np.uint8)
            name = os.path.splitext(os.path.split(image_name)[1])[0]
            poss = [position for position in name.split("_")][0:4]
            poss = [int(position) for position in poss]
            sketch_data[poss[0]: poss[2], poss[1]: poss[3]] = now_data
            pass
        self._save_data_to_image(sketch_data, self.result_file_name, "L")
        pass

    pass


def new_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

if __name__ == '__main__':

    lf = 0.7
    hf = 1.4
    root_path = "image_sar"
    path = new_dir("{}_{}_{}".format(root_path, lf, hf))

    images = [os.path.join(root_path, "3.png")]
    new_images = ["{}/3.png".format(path)]
    # copy
    for index, img in enumerate(images):
        shutil.copy(img, new_images[index])
    # run
    for image in new_images:
        sketch_large_image = SketchLargeImageToSar(image_file=image, lf=lf, hf=hf)
        sketch_large_image.run()

    pass
