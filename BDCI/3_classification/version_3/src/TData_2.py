"""
    对于每个图片：
        1.补边
        2.根据步长滑窗
    参数：
        批大小、当前批次数
    输出：
        总批次数、处理后的一批数据
"""
import numpy as np
from PIL import Image


class TData:

    def __init__(self, image_size, batch_size, stripe, image_file):
        self._batch_size = batch_size
        self._image_size = image_size
        self._stripe = stripe

        # 原始数据
        self.origin_image_data = np.copy(np.asarray(Image.open(image_file)))
        self.origin_x, self.origin_y = len(self.origin_image_data), len(self.origin_image_data[0])

        # 补边之后的数据
        self.new_image_data = np.zeros([self.origin_x + self._image_size - 1,
                                        self.origin_y + self._image_size - 1, 3], dtype=np.uint8)
        self.new_image_data = self.padding()
        self.new_x, self.new_y = len(self.new_image_data), len(self.new_image_data[0])

        self._pre_row_number = (self.origin_y - self._stripe // 2) // self._stripe + 1
        self._pre_col_number = (self.origin_x - self._stripe // 2) // self._stripe + 1
        self.batch_all_number = self._pre_col_number * self._pre_row_number // self._batch_size
        pass

    # 补边
    def padding(self):
        self.new_image_data[self._image_size // 2:self.origin_x + self._image_size // 2,
            self._image_size // 2:self.origin_y + self._image_size // 2, :] = self.origin_image_data[:, :, :]

        top_data = self.new_image_data[self._image_size // 2: self._image_size - 1, :, :]
        self.new_image_data[0: self._image_size // 2, :, :] = np.flip(top_data, axis=0)

        bottom_data = self.new_image_data[self.origin_x: self.origin_x + self._image_size // 2, :, :]
        self.new_image_data[self.origin_x + self._image_size // 2:, :, :] = np.flip(bottom_data, axis=0)

        left_data = self.new_image_data[:, self._image_size // 2: self._image_size - 1, :]
        self.new_image_data[:, 0: self._image_size // 2, :] = np.flip(left_data, axis=1)

        right_data = self.new_image_data[:, self.origin_y: self.origin_y + self._image_size // 2, :]
        self.new_image_data[:, self.origin_y + self._image_size // 2:, :] = np.flip(right_data, axis=1)

        return self.new_image_data

    # 获取一批数据：返回数据和位置
    def get_batch_data(self, batch_number):
        datas = []
        positions = []

        start_x = (batch_number * self._batch_size) // self._pre_row_number
        start_y = (batch_number * self._batch_size) % self._pre_row_number

        count = 0

        for y in range(start_y, self._pre_col_number):
            count += 1
            if count > self._batch_size:
                return datas, positions
            crop_data, position = self.get_data_and_position_by_x_y(start_x, y)
            datas.append(crop_data)
            positions.append(position)
            pass

        for x in range(start_x + 1, self._pre_row_number):
            for y in range(0, self._pre_col_number):
                count += 1
                if count > self._batch_size:
                    return datas, positions
                crop_data, position = self.get_data_and_position_by_x_y(x, y)
                datas.append(crop_data)
                positions.append(position)
            pass
        return datas, positions

    def get_data_and_position_by_x_y(self, x, y):
        x_center = x * self._stripe + self._image_size // 2 + self._stripe // 2
        y_center = y * self._stripe + self._image_size // 2 + self._stripe // 2
        crop_data = self.new_image_data[x_center - self._image_size // 2: x_center + self._image_size // 2 + 1,
                    y_center - self._image_size // 2: y_center + self._image_size // 2 + 1, :]
        crop_data = crop_data / 255.0
        position = (x_center - self._image_size // 2, y_center - self._image_size // 2)
        return crop_data, position

    pass


if __name__ == '__main__':
    test_data = TData(227, batch_size=256, stripe=100, image_file="../data/CCF-testing/1.png")

    for x in range(test_data.batch_all_number):
        datas, positions = test_data.get_batch_data(x)
        print("{}/{} {} {} {} {} ({},{})".format(x, test_data.batch_all_number,
                                                 len(datas), len(datas[0]), len(datas[0][0]),
                                                 len(datas[0][0][0]), positions[-1][0], positions[-1][1]))
        pass
