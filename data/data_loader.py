import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_DIR = '/home/tracy/data/TrafficSign_single/Images'


class DataSet:
    def __init__(self, batch_size=1, mode='train'):
        self.mode = mode
        self.batch_size = batch_size
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT
        self.dataset_dir = os.path.join(DATASET_DIR, mode)
        self.classes = os.listdir(self.dataset_dir)
        self.probe = [1,2,3,4,5]


    def __next__(self):
        return self.load()


    def load_img(self, img_path):
        img = tf.io.read_file(img_path)

        # Turn image in 64x64x3   0 -- 1.0
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.width, self.height])


    def random_load_img(self, dir_path):
        img_list = os.listdir(dir_path)
        img_name = np.random.choice(img_list)
        img_path = os.path.join(dir_path, img_name)
        return self.load_img(img_path)


    def pair_generate(self, class_pair):
        '''

        :param class_pair: list of classes of choosen pair
        :return: [img1, img2]
        '''
        return list(map(self.random_load_img,
                                  list(map(lambda x: self.dataset_dir + '/' + x,
                                      class_pair) )           # [class1, class2] / [class1, class1]
                                 ))


    def load(self):
        """
        Load batches of image1 and image2 and labels of them
        :return:
        img1    [batch, width, height, 3]
        img2    [batch, width, height, 3]
        labe    [batch, ]
        """
        img1 = []
        img2 = []
        labels = []

        if self.mode == 'train':

            for i in range(self.batch_size):
                label = np.random.randint(0, 2)
                if label == 0:          # Different class
                    negative_pair = self.pair_generate(np.random.choice(self.classes, 2, replace=False))
                    img1.append(tf.expand_dims(negative_pair[0], axis=0))
                    img2.append(tf.expand_dims(negative_pair[1], axis=0))
                    labels.append(tf.expand_dims(label, axis=0))
                else:                   # Same classes
                    positive_pair = self.pair_generate([np.random.choice(self.classes)] * 2)
                    img1.append(tf.expand_dims(positive_pair[0], axis=0))
                    img2.append(tf.expand_dims(positive_pair[1], axis=0))
                    labels.append(tf.expand_dims(label, axis=0))

            img1 = tf.concat(img1, axis=0)          # [batch, width, height, channel]
            img2 = tf.concat(img2, axis=0)
            labels = tf.concat(labels, axis=0)      # [batch, label]

        elif self.mode == 'val':
            pass
        return img1, img2, labels


    def test(self):
        print(self.classes)

def get_cls(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [64,64])


def process_path(file_path):
    cls = get_cls(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, cls


def build_dataset(mode='train', cls='*', shuffle_buffer_size=1000, batch_size=1):
    data_path = os.path.join(DATASET_DIR, mode)
    list_ds = tf.data.Dataset.list_files(str(data_path + f'/{cls}/*.jpg'))

    dataset = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)#.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def load(mode='train', batch_size=2):
    dataset = build_dataset(mode=mode)
    if mode == 'train':
        label = np.random.randint(0, 2)
        img1, cls1 = next(iter(dataset))
        if label == 0:
            pass


def get_len_dataset(mode='train'):
    dataset_dir = os.path.join(DATASET_DIR, mode)
    length = 0
    for root, dirs, files in os.walk(dataset_dir):
        length_1 = len(files)
        length += length_1
    print(length)


def show_result(img1, img2, labels):
    num = img1.get_shape()[0]

    fig = plt.figure()
    for i in range(num):
        ax1 = plt.subplot(8, 8, 2*i+1)
        ax1.set_title(str(labels[i].numpy()))
        ax1.imshow(img1[i].numpy())

        ax2 = plt.subplot(8, 8, 2*i+2)
        ax2.set_title(str(labels[i].numpy()))
        ax2.imshow(img2[i].numpy())

    plt.show()


# show_result()
# dataset = DataSet(batch_size=32)
# img1, img2, labels = next(dataset)
# print(labels)
# show_result(img1, img2, labels)

# get_len_dataset()
if __name__ == '__main__':
    with tf.device('/GPU:0'):
        print('No Problem')
        print(tf.executing_eagerly())
        dataset = DataSet()
        img1, img2, labels = next(dataset)
        print(img1.numpy())