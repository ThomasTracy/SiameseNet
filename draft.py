import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random
import os

from model import SiameseNet
from pprint import pprint
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_data_gen = image_generator.flow_from_directory(directory='/home/tracy/data/TrafficSign_single/Images/00000',
#                                                      batch_size=1,
#                                                      shuffle=True,
#                                                      )

# data_dir = '/home/tracy/data/TrafficSign_single/Images'
# list_ds = tf.data.Dataset.list_files(str(data_dir+'/*/*'))
# for f in list_ds.take(-1):
#     print(len(f.numpy()))


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [64,64])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label

def train_data_gen(dataset, shuffle_buffer_size=1000):
    #取出1000张图片， 打乱放入缓存器当中
    dataset = dataset.shuffle(buffer_size = shuffle_buffer_size)

    #无限重复接收
    dataset = dataset.repeat()
    dataset = dataset.batch(2)

    #在训练时后台取数据放入寄存器中
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def haha():
    a = tf.random.normal([1,164,164,3])
    b = tf.random.normal([1,164,164,3])
    label = tf.constant([1, 0])
    net = SiameseNet.SiameseNet()
    loss = net(a, b, label)

    seq = net.encoder
    seq(a)
    print(seq.summary())

    pprint(net.layers)
    print('\033[1;32mLoss is\033[0m')
    print(loss)

def split_dataset():
    data_path = '/home/tracy/data/TrafficSign_single/Images/all'
    classes = os.listdir(data_path)
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        imgs = os.listdir(class_path)
        num_val = int(0.15 * len(imgs))
        num_test = int(0.15 * len(imgs))
        random.shuffle(imgs)
        for _ in range(num_val):
            img = imgs.pop()
            img_path = os.path.join(class_path, img)
            save_path = os.path.join('/home/tracy/data/TrafficSign_single/Images/val', cls)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, img)
            im = Image.open(img_path)
            im.save(save_path)
        for _ in range(num_test):
            img = imgs.pop()
            img_path = os.path.join(class_path, img)
            save_path = os.path.join('/home/tracy/data/TrafficSign_single/Images/test', cls)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, img)
            im = Image.open(img_path)
            im.save(save_path)
        for img in imgs:
            img_path = os.path.join(class_path, img)
            save_path = os.path.join('/home/tracy/data/TrafficSign_single/Images/train', cls)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, img)
            im = Image.open(img_path)
            im.save(save_path)

        # pprint(imgs)

if __name__ == '__main__':


    # truth = tf.constant([0, 1, 1, 0])
    # one_hot_truth = tf.one_hot(truth, 2)
    # print(one_hot_truth)
    # pred = tf.constant([[0.01, 0.5], [100.,1000], [0.0001, 0.], [0.1,0.7]])
    # res = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_truth, logits=pred)
    # label_pred = tf.nn.softmax(pred)
    # print(label_pred)
    # print('loss', res)
    # print(numpy.random.choice(b, 6, replace=False))
    # print(tf.tile(tf.range(0, 2, dtype=tf.float32), [0]))
    # print(1//4)
    x = tf.constant([0.7, 0.7])
    y = tf.constant([1., 0.])
    z = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)
    w = -1 * numpy.log(1/(1 + numpy.exp(-0.7)))
    a = tf.constant([0,1,2,3])
    b = tf.constant([[1], [2], [3]])
    b = tf.squeeze(b)
    print(numpy.exp(-0.7))

    print(z)
    print(w)
    print(tf.cast(a>0, tf.float32))
    print(b.get_shape())



