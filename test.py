import tensorflow as tf

from model.SiameseNet import SiameseNet
from data.data_loader import DataSet
import matplotlib.pyplot as plt


def show_result(img1, img2, label):
    num = img1.get_shape()[0]
    print(num)

    for i in range(num):
        ax1 = plt.subplot(5,4,2*i + 1)
        ax1.imshow(img1[i].numpy())
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(label[i].numpy())

        ax2 = plt.subplot(5,4,2*i+2)
        ax1.set_title(label[i].numpy())
        ax2.imshow(img2[i].numpy())
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.show()

model = SiameseNet()
# latest = tf.train.latest_checkpoint('/home/tracy/PycharmProjects/SiameseNet/checkpoint/')
# print(latest)
# model.load_weights(latest)
# model.load('/home/tracy/PycharmProjects/SiameseNet/checkpoint/', model)

model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/my_model')

test_dataset = DataSet(mode='test', batch_size=10)

img1, img2 = next(test_dataset)

label = model.prediction(img1, img2)
print(label)

show_result(img1, img2, label)

