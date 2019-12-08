import os
import tensorflow as tf
import logging
import time
import datetime
import configparser
import argparse
import pprint

from shutil import copyfile
from absl import flags, app
from train.train_builder import train


FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint', '/home/tracy/PycharmProjects/SiameseNet/checkpoint',
                    'directory to save checkpoints')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)]
    )

def main(_):
    if FLAGS.checkpoint:
        print('dir is: %s'%FLAGS.checkpoint)


def config_gen():
    config = configparser.ConfigParser()
    config['Train'] = {'model_dir': '/home/tracy/PycharmProjects/SiameseNet/models',
                       'train_log_dir': '/home/tracy/PycharmProjects/SiameseNet/logs',
                       'dataset_path': '/home/tracy/data/TrafficSign_single/Images',
                       'learning_rate': 0.1,
                       'batch_size': 1}
    config['Eval'] = {'dataset_path': '/home/tracy/data/TrafficSign_single/Images',
                      'batch_size': 1}

    with open('traffic_signs.conf', 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='traffic_signs.conf', help='Path to config file')
    parser.add_argument('--model_dir', type=str,
                        default='/home/tracy/PycharmProjects/SiameseNet/checkpoint', help='Path to save models')
    parser.add_argument('--train_log_dir', type=str,
                        default='/home/tracy/PycharmProjects/SiameseNet/logs', help='Path to save logging files')
    parser.add_argument('--dataset_path', type=str,
                        default='/home/tracy/data/TrafficSign_single/Images', help='Path to dataset')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='/home/tracy/PycharmProjects/SiameseNet/tensorboard', help='Path to tensorboard files')

    parser.add_argument('--learning_rate', type=int, default=0.001, help='Path to config file')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluate batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--step_per_epoch', type=int, default=100, help='Total running steps per epoch')

    configs = vars(parser.parse_args())
    pprint.pprint(configs)
    train(configs)
