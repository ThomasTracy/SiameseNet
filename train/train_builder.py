import os
import time
import datetime
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.data_loader import DataSet
from train import TrainEngine
from model import SiameseNet
from shutil import copyfile

# Use f-string to format file name. E.g. 2019_12_03_03:42.log
real_log = f"{datetime.datetime.now():%Y_%m_%d_%H:%M}.log"
logging.basicConfig(#filename=real_log,
                    format='%(asctime)s - %(levelname)s - %(message)s', # 打印日志的时间, 级别名称, 日志信息
                    datefmt='%Y.%m.%d.%H-%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def show_result(img1, img2, labels):

    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax1.set_title(str(labels[0].numpy()))
    ax1.imshow(img1[0].numpy())
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(str(labels[0].numpy()))
    ax2.imshow(img2[0].numpy())

    plt.show()


def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Cteate model's folder
    model_dir = config['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create log's folder
    log_dir = config['train_log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = f"SiameseNet_{datetime.datetime.now():%Y_%m_%d-%H:%M}.log"
    log_name = os.path.join(log_dir, log_name)
    print(f"\033[1;32mAll Infomations can be found in {log_name}\033[0m")

    # Initialize data loader
    data_dir = config['dataset_path']
    train_dataset = DataSet(mode='train')
    val_dataset = DataSet(mode='val')

    train_engine = TrainEngine.TranEngine()

    # Training options
    model = SiameseNet.SiameseNet()
    # TODO: add restore functions

    optimizer = tf.keras.optimizers.RMSprop(config['learning_rate'])

    # Metrics to gather results
    train_loss = tf.metrics.Mean(name='train_loss')
    train_acc = tf.metrics.Mean(name='train_acc')
    val_loss = tf.metrics.Mean(name='val_loss')
    val_acc = tf.metrics.Mean(name='val_acc')

    # Summary writers
    current_time = datetime.datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
    train_summary_writer = tf.summary.create_file_writer(config['tensorboard_dir'])

    def loss(img1, img2, label):
        return model(img1, img2, label)

    # Forward and upgrade gradients
    def train_step(state):
        img1, img2, labels = state['train_data']
        with tf.GradientTape() as tape:
            loss, label_predict, acc = model(img1, img2, labels)

        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        # show_result(img1, img2, labels)
        # print(f"\033[1;32mLoss: {loss.numpy()} | Label: {labels.numpy()} | Prediction: {label_predict.numpy()} | Acc: {acc.numpy()}\033[0m")

        train_loss(loss)
        train_acc(acc)

        # if state['total_steps'] % 100 ==0:
        #     logging.info(f"Step: {state['total_steps']} | Loss: {loss.numpy()} | Loss-avg: {train_loss.result().numpy()}")

    def val_step(img1, img2, labels):
        loss,_ , acc = model(img1, img2, labels)
        val_loss(loss)
        val_acc(acc)


    def start(state):
        logging.info("\033[1;31m************** Start Training **************\033[0m")

    def end(state):
        logging.info("\033[1;31m************** End Training **************\033[0m")


    def end_epoch(state):
        epoch = state['current_epoch'] + 1

        logging.info(f"\033[1;32m************** End Epoch {epoch} **************\033[0m")
        template = 'Epoch {} | Loss: {:.6f} | Accuracy: {:.3%} | ' \
                   'Val Loss: {:.6f} | Val Accuracy: {:.3%}'
        logging.info(template.format(epoch, train_loss.result(),
                                     train_acc.result(),
                                     val_loss.result(),
                                     val_acc.result()))
        current_loss = val_loss.result().numpy()
        if current_loss < state['best_val_loss']:
            logging.info("\033[1;32m************** Saving the best model with loss: "
                         "{:.6f} **************\033[0m".format(current_loss))
            state['best_val_loss'] = current_loss
            model.save(save_dir=config['model_dir'], model=model)

        #TODO: Early stopping
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch*config['step_per_epoch'])
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch*config['step_per_epoch'])

        # Reset metrics
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()


    def start_step(state):
        img1, img2, labels = state['train_data']
        if state['total_steps']%100 == 0:
            logging.info(f'Trainning step')

    train_engine.hooks['start'] = start
    train_engine.hooks['end'] = end
    train_engine.hooks['end_epoch'] = end_epoch
    train_engine.hooks['train_step'] = train_step

    time_start = time.time()
    # with tf.device('/gpu:0'):
    train_engine.train(loss_func=loss,
                       train_dataset=train_dataset,
                       val_dataset=val_dataset,
                       epochs=config['epochs'],
                       step_per_epoch=config['step_per_epoch'])

    time_end = time.time()
    total_time = time_end - time_start
    h, m, s = total_time//3600, total_time%3600//60, total_time%3600%60
    logging.info(f"\033[1;31m************** Totally used {h}hour {m}minute {s}second **************\033[0m")
    copyfile(real_log, log_name)



def train_step():

    model = SiameseNet.SiameseNet()
    train_data = DataSet(mode='train', batch_size=4)
    img1, img2, labels = next(train_data)
    loss, pred_label, acc = model(img1, img2, labels)
    print(labels)
    print('loss: ', loss)
    print('predict_label:', pred_label)
    print('acc: ', acc)


if __name__ == '__main__':
    train_step()
    print('Dont Run')