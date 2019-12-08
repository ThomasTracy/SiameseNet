import logging
import numpy as np

from tqdm import tqdm


class TranEngine(object):
    """
    Engine for training on every epoch
    Also hooks for certain action
    """

    def __init__(self):
        self.hooks = {name: lambda state: None
                      for name in ['start',
                                   'end_epoch',
                                   'train_step',
                                   'end']}

    def train(self, loss_func, train_dataset, val_dataset, epochs, step_per_epoch, **kwargs):

        #State of train procedure
        state = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'loss_func': loss_func,
            'train_data': None,
            'total_steps': 1,
            'current_epoch': 0,
            'epochs': epochs,
            'step_per_epoch': step_per_epoch,
            'best_val_loss': np.inf,
            'early_stopping_triggered': False
        }

        self.hooks['start'](state)

        for epoch in range(state['epochs']):
            for _ in tqdm(range(state['step_per_epoch'])):
                img1, img2, label = next(train_dataset)
                state['train_data'] = (img1, img2, label)
                self.hooks['train_step'](state)
                # self.hooks['end_step'](state)
                state['total_steps'] += 1

            self.hooks['end_epoch'](state)
            state['current_epoch'] += 1

            if state['early_stopping_triggered']:
                print('\033[1;31mEarly stopped\033[0m')
                break

        self.hooks['end'](state)
        logging.info('\033[1;32mTraining Finished!\033[0m')


if __name__ == '__main__':
    print('\033[1;32mEarly stopped\033[0m')