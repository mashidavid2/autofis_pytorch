import time
import numpy as np


class Timer:
    def __init__(self, train_size, batch_size, epoch_size, unit_batch=100):
        self._start_time = None
        self._train_size = train_size
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._unit_batch = unit_batch

        self._num_batches_per_epoch = int(np.ceil(train_size / batch_size))
        self._num_total_batches = epoch_size * self._num_batches_per_epoch
        self._num_finished_batches = 0
        self.elapsed = 0
        self.eta = -1 # estimated time of accomplish

    def __call__(self, stage='train', early_stop=False, total_epoch=None):
        """
        elapsed time is calculated, only depending on training time (exclude testing time)
        """
        assert stage in ['train', 'test', 'info']
        if stage == 'train':
            self._num_finished_batches += self._unit_batch
        elif stage == 'test':
            if not early_stop:
                pass
            elif early_stop and total_epoch is None:
                self._num_finished_batches = self._num_total_batches
            else:
                self._num_total_batches = total_epoch * self._num_batches_per_epoch
        else:
            pass

        self.elapsed = int(time.time() - self._start_time)
        self.eta = int((self._num_total_batches - self._num_finished_batches) /
                       self._num_finished_batches * self.elapsed)
        self.eta = max(0, self.eta)
        return self.elapsed, self.eta

    def start(self):
        self._start_time = time.time()
