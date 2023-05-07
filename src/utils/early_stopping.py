import numpy as np
from typing import Callable


class EarlyStopping:
    def __init__(self, patience=10, delta=0, mode='min', save_callback: Callable[[], None] = None,
                 verbose=False, trace_func=print):
        """
        if the lesser monitor_value the better value be, mode should be 'min'.
        """
        assert mode in ['min', 'max'], 'mode must be "min" or "max"'
        assert save_callback is not None, 'save_callback must be defined'
        self._patience = patience
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_monitor_value = None
        self.early_stop = False
        self.save_callback = save_callback
        self.verbose = verbose
        self.trace_func = trace_func
        self.store_info = None
        if mode == 'min':
            self.monitor_op = np.greater
            self.delta = delta
        else:
            self.monitor_op = np.less
            self.delta = -delta

    def __call__(self, monitor_value, store_info=None):
        if self.best_monitor_value is None:
            self.best_monitor_value = monitor_value
            self.store_info = store_info
            self.save_callback()
        else:
            if self.monitor_op(monitor_value, self.best_monitor_value + self.delta):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                if self.verbose:
                    self.trace_func(f'best monitor value changed, ({self.best_monitor_value} --> {monitor_value})')
                self.best_monitor_value = monitor_value
                self.store_info = store_info
                self.counter = 0
                self.save_callback()

    def init(self):
        self.patience = self._patience
        self.best_monitor_value = None
        self.early_stop = False
        self.counter = 0
        self.store_info = None

    def get_store_info(self):
        return self.store_info
