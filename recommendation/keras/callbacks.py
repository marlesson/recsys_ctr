import math
from typing import List, Dict

import numpy as np

from keras.callbacks import Callback, LearningRateScheduler
import keras.backend as K

class StepLR(LearningRateScheduler):
    def __init__(self, step_size, gamma=0.1, verbose=0):
        super().__init__(self._get_lr, verbose)
        self._step_size = step_size
        self._gamma = gamma

    def _get_lr(self, epoch: int, lr: float):
        if not hasattr(self, "_lr0"):
            self._lr0 = lr

        return self._lr0 * self._gamma ** (epoch // self._step_size)


class ExponentialLR(LearningRateScheduler):
    def __init__(self, gamma, verbose=0):
        super().__init__(self._get_lr, verbose)
        self._gamma = gamma

    def _get_lr(self, epoch: int, lr: float):
        if not hasattr(self, "_lr0"):
            self._lr0 = lr

        return self._lr0 * self._gamma ** epoch


class CosineAnnealingLR(LearningRateScheduler):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super().__init__(self._get_lr, verbose)
        self._T_max = T_max
        self._eta_min = eta_min

    def _get_lr(self, epoch: int, lr: float):
        if not hasattr(self, "_lr0"):
            self._lr0 = lr

        return self._eta_min + (self._lr0 - self._eta_min) * (1 + math.cos(math.pi * epoch / self._T_max)) / 2


class CosineAnnealingWithRestartsLR(CosineAnnealingLR):
    def __init__(self, T_max, eta_min=0, T_mult=1, verbose=0):
        super().__init__(T_max, eta_min, verbose)

        self._T_mult = T_mult

        self._restart_every = T_max
        self._restarts = 0
        self._restarted_at = 0

    def _restart(self, epoch: int):
        self._restart_every *= self._T_mult
        self._restarted_at = epoch

    def _get_lr(self, epoch: int, lr: float):
        if (epoch - self._restarted_at) >= self._restart_every:
            self._restart(epoch)

        return super()._get_lr(epoch, lr)


class LearningRateFinder(Callback):
    def __init__(self, iterations: int, initial_lr: float, end_lr: float = 10.0, linear=False, stop_dv=True) -> None:
        super().__init__()

        ratio = end_lr / initial_lr
        self._gamma = (ratio / iterations) if linear else ratio ** (1 / iterations)

        self._iterations = iterations
        self._current_iteration = 0
        self._stop_dv = stop_dv
        self._best_loss = 1e9
        self.learning_rates: List[float] = []
        self.loss_values: List[float] = []

    def on_batch_begin(self, batch: int, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        if not hasattr(self, "_lr0"):
            self._lr0 = lr

        lr = self._lr0 * self._gamma ** batch

        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch: int, logs=None):
        self._current_iteration += 1

        self.learning_rates.append(float(K.get_value(self.model.optimizer.lr)))
        loss = logs['loss']
        self.loss_values.append(loss)

        if loss < self._best_loss:
            self._best_loss = loss

        if self._current_iteration >= self._iterations or (self._stop_dv and loss > 10 * self._best_loss):
            self.model.stop_training = True

    def get_loss_derivatives(self, sma: int = 1):
        derivatives = [0] * (sma + 1)
        for i in range(1 + sma, len(self.loss_values)):
            derivative = (self.loss_values[i] - self.loss_values[i - sma]) / sma
            derivatives.append(derivative)
        return derivatives



