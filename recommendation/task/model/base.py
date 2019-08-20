import abc
import json
import logging
import multiprocessing
import os
import shutil
from collections import OrderedDict
from contextlib import redirect_stdout
from typing import Type, Tuple, List, Callable

import luigi
import pandas as pd
import numpy as np
import keras
import tensorflow as tf

from keras import backend as K
from keras.activations import relu as keras_relu, selu as keras_selu, tanh as keras_tanh, sigmoid as keras_sigmoid, \
    linear as keras_linear
from keras.callbacks import ModelCheckpoint as KerasModelCheckpoint, EarlyStopping as KerasEarlyStopping, \
    CSVLogger as KerasCSVLogger, TensorBoard as KerasTensorBoard, Callback as KerasCallback, \
    ReduceLROnPlateau as KerasReduceLROnPlateau
from keras.engine.training import Model as KerasModel
from keras.initializers import lecun_normal as keras_lecun_normal, he_normal as keras_he_normal, \
    glorot_normal as keras_glorot_normal, lecun_uniform as keras_lecun_uniform, he_uniform as keras_he_uniform, \
    glorot_uniform as keras_glorot_uniform
from keras.losses import binary_crossentropy as keras_binary_crossentropy

from recommendation.files import get_params_path, get_torch_weights_path, get_params, get_history_path, \
    get_tensorboard_logdir, get_classes_path, get_task_dir, get_keras_weights_path, get_history_plot_path

from keras.optimizers import Adam as KerasAdam, RMSprop as KerasRMSprop, SGD as KerasSGD, Adadelta as KerasAdadelta, \
    Adagrad as KerasAdagrad, Adamax as KerasAdamax
from keras.regularizers import l1 as keras_l1, l2 as keras_l2
from keras.optimizers import Optimizer as KerasOptimizer
from recommendation.keras.callbacks import StepLR as KerasStepLR, ExponentialLR as KerasExponentialLR, \
    CosineAnnealingLR as KerasCosineAnnealingLR, CosineAnnealingWithRestartsLR as KerasCosineAnnealingWithRestartsLR, \
    LearningRateFinder as KerasLearningRateFinder
from recommendation.plot import plot_history, plot_loss_per_lr, plot_loss_derivatives_per_lr, plot_confusion_matrix
from recommendation.task.cuda import CudaRepository
from recommendation.task.data_preparation.data_preparation import PrepareDataFrames
import keras_metrics

# confirm Keras sees the GPU
from keras import backend

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

KERAS_OPTIMIZERS = dict(adam=KerasAdam, rmsprop=KerasRMSprop, sgd=KerasSGD, adadelta=KerasAdadelta,
                        adagrad=KerasAdagrad, adamax=KerasAdamax)
KERAS_LOSS_FUNCTIONS = dict(binary_crossentropy=keras_binary_crossentropy)
KERAS_ACTIVATION_FUNCTIONS = dict(relu=keras_relu, selu=keras_selu, tanh=keras_tanh, sigmoid=keras_sigmoid,
                                  linear=keras_linear)
KERAS_WEIGHT_INIT = dict(lecun_normal=keras_lecun_normal, he_normal=keras_he_normal, glorot_normal=keras_glorot_normal,
                         lecun_uniform=keras_lecun_uniform,
                         he_uniform=keras_he_uniform, glorot_uniform=keras_glorot_uniform)
KERAS_LR_SCHEDULERS = dict(step=KerasStepLR, exponential=KerasExponentialLR, cosine_annealing=KerasCosineAnnealingLR,
                           cosine_annealing_with_restarts=KerasCosineAnnealingWithRestartsLR,
                           reduce_on_plateau=KerasReduceLROnPlateau)
KERAS_REGULARIZERS = dict(l1=keras_l1, l2=keras_l2)

SEED = 42

DEFAULT_DEVICE = "cuda" if len(backend.tensorflow_backend._get_available_gpus()) > 0 else "cpu"

class BaseModelTraining(luigi.Task):
    __metaclass__ = abc.ABCMeta
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    training_type: str = luigi.ChoiceParameter(choices=["fit", "lr_find", "debug"], default="fit")
    mode: str = luigi.ChoiceParameter(choices=["classification", "localization"], default="localization")
    lr_find_iterations: int = luigi.IntParameter(default=100)

    val_size: float = luigi.FloatParameter(default=0.2)
    input_shape: Tuple[int, int] = luigi.TupleParameter(default=(224, 224))
    not_use_multiprocessing: bool = luigi.BoolParameter(default=False)
    learning_rate: float = luigi.FloatParameter(1e-3)
    batch_size: int = luigi.IntParameter(default=50)
    val_batch_size: int = luigi.IntParameter(default=100)
    epochs: int = luigi.IntParameter(default=100)
    monitor_metric: str = luigi.Parameter(default="val_loss")
    monitor_mode: str = luigi.ChoiceParameter(choices=["min", "max", "auto"], default="auto")
    early_stopping_patience: int = luigi.IntParameter(default=10)
    early_stopping_min_delta: float = luigi.FloatParameter(default=1e-6)
    generator_workers: int = luigi.IntParameter(default=multiprocessing.cpu_count())
    generator_max_queue_size: int = luigi.IntParameter(default=20)
    initial_weights_path = luigi.Parameter(default=None)

    seed: int = luigi.IntParameter(default=42)

    def requires(self):
        return [PrepareDataFrames(val_size=self.val_size, seed=self.seed)]

    def output(self):
        return luigi.LocalTarget(get_task_dir(self.__class__, self.task_id))

    @property
    def resources(self):
        return {"cuda": 1} if self.device == "cuda" else {}

    @property
    def device_id(self):
        if not hasattr(self, "_device_id"):
            if self.device == "cuda":
                self._device_id = CudaRepository.get_avaliable_device()
            else:
                self._device_id = None
        return self._device_id

    def _save_params(self):
        params = self._get_params()
        with open(get_params_path(self.output_path), "w") as params_file:
            json.dump(params, params_file, default=lambda o: dict(o), indent=4)

    def _get_params(self):
        return self.param_kwargs

    @property
    def output_path(self):
        if hasattr(self, "_output_path"):
            return self._output_path
        return self.output().path

    @property
    def train_dataset(self) -> pd.DataFrame:
        train_df = pd.read_csv(self.input()[0][0].path, sep=";")
        return train_df

    @property
    def val_dataset(self) -> pd.DataFrame:
        val_df = pd.read_csv(self.input()[0][1].path, sep=";")
        return val_df

    def get_test_df(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0][2].path, sep=";")

    @property
    def test_dataset(self) -> pd.DataFrame:
        return self.get_test_df()

    def train(self):
        dict(fit=self.fit, lr_find=self.lr_find, debug=self.debug)[self.training_type]()

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def lr_find(self):
        pass

    def debug(self):
        print("Debug não implementado!")

    def after_train(self):
        pass

    def run(self):
        os.makedirs(self.output_path, exist_ok=True)
        self._save_params()
        try:
            self.train()
            if self.training_type == "fit":
                self.after_train()
        except Exception:
            shutil.rmtree(self.output_path)
            raise
        finally:
            if self.device == "cuda":
                CudaRepository.put_available_device(self.device_id)

class BaseKerasModelTraining(BaseModelTraining):
    __metaclass__ = abc.ABCMeta

    optimizer: str = luigi.ChoiceParameter(choices=KERAS_OPTIMIZERS.keys(), default="adam")
    optimizer_params: dict = luigi.DictParameter(default={})
    lr_scheduler: str = luigi.ChoiceParameter(choices=KERAS_LR_SCHEDULERS.keys(), default=None)
    lr_scheduler_params: dict = luigi.DictParameter(default={})
    loss_function: str = luigi.ChoiceParameter(choices=KERAS_LOSS_FUNCTIONS.keys(), default="binary_crossentropy")
    kernel_regularizer: str = luigi.ChoiceParameter(choices=KERAS_REGULARIZERS.keys(), default=None)
    kernel_regularizer_value: float = luigi.FloatParameter(default=0.01)

    @abc.abstractmethod
    def create_model(self) -> KerasModel:
        """Retornar um modelo do Keras"""
        pass

    def _get_metrics(self):
        return []

    def lr_find(self):
        with self.tensorflow_device:
            train_loader = self.get_train_generator()

            self.learning_rate = 1e-6
            lr_finder = KerasLearningRateFinder(min(len(train_loader), self.lr_find_iterations), self.learning_rate)

            self.keras_model = self.create_model()
            if self.initial_weights_path:
                self.keras_model.load_weights(self.initial_weights_path, by_name=True, skip_mismatch=True)
            self.keras_model.compile(optimizer=self._get_optimizer(), loss=self._get_loss_function())

            self.keras_model.fit_generator(train_loader, steps_per_epoch=self.lr_find_iterations, epochs=1, verbose=1,
                                           callbacks=[lr_finder])

            loss_per_lr_path = os.path.join(self.output_path, "loss_per_lr.jpg")
            loss_derivatives_per_lr_path = os.path.join(self.output_path, "loss_derivatives_per_lr.jpg")

            plot_loss_per_lr(lr_finder.learning_rates, lr_finder.loss_values) \
                .savefig(loss_per_lr_path)
            plot_loss_derivatives_per_lr(lr_finder.learning_rates, lr_finder.get_loss_derivatives(5)) \
                .savefig(loss_derivatives_per_lr_path)

    def fit(self):
        with self.tensorflow_device:
            try:
                self.keras_model = self.create_model()
                if self.initial_weights_path:
                    self.keras_model.load_weights(self.initial_weights_path, by_name=True, skip_mismatch=True)
                if self.kernel_regularizer:
                    for layer in self.keras_model.layers:
                        if hasattr(layer, 'kernel_regularizer'):
                            layer.kernel_regularizer = KERAS_REGULARIZERS[self.kernel_regularizer](
                                self.kernel_regularizer_value)
                if self.mode == "classification":
                    metrics = self._get_metrics() + ["acc", keras_metrics.precision(), keras_metrics.recall()]
                else:
                    metrics = self._get_metrics() + []
                self.keras_model.compile(optimizer=self._get_optimizer(), loss=self._get_loss_function(),
                                         metrics=metrics)

                with open("%s/summary.txt" % self.output_path, "w") as summary_file:
                    with redirect_stdout(summary_file):
                        self.keras_model.summary()

                try:

                    self.keras_model.fit(x = self.train_generator[0], 
                                         y =  self.train_generator[1],
                                        epochs=self.epochs,
                                        validation_data=self.val_generator,
                                        verbose=1, 
                                        callbacks=self._get_callbacks(self.keras_model))
                except KeyboardInterrupt:
                    print("Finalizando o treinamento a pedido do usuário...")

                history_df = pd.read_csv(get_history_path(self.output_path))

                plot_history(history_df).savefig(get_history_plot_path(self.output_path))

            finally:
                K.clear_session()

    def _get_loss_function(self) -> Callable:
        return KERAS_LOSS_FUNCTIONS[self.loss_function]

    def _get_optimizer(self) -> KerasOptimizer:
        return KERAS_OPTIMIZERS[self.optimizer](lr=self.learning_rate,
                                                **self.optimizer_params)

    def _get_callbacks(self, keras_model: KerasModel) -> List[KerasCallback]:
        tensorboard_callback = KerasTensorBoard(get_tensorboard_logdir(self.task_id), batch_size=self.batch_size)
        callbacks = [*self._get_extra_callbacks(keras_model, tensorboard_callback),
                     KerasModelCheckpoint(get_keras_weights_path(self.output_path), save_best_only=True,
                                          monitor=self.monitor_metric, mode=self.monitor_mode),
                     KerasEarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta,
                                        monitor=self.monitor_metric, mode=self.monitor_mode, verbose=True),
                     KerasCSVLogger(get_history_path(self.output_path)),
                     tensorboard_callback]
        if self.lr_scheduler:
            callbacks.append(KERAS_LR_SCHEDULERS[self.lr_scheduler](**self.lr_scheduler_params))
        return callbacks

    def _get_extra_callbacks(self, keras_model: KerasModel, tensorboard_callback: KerasTensorBoard) -> List[
        KerasCallback]:
        return []

    def get_train_generator(self):
        df = self.train_dataset
        
        df_Y   = df['0'] 
        df_X = df.drop(['0'], axis=1)

        return (df_X, df_Y)

    def get_val_generator(self):
        df = self.val_dataset

        df_Y = df['0'] 
        df_X = df.drop(['0'], axis=1)

        return (df_X, df_Y)

    def get_test_generator(self):
        return ImageIterator(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, seed=self.seed)

    @property
    def train_generator(self):
        if not hasattr(self, "_train_generator"):
            self._train_generator = self.get_train_generator()
        return self._train_generator

    @property
    def val_generator(self):
        if not hasattr(self, "_val_generator"):
            self._val_generator = self.get_val_generator()
        return self._val_generator

    @property
    def test_generator(self):
        if not hasattr(self, "_test_generator"):
            self._test_generator = self.get_test_generator()
        return self._test_generator

    def get_trained_model(self) -> KerasModel:
        model = self.create_model()
        model.load_weights(get_keras_weights_path(self.output_path))
        return model

    @property
    def tensorflow_device(self) -> tf.device:
        if self.device == "cuda":
            return tf.device(f"/device:GPU:{self.device_id}")
        else:
            return tf.device(f"/cpu:0")


class ClassifierWithTransferLearningKerasModelTraining(BaseKerasModelTraining):
    __metaclass__ = abc.ABCMeta

    frozen_layers = luigi.IntParameter(default=9)

    @abc.abstractmethod
    def create_base_model(self) -> KerasModel:
        """Retornar um modelo pré-treinado do Keras"""
        pass

    @abc.abstractmethod
    def create_model_with(self, base_model: KerasModel) -> KerasModel:
        """Retornar um modelo utilizando o modelo pré-treinado passado como parâmetro"""
        pass

    def create_model(self) -> KerasModel:
        self.base_model = self.create_base_model()

        for layer in self.base_model.layers[:self.frozen_layers]:
            layer.trainable = False

        self.keras_model = self.create_model_with(self.base_model)

        return self.keras_model

    def after_train(self):
        model = self.get_trained_model()
        # self.eval_thresholds(model)
        # test_probas = pred_probas_for_classifier(model, self.test_generator)
        # for score_threshold in self.score_thresholds_to_eval:
        #     self.generate_submission_file(probas=test_probas, threshold=score_threshold)

    # def eval_thresholds(self, model: KerasModel = None) -> pd.DataFrame:
    #     with self.tensorflow_device:
    #         if model is None:
    #             model = self.get_trained_model()

    #         results = []
    #         train_probas = pred_probas_for_classifier(model, self.train_generator)
    #         val_probas = pred_probas_for_classifier(model, self.val_generator)

    #         for score_threshold in self.score_thresholds_to_eval:
    #             train_report = evaluate_classifier(self.train_generator, probas=train_probas,
    #                                                filenames=self.train_dataset.im_list,
    #                                                ground_truths=self.train_dataset.labels, threshold=score_threshold)
    #             val_report = evaluate_classifier(self.val_generator, probas=val_probas,
    #                                              filenames=self.val_dataset.im_list,
    #                                              ground_truths=self.val_dataset.labels, threshold=score_threshold)

    #             results.append(
    #                 OrderedDict(threshold=score_threshold, acc=train_report.acc, precision=train_report.precision,
    #                             recall=train_report.recall, f1_score=train_report.f1_score, val_acc=val_report.acc,
    #                             val_precision=val_report.precision, val_recall=val_report.recall,
    #                             val_f1_score=val_report.f1_score,))


    #             plot_confusion_matrix(train_report.confusion_matrix, CLASS_NAMES)\
    #                 .savefig(os.path.join(self.output_path, "confusion_matrix_%.2f.png" % score_threshold))
    #             plot_confusion_matrix(val_report.confusion_matrix, CLASS_NAMES) \
    #                 .savefig(os.path.join(self.output_path, "val_confusion_matrix_%.2f.png" % score_threshold))

    #         df = pd.DataFrame(results)
    #         df.to_csv(os.path.join(self.output_path, "eval_thresholds.csv"), index=False)
    #         return df

    def generate_submission_file(self, model: KerasModel = None, probas: np.ndarray = None, threshold=0.5):
        pass
        # with self.tensorflow_device:
        #     if probas is None:
        #         if model is None:
        #             model = self.get_trained_model()
        #         probas = pred_probas_for_classifier(model, self.test_generator)

        #     test_df = self.get_test_df()
        #     test_df = test_df[["PatientID"]]
        #     test_df.columns = ["patientId"]
        #     test_df["HasPnemonia"] = (probas > threshold).astype(int)

        #     test_df.to_csv(os.path.join(self.output_path, "submission_%.2f.csv" % threshold), index=False)


def load_keras_model_from_task_dir(model_cls: Type[BaseKerasModelTraining], task_dir: str) -> BaseKerasModelTraining:
    model_training = model_cls(**get_params(task_dir))
    model_training._output_path = task_dir
    return model_training


def load_keras_model(model_cls: Type[BaseKerasModelTraining], task_id: str) -> BaseKerasModelTraining:
    task_dir = get_task_dir(model_cls, task_id)

    return load_keras_model_from_task_dir(model_cls, task_dir)
