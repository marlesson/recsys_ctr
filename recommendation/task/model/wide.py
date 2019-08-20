from typing import List, Tuple

import luigi
import keras
from keras import applications
from keras.layers import Dropout, Dense, Input
from keras.layers import BatchNormalization
from keras.models import Sequential, Model

from recommendation.task.model.base import ClassifierWithTransferLearningKerasModelTraining, KERAS_ACTIVATION_FUNCTIONS, KERAS_WEIGHT_INIT

# - Para múltiplas GPUs:
class WideRecommender(ClassifierWithTransferLearningKerasModelTraining):
    input_shape: Tuple[int, int] = luigi.TupleParameter(default=(100,))
    batch_size: int = luigi.IntParameter(default=10)
    learning_rate = luigi.FloatParameter(default=1e-5)
    dense_layers: List[int] = luigi.ListParameter(default=[512, 512])
    dropout: float = luigi.FloatParameter(default=None)
    activation_function: str= luigi.ChoiceParameter(choices=KERAS_ACTIVATION_FUNCTIONS.keys(), default="relu")
    kernel_initializer: str = luigi.ChoiceParameter(choices=KERAS_WEIGHT_INIT.keys(), default="glorot_uniform")

    def create_base_model(self) -> Model:
        x_input = Input(shape=self.input_shape, name='wide_inp')

        wide    = Dense(self.input_shape[0], 
                        activation=self.activation_function, 
                        kernel_initializer=self.kernel_initializer,
                        name='wide_mlp')(x_input)
        
        output = Dense(1, activation='sigmoid', 
                          kernel_initializer=self.kernel_initializer)(wide)


        model  = Model(x_input, output, name='Wide')


        return model

    def create_model_with(self, base_model: Model) -> Model:
        return base_model        