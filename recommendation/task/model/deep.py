from typing import List, Tuple

import luigi
import keras
from keras import applications
from keras.layers import Dropout, Dense, Input
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Flatten, concatenate, Lambda, Dropout
from keras.layers import Input, concatenate, Embedding, Reshape
import pandas as pd
from recommendation.task.model.base import ClassifierWithTransferLearningKerasModelTraining, KERAS_ACTIVATION_FUNCTIONS, KERAS_WEIGHT_INIT
from keras.regularizers import l2, l1_l2

# - Para múltiplas GPUs:
class DeepRecommender(ClassifierWithTransferLearningKerasModelTraining):
    input_shape: Tuple[int, int] = luigi.TupleParameter(default=(100,))
    batch_size: int = luigi.IntParameter(default=10)
    learning_rate = luigi.FloatParameter(default=1e-5)
    dense_layers: List[int] = luigi.ListParameter(default=[512, 512])
    dropout: float = luigi.FloatParameter(default=None)
    activation_function: str= luigi.ChoiceParameter(choices=KERAS_ACTIVATION_FUNCTIONS.keys(), default="relu")
    kernel_initializer: str = luigi.ChoiceParameter(choices=KERAS_WEIGHT_INIT.keys(), default="glorot_uniform")

    EMBEDDINGS_COLS = ['C1', 'C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17',
                        'C18','C19','C20','C21','C22','C23','C24','C25','C26']

    CONT_COLS       = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
    

    def create_base_model(self) -> Model:
        unique_vals     = self.build_unique_vals(DeepRecommender.EMBEDDINGS_COLS)


        #  Build Imputs Embs + Cout
        embeddings_tensors = []
        n_factors = 5
        reg       = 1e-3

        for ec in DeepRecommender.EMBEDDINGS_COLS:
            layer_name = ec + '_inp'
            t_inp, t_build = self.embedding_input(
                layer_name, unique_vals[ec], n_factors, reg)
            embeddings_tensors.append((t_inp, t_build))
            del(t_inp, t_build)

        continuous_tensors = []
        for cc in DeepRecommender.CONT_COLS:
            layer_name = cc + '_in'
            t_inp, t_build = self.continous_input(layer_name)
            continuous_tensors.append((t_inp, t_build))
            del(t_inp, t_build)

        # Input
        inp_layer =  [et[0] for et in embeddings_tensors]
        inp_layer += [ct[0] for ct in continuous_tensors]
        
        # Emb, Reshap
        inp_embed =  [et[1] for et in embeddings_tensors]
        inp_embed += [ct[1] for ct in continuous_tensors]

        # Concatenated Embeddings
        embs_concat = concatenate(inp_embed)
        embs_concat = Flatten()(embs_concat)

        deep   = Dense(1024, activation='relu', kernel_initializer=self.kernel_initializer)(embs_concat)
        deep   = Dense(512,  activation='relu', kernel_initializer=self.kernel_initializer)(deep)
        deep   = Dense(256,  activation='relu', kernel_initializer=self.kernel_initializer)(deep)

        output = Dense(1,  activation='sigmoid', kernel_initializer=self.kernel_initializer)(deep)
        
        model  = Model(inp_layer, output, name='Deep')

        return model

    def build_unique_vals(self, columns):
        train_df = self.train_dataset[columns]
        val_df   = self.val_dataset[columns]

        df       = pd.concat([train_df, val_df])

        unique_vals = {}

        for c in columns:
            unique_vals[c] = len(df[c].unique())

        return unique_vals

    def embedding_input(self, name, n_in, n_out, reg):
        inp = Input(shape=(1,), dtype='int64', name=name)
        return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)


    def continous_input(self, name):
        inp = Input(shape=(1,), dtype='float32', name=name)
        return inp, Reshape((1, 1))(inp)        

    def get_train_generator(self):
        df   = self.train_dataset
        
        df_Y = df['TARGET'] 
        df_X = df.drop(['TARGET'], axis=1)

        columns = DeepRecommender.EMBEDDINGS_COLS+DeepRecommender.CONT_COLS
        return ([df_X[c] for c in columns], df_Y)

    def get_val_generator(self):
        df   = self.val_dataset

        df_Y = df['TARGET'] 
        df_X = df.drop(['TARGET'], axis=1)

        columns = DeepRecommender.EMBEDDINGS_COLS+DeepRecommender.CONT_COLS
        return ([df_X[c] for c in columns], df_Y)

    def get_test_generator(self):
        df_X   = self.test_dataset

        columns = DeepRecommender.EMBEDDINGS_COLS+DeepRecommender.CONT_COLS
        return [df_X[c] for c in columns]

    def create_model_with(self, base_model: Model) -> Model:
        return base_model            