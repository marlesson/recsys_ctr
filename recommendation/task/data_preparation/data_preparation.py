# Criteo AI Lab
#
# https://www.kaggle.com/c/criteo-display-ad-challenge
# https://ailab.criteo.com/kaggle-contest-dataset-now-available-academic-use/
#

from typing import List, Tuple
import luigi
import os
import math
import pandas as pd
import numpy as np
import requests
import zipfile

#from imblearn.over_sampling.random_over_sampler import RandomOverSampler
#from imblearn.under_sampling.prototype_selection.random_under_sampler import RandomUnderSampler
from sklearn.model_selection import train_test_split

SEED = 42

DATASET_DIR = "output/dataset"

TRAIN_FILE  = "train.txt"
TEST_FILE   = "test.txt"

# class DownloadDataset(luigi.Task):
#     def output(self):
#         return luigi.LocalTarget("output/dataset_rsna_jpg.zip")

#     def run(self):
#         # Streaming, so we can iterate over the response.
#         r = requests.get("http://gxserver.heurys.com.br/rsna/dataset_rsna_jpg.zip", stream=True)
#         output_path = self.output().path

#         # Total size in bytes.
#         total_size = int(r.headers.get('content-length', 0))
#         block_size = 1024
#         wrote = 0
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'wb') as f:
#             for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
#                              unit_scale=True):
#                 wrote = wrote + len(data)
#                 f.write(data)
#         if total_size != 0 and wrote != total_size:
#             raise ConnectionError("ERROR, something went wrong")


# class UnzipDataset(luigi.Task):
#     def requires(self):
#         return DownloadDataset()

#     def output(self):
#         return luigi.LocalTarget(DATASET_DIR)

#     def run(self):
#         with zipfile.ZipFile(self.input().path, "r") as zip_ref:
#             zip_ref.extractall(self.output().path)


DATASET_COLUMNS = ['TARGET', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                    'C1', 'C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17',
                    'C18','C19','C20','C21','C22','C23','C24','C25','C26']

import category_encoders as ce

class PrepareDataFrames(luigi.Task):
    val_size: float = luigi.FloatParameter(default=0.2)
    seed: int = luigi.IntParameter(default=42)
    no_cross_columns: bool = luigi.BoolParameter(default=False)

    # def requires(self):
    #    return None

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return (luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "train_%.2f_%d_%d_%s.csv" % (
                                                   self.val_size, self.seed, self.no_cross_columns, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "val_%.2f_%d_%d_%s.csv" % (
                                                    self.val_size, self.seed, self.no_cross_columns, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "test_%.2f_%d_%d_%s.csv" % (
                                                    self.val_size, self.seed, self.no_cross_columns, task_hash))))

    def run(self):
        # Train Dataset
        train_df = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_FILE), sep='\t', header=None)
        train_df.columns   = DATASET_COLUMNS
        
        # Test Dataset
        test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE), sep='\t', header=None)
        test_df.columns    = DATASET_COLUMNS[1:]
        
        # Preprocess Datasets
        train_df, test_df  = self.preprocess(train_df, test_df)

        # Sprint Val Dataset
        train_df, val_df   = train_test_split(train_df, test_size=self.val_size, random_state=self.seed)

        train_df.to_csv(self.output()[0].path, sep=";", index=False)
        val_df.to_csv(self.output()[1].path, sep=";", index=False)
        test_df.to_csv(self.output()[2].path, sep=";", index=False)
        

    def preprocess(self, df, test_df = None):
        print(df.describe(include = ['object', 'float', 'int']))

        categorical_columns = list(df.select_dtypes(include=['object']).columns)

        if not self.no_cross_columns:
            # my understanding on how to replicate what layers.crossed_column does. One
            # can read here: https://www.tensorflow.org/tutorials/linear.
            df = self.cross_columns(df, categorical_columns)
            
        # Encoder categorical Columns
        #
        df, test_df = self.encoder_categorical_columns(df, test_df)

        print(df.info())

        return df, test_df

    def encoder_categorical_columns(self, df, test_df = None):
        """
        Encoder Categorical Columns
        """

        # Categorical Columns After Cross
        categorical_columns = list(df.select_dtypes(include=['object']).columns)

        # encoder = ce.OneHotEncoder(cols=list(df.select_dtypes(include=['object']).columns),
        #                             use_cat_names=True, drop_invariant=True )

        encoder = ce.OrdinalEncoder(cols=categorical_columns)
        df_t    = encoder.fit_transform(df[DATASET_COLUMNS[1:]])

        if test_df is not None:
            test_df = encoder.transform(test_df)

        df_t['TARGET'] = df['TARGET']

        return df_t, test_df

    def cross_columns(self, df, x_cols):
        """simple helper to build the crossed columns in a pandas dataframe
        """
        for c1 in x_cols:
            for c2 in x_cols:
                if c1 != c2:
                    df["{}_{}".format(c1, c2)] = df.apply(lambda row: "{}_{}".format(row[c1], row[c2]), axis=1)

        return df

