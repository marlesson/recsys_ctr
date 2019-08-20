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

TRAIN_FILE  = "ctr_criteo_labs_train_sample.txt"


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

    # def requires(self):
    #    return None

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return (luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "train_%.2f_%d_%s.csv" % (
                                                   self.val_size, self.seed, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "val_%.2f_%d_%s.csv" % (self.val_size, self.seed, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, "test.csv")))

    def run(self):
        df = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_FILE), sep='\t', header=None)
        df.columns = DATASET_COLUMNS
        
        df = self.preprocess(df)

        train_df, val_df = train_test_split(df, test_size=self.val_size, random_state=self.seed)

        train_df.to_csv(self.output()[0].path, sep=";", index=False)
        val_df.to_csv(self.output()[1].path, sep=";", index=False)


    def preprocess(self, df):
        print(df.info())
        # encoder = ce.OneHotEncoder(cols=list(df.select_dtypes(include=['object']).columns),
        #                             use_cat_names=True, drop_invariant=True )

        encoder = ce.OrdinalEncoder(cols=list(df.select_dtypes(include=['object']).columns))
        df_t    = encoder.fit_transform(df)

        return df_t
