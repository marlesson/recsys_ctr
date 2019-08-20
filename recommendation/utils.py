import numpy as np
import pandas as pd


def merge_submission_files(classification_submission_file: str, localization_submission_file: str) -> pd.DataFrame:
    clf_df = pd.read_csv(classification_submission_file)
    loc_df = pd.read_csv(localization_submission_file)

    loc_df.loc[clf_df.HasPnemonia == 0, "PredictionString"] = np.nan
    return loc_df