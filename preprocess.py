import pandas as pd
from scipy.stats import zscore


def setup_trainFeature(df: pd.DataFrame, training_col: list)-> pd.DataFrame:

    """
    Get the training dataframe from file
    Args:
        training_col: the features for trainign model

    Returns:
        The dataframe of training data
    """

    return df.loc[:, training_col]

def split_train_test(filename: str, train_ratio: float, val_ratio: float)-> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Split for training, validation and test dataset
    Args:
        filename: The filename of training data file

    Returns:
        The train dataframe and test dataframe
    """

    df = pd.read_csv(filename)

    N = len(df)

    train2val = train_ratio + val_ratio

    traindf = df[0: int(N*train_ratio)]
    valdf = df[int(N*train_ratio):int(N*train2val)]
    testdf = df[int(N*train2val):]

    return traindf, valdf, testdf

def calc_zscore(df: pd.DataFrame):

    """
    z score transform
    Args:
        df: pandans dataframe

    Returns:
        The values in all of columns are transformed into z_score
    """

    for key in df.keys():
        df[key] = zscore(df[key])

    return df
















# preprocess training data


# generate training batch size sample randomly