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

def split_train_test(filename: str, train_ratio: float, val_ratio: float, train_col: list)-> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Split for training, validation and test dataset
    Args:
        filename: The filename of training data file
        train_ratio: The ratio for splitting into training data
        val_ratio: The ratio for splitting into validation data
        test_ratio: The ratio for splitting into test data
        train_cols: The columns used for training
    Returns:
        The train dataframe and test dataframe
    """

    df = pd.read_csv(filename)

    N = len(df)

    train2val = train_ratio + val_ratio

    traindf = df[0: int(N*train_ratio)]
    valdf = df[int(N*train_ratio):int(N*train2val)]
    testdf = df[int(N*train2val):]

    traindf = setup_trainFeature(traindf, train_col)
    valdf = setup_trainFeature(valdf, train_col)
    testdf = setup_trainFeature(testdf, train_col)

    return traindf, valdf, testdf

def normalize(traindf: pd.DataFrame, valdf: pd.DataFrame, testdf: pd.DataFrame):

    """
    z score transform
    Args:
        traindf: training dataframe
        valdf: validation dataframe
        testdf: test dataframe

    Returns:
        The values in all of columns are transformed into z_score
    """

    # The mean and standard deviation should only be computed using the training data
    # so that the models have no access to the values in the validation and test sets.

    train_mean = traindf.mean()
    train_std = traindf.std()

    traindf = (traindf - train_mean)/train_std
    valdf = (valdf - train_mean) / train_std
    testdf = (testdf - train_mean) / train_std

    return traindf, valdf, testdf


















