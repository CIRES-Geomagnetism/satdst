import pandas as pd

def setup_trainFeature(df: pd.DataFrame, training_col: list)-> pd.DataFrame:

    """
    Get the training dataframe from file
    Args:
        training_col: the features for trainign model

    Returns:
        The dataframe of training data
    """



    return df.loc[:, training_col]

def split_train_test(filename: str)-> tuple [pd.DataFrame, pd.DataFrame]:

    """
    Get all the training data except for 2002-2003. The original codes use 2002-2003 for test data
    Args:
        filename: The filename of training data file

    Returns:
        The train dataframe and test dataframe
    """

    df = pd.read_csv(filename)
    traindf_front = df.loc[(df['Decyear'] < 2002.0)]
    traindf_back = df.loc[(df['Decyear'] > 2003.0)]

    traindf = traindf_front + traindf_back

    testdf = df.loc[(df['Decyear'] >= 2002.0) & (df['Decyear'] <= 2003.0)]

    return traindf, testdf







# preprocess training data

# z score transform
# generate training batch size sample randomly